# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import regex
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    """
    将文本段落分割成句子列表，根据指定的语言和长度限制进行处理
    
    Args:
        text (str): 输入的文本
        tokenize: 分词函数，用于非中文文本的长度计算
        lang (str): 语言类型，默认为"zh"（中文）
        token_max_n (int): 每句话的最大token数
        token_min_n (int): 每句话的最小token数
        merge_len (int): 合并阈值，小于该长度的句子会被合并到前一句
        comma_split (bool): 是否使用逗号作为分割符
        
    Returns:
        list: 分割后的句子列表
    """
    
    def calc_utt_length(_text: str):
        """计算文本长度，中文按字符数计算，其他语言按分词结果计算"""
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        """判断是否应该将短句合并到前一句"""
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    # 根据语言确定句子结束标点符号
    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:
        pounc = ['.', '?', '!', ';', ':']
    
    # 如果允许用逗号分割，则添加逗号到标点符号列表
    if comma_split:
        pounc.extend(['，', ','])

    # 确保文本以标点符号结尾
    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    # 按标点符号分割文本，得到初步的句子列表
    st = 0  # 句子起始位置
    utts = []  # 存储分割后的句子
    for i, c in enumerate(text):
        if c in pounc:
            # 如果当前标点前有内容，则将其作为一个句子添加
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            # 处理引号情况：如果标点后是引号，则将引号加入到最后一个句子中
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)  # 取出最后一个句子
                utts.append(tmp + text[i + 1])  # 将引号添加到句子末尾
                st = i + 2  # 更新下一个句子的起始位置
            else:
                st = i + 1  # 更新下一个句子的起始位置

    # 根据最大最小长度要求合并句子
    final_utts = []  # 最终的句子列表
    cur_utt = ""  # 当前正在构建的句子
    for utt in utts:
        # 如果当前句子加上新句子超过最大长度，且当前句子已达到最小长度，则保存当前句子
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt  # 将新句子添加到当前句子中
    
    # 处理剩余的句子
    if len(cur_utt) > 0:
        # 如果剩余句子太短且最终句子列表不为空，则将其合并到最后一句中
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            # 否则作为独立句子添加
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))
