# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging, load_wav
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:
    """
    CosyVoice TTS模型的主要接口类
    
    该类封装了CosyVoice模型的初始化、推理和管理功能，支持多种TTS模式：
    1. SFT模式（Speaker Fine-Tuning）：使用预训练的说话人
    2. Zero-shot模式：通过提示语音快速克隆说话人
    3. Cross-lingual模式：跨语言语音合成
    4. Instruct模式：指令引导的语音合成
    5. Voice conversion模式：语音转换
    """

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        """
        初始化CosyVoice模型
        
        Args:
            model_dir (str): 模型目录路径或ModelScope模型标识符
            load_jit (bool): 是否加载JIT编译优化的模型组件，默认False
            load_trt (bool): 是否加载TensorRT优化的模型组件，默认False
            fp16 (bool): 是否使用FP16精度推理，默认False
            trt_concurrent (int): TensorRT并发执行实例数，默认1
        """
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        """
        列出所有可用的预训练说话人
        
        Returns:
            list: 可用说话人ID列表
        """
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        """
        添加一个新的零样本说话人
        
        Args:
            prompt_text (str): 提示文本
            prompt_speech_16k (Tensor): 提示语音（16kHz采样率）
            zero_shot_spk_id (str): 新说话人的唯一标识符
            
        Returns:
            bool: 添加成功返回True
        """
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_speech_16k, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        """
        保存当前说话人信息到磁盘
        """
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        """
        SFT（Speaker Fine-Tuning）模式推理
        
        使用预训练的说话人进行语音合成
        
        Args:
            tts_text (str): 待合成的文本
            spk_id (str): 预训练说话人ID
            stream (bool): 是否启用流式合成，默认False
            speed (float): 语音速度调节因子，默认1.0（正常速度）
            text_frontend (bool): 是否使用文本前端处理，默认True
            
        Yields:
            dict: 包含合成语音的字典，键为'tts_speech'
        """
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        """
        Zero-shot模式推理
        
        通过提示语音快速克隆说话人并合成语音
        
        Args:
            tts_text (str): 待合成的文本
            prompt_text (str): 提示文本，描述提示语音的内容
            prompt_speech_16k (Tensor): 提示语音（16kHz采样率）
            zero_shot_spk_id (str): 预定义的零样本说话人ID，可选
            stream (bool): 是否启用流式合成，默认False
            speed (float): 语音速度调节因子，默认1.0
            text_frontend (bool): 是否使用文本前端处理，默认True
            
        Yields:
            dict: 包含合成语音的字典，键为'tts_speech'
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        """
        Cross-lingual（跨语言）模式推理
        
        支持使用一种语言的提示语音合成另一种语言的文本
        
        Args:
            tts_text (str): 待合成的文本（可以是不同语言）
            prompt_speech_16k (Tensor): 提示语音（16kHz采样率）
            zero_shot_spk_id (str): 预定义的零样本说话人ID，可选
            stream (bool): 是否启用流式合成，默认False
            speed (float): 语音速度调节因子，默认1.0
            text_frontend (bool): 是否使用文本前端处理，默认True
            
        Yields:
            dict: 包含合成语音的字典，键为'tts_speech'
        """
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        """
        Instruct（指令引导）模式推理
        
        通过自然语言指令控制语音的情感、风格等特性
        
        Args:
            tts_text (str): 待合成的文本
            spk_id (str): 预训练说话人ID
            instruct_text (str): 指令文本，用于控制合成语音的风格
            stream (bool): 是否启用流式合成，默认False
            speed (float): 语音速度调节因子，默认1.0
            text_frontend (bool): 是否使用文本前端处理，默认True
            
        Yields:
            dict: 包含合成语音的字典，键为'tts_speech'
        """
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        """
        Voice Conversion（语音转换）模式推理
        
        将源语音转换为提示语音的音色
        
        Args:
            source_speech_16k (Tensor): 源语音（16kHz采样率）
            prompt_speech_16k (Tensor): 提示语音（16kHz采样率），决定输出音色
            stream (bool): 是否启用流式合成，默认False
            speed (float): 语音速度调节因子，默认1.0
            
        Yields:
            dict: 包含转换后语音的字典，键为'tts_speech'
        """
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):
    """
    CosyVoice2模型接口类，继承自CosyVoice
    
    CosyVoice2是CosyVoice的升级版本，具有更强的性能和新特性
    """

    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        """
        初始化CosyVoice2模型
        
        Args:
            model_dir (str): 模型目录路径或ModelScope模型标识符
            load_jit (bool): 是否加载JIT编译优化的模型组件，默认False
            load_trt (bool): 是否加载TensorRT优化的模型组件，默认False
            load_vllm (bool): 是否加载vLLM优化的LLM组件，默认False
            fp16 (bool): 是否使用FP16精度推理，默认False
            trt_concurrent (int): TensorRT并发执行实例数，默认1
        """
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def inference_instruct(self, *args, **kwargs):
        """
        在CosyVoice2中未实现传统instruct模式
        """
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        """
        CosyVoice2专用的增强版Instruct模式推理
        
        结合了指令控制和零样本语音克隆的能力
        
        Args:
            tts_text (str): 待合成的文本
            instruct_text (str): 指令文本，用于控制合成语音的风格
            prompt_speech_16k (Tensor): 提示语音（16kHz采样率）
            zero_shot_spk_id (str): 预定义的零样本说话人ID，可选
            stream (bool): 是否启用流式合成，默认False
            speed (float): 语音速度调节因子，默认1.0
            text_frontend (bool): 是否使用文本前端处理，默认True
            
        Yields:
            dict: 包含合成语音的字典，键为'tts_speech'
        """
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()