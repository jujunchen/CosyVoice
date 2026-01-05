# Copyright (c) 2020 Mobvoi Inc (Di Wu)
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
import argparse
import glob

import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    val_scores = []
    if args.val_best:
        # 获取所有yaml配置文件，这些文件包含模型训练的指标信息
        yamls = glob.glob('{}/*.yaml'.format(args.src_path))
        # 过滤掉训练初期的文件（train或init开头的）和初始模型
        yamls = [
            f for f in yamls
            if not (os.path.basename(f).startswith('train')
                    or os.path.basename(f).startswith('init'))
        ]
        # 从yaml文件中提取验证指标信息：epoch、step、loss和tag
        for y in yamls:
            with open(y, 'r') as f:
                dic_yaml = yaml.load(f, Loader=yaml.BaseLoader)
                loss = float(dic_yaml['loss_dict']['loss'])  # 验证集上的损失值
                epoch = int(dic_yaml['epoch'])  # 训练的轮数
                step = int(dic_yaml['step'])  # 训练的步数
                tag = dic_yaml['tag']  # 模型标签
                val_scores += [[epoch, step, loss, tag]]
        # 根据损失值(loss)进行排序，选择损失值最低的前num个模型
        # reverse=False 表示升序排列，即loss最小的在前面
        sorted_val_scores = sorted(val_scores,
                                   key=lambda x: x[2],
                                   reverse=False)
        print("best val (epoch, step, loss, tag) = " +
              str(sorted_val_scores[:args.num]))
        # 根据排序结果，生成对应模型文件的路径列表
        # 模型文件名格式为 'epoch_{}_whole.pt'.format(score[0])，其中score[0]是epoch数
        path_list = [
            args.src_path + '/epoch_{}_whole.pt'.format(score[0])
            for score in sorted_val_scores[:args.num]
        ]
    print(path_list)
    avg = {}
    num = args.num
    assert num == len(path_list)
    # 加载所有选中的模型文件，并对参数进行累加
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            # 'step'和'epoch'等训练状态信息不需要平均，仅对模型参数进行平均
            if k not in ['step', 'epoch']:
                if k not in avg.keys():
                    avg[k] = states[k].clone()
                else:
                    avg[k] += states[k]
    # 对累加的参数求平均值，实现模型平均
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == '__main__':
    main()