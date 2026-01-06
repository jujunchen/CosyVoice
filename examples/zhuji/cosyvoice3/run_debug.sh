#!/bin/bash

set -e  # 如果有任何命令失败则退出

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 只使用第一张GPU
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name> [options]"
    echo "Example: $0 hift --gan"
    exit 1
fi

model=$1
shift

# 预训练模型目录
pretrained_model_dir="../../../pretrained_models/Fun-CosyVoice3-0.5B"

echo "Start debugging train.py with model: $model"
echo "Pretrained model dir: $pretrained_model_dir"

# 直接运行 train.py 而不是使用 torchrun
# 这样可以更容易地使用 Python 调试器
python -m pdb -c continue cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config conf/cosyvoice3.yaml \
    --train_data data/output/train.data.list \
    --cv_data data/output/test.data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --model $model \
    --checkpoint $pretrained_model_dir/$model.pt \
    --model_dir `pwd`/exp/cosyvoice3/$model/torch_ddp \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice3/$model/torch_ddp \
    --dist_backend nccl \
    --num_workers 1 \
    --prefetch 100 \
    --pin_memory \
    --use_amp \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer \
    "$@"

echo "Debugging completed."