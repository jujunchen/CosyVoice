#!/bin/bash
# 载入路径和环境配置（定义数据目录、Python 环境等）
. ./path.sh || exit 1;

# 控制要运行的阶段（stage）范围：
# 从 stage 开始，一直到 stop_stage 结束，中间每个阶段都有独立的 if 块
stage=5
stop_stage=5

# LibriTTS 数据集下载地址
data_url=www.openslr.org/resources/60
# LibriTTS 数据存放目录（需要根据自己机器实际路径修改）
data_dir=/mnt/nvme0/projects/CosyVoice/examples/zhuji/cosyvoice3/data/input
# 预训练 CosyVoice3 模型目录
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B

# stage -1：下载 LibriTTS 原始数据
# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "Data Download"
#   # 下载不同子集：训练、开发、测试等
#   for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
#     local/download_and_untar.sh ${data_dir} ${data_url} ${part}
#   done
# fi

# stage 0：数据准备，生成 Kaldi 风格数据目录
# 主要生成 wav.scp/text/utt2spk/spk2utt 等文件
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train test; do
    mkdir -p data/output/$x
    # 使用本地脚本将原始 LibriTTS 转为所需格式，可选加 --instruct 标志
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/output/$x --instruct
  done
fi

# stage 1：提取说话人嵌入（campplus speaker embedding）
# 会在每个 data/$x 下生成 spk2embedding.pt 和 utt2embedding.pt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in train test; do
    tools/extract_embedding.py --dir data/output/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# stage 2：提取离散语音 token
# 会在每个 data/$x 下生成 utt2speech_token.pt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in train test; do
    tools/extract_speech_token.py --dir data/output/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx
  done
fi

# stage 3：准备训练所需的 parquet 格式数据
# 需要前面已经准备好 wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train test; do
    mkdir -p data/output/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --instruct \
      --src_dir data/output/$x \
      --des_dir data/output/$x/parquet
  done
fi

# ================== 下面是训练与模型导出相关部分 ==================

# 训练 LLM / flow / hifigan 所用 GPU 设置
export CUDA_VISIBLE_DEVICES="0"
# 解析可见 GPU 数量
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# 分布式训练的作业 ID（任意标识）
job_id=1986
# 分布式后端
dist_backend="nccl"
# DataLoader 相关参数
num_workers=2
prefetch=30
# 训练引擎，可选 torch_ddp / deepspeed 等
train_engine=torch_ddp

# stage 5：训练模型（当前注释写的是 “We only support llm training for now”，实际上循环里有 llm/flow/hifigan）
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  # 如果使用 deepspeed，则有独立的优化器配置
  if [ $train_engine = 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # 组合训练/开发集的 parquet 列表
  cat data/output/train/parquet/data.list > data/output/train.data.list
  cat data/output/test/parquet/data.list > data/output/test.data.list
  # NOTE: 这里提示后续会更新 llm/hift 训练
  for model in flow; do
    # 使用 torchrun 启动分布式训练
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data data/output/train.data.list \
      --cv_data data/output/test.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice3/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice3/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# stage 6：对训练好的模型做参数平均，生成最终 checkpoint
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in flow hift; do
    decode_checkpoint=`pwd`/exp/cosyvoice3/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice3/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

# stage 7：导出推理用模型（JIT 与 ONNX），用于加速推理
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi