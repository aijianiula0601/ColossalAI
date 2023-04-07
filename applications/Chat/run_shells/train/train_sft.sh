#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit


#--------------------------------------------------------------------
# 先构建docker镜像
# 进入docker目录下执行：
#   sudo docker build -t clossalai:v1 .
# transformer安装：
#   pip install git+https://github.com/huggingface/transformers
#--------------------------------------------------------------------

cd ../../../../

llama_model_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama/new_llama_7b"
model_name="llama"

save_base_dir="/mnt/cephfs/hjh/train_record/nlp/colossalAi"
train_output_dir="${save_base_dir}/train_stft_outputs"
save_path="${train_output_dir}/Coati-7B"
dataset_path="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/ft_52k/debug_alpaca_data_cleaned.json"

torchrun --standalone --nproc_per_node=4 applications/Chat/examples/train_sft.py \
  --pretrain ${llama_model_dir} \
  --model ${model_name} \
  --strategy colossalai_zero2 \
  --log_interval 10 \
  --save_path ${save_path} \
  --dataset ${dataset_path} \
  --batch_size 4 \
  --accimulation_steps 8 \
  --lr 2e-5 \
  --max_datasets_size 512 \
  --max_epochs 1
