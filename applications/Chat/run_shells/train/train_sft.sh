#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit


#--------------------------------------------------------------------
# 切换环境
# conda activate colossalai
# transformer安装(报错：cannot import name 'LlamaConfig' from 'transformers')：
#   pip install git+https://github.com/huggingface/transformers
# 问题：
#   v100 32G 显存，batch_size=2都无法训练起来，而且它不支持deepspeech优化
#--------------------------------------------------------------------

cd ../../../../

llama_model_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama/for_colossalai_llama_7b"
model_name="llama"

save_base_dir="/mnt/cephfs/hjh/train_record/nlp/colossalAi"
train_output_dir="${save_base_dir}/train_stft_outputs"
save_path="${train_output_dir}/Coati-7B"
dataset_path="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/ft_52k/alpaca_data_cleaned.json"

torchrun --standalone --nproc_per_node=4 applications/Chat/examples/train_sft.py \
  --pretrain ${llama_model_dir} \
  --model ${model_name} \
  --strategy colossalai_zero2 \
  --log_interval 10 \
  --save_path ${save_path} \
  --dataset ${dataset_path} \
  --batch_size 2 \
  --accimulation_steps 8 \
  --lr 2e-5 \
  --max_datasets_size 10000 \
  --max_epochs 1
