#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 检查传入的参数数量
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_model_name> <dataset_name>"
    exit 1
fi

base_model_name=$1
dataset_name=$2

# 根据base_model_name的值设置base_model_names数组
if [ "$base_model_name" == "all" ]; then
    base_model_names=("gpt2-xl" "gpt2-large" "gpt2")
else
    base_model_names=("$base_model_name")
fi

for base_model_name in "${base_model_names[@]}"; do
    echo "Training with base_model_name: $base_model_name"
    deepspeed train_sft.py \
      --base_model_name ${base_model_name} \
      --dataset_name ${dataset_name} \
      --learning_rate 1e-5 \
      --global_batch_size 64 \
      --mini_batch_size 8 \
      --epochs 1
done