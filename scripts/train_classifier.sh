#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 检查传入的参数数量
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name=$1

# 根据base_model_name的值设置base_model_names数组
if [ "$dataset_name" == "all" ]; then
    dataset_names=("yelp" "agnews" "dailydialog")
else
    dataset_names=("$dataset_name")
fi

for dataset_name in "${dataset_names[@]}"; do
    echo "Training on dataset_name: $dataset_name"
    deepspeed train_classifier.py \
      --dataset_name ${dataset_name} \
      --learning_rate 1e-5 \
      --global_batch_size 64 \
      --mini_batch_size 8
done