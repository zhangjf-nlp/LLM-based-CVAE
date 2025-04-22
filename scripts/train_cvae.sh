#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 检查传入的参数数量
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <base_model_name> <dataset_name> <cvae_model_path> <vae_type> <skip> <ghost>"
    exit 1
fi

base_model_name=$1
dataset_name=$2
cvae_model_path=$3
vae_type=$4
skip=$5
ghost=$6

# 根据base_model_name的值设置base_model_names数组
if [ "$base_model_name" == "all" ]; then
    base_model_names=("gpt2" "gpt2-large" "gpt2-xl")
else
    base_model_names=("$base_model_name")
fi

if [ "$dataset_name" == "all" ]; then
    dataset_names=("yelp" "agnews" "dailydialog")
else
    dataset_names=("$dataset_name")
fi

for base_model_name in "${base_model_names[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "Training ${base_model_name} on ${dataset_name}"
        deepspeed train_cvae.py \
            --base_model_name ${base_model_name} \
            --dataset_name ${dataset_name} \
            --cvae_model_path ${cvae_model_path} \
            --vae_type ${vae_type} \
            --add_skip_connection ${skip} \
            --add_ghost_skip_connection ${ghost} \
            --learning_rate 1e-5
    done
done