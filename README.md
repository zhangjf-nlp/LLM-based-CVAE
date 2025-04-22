# LLM-based Conditional Variational Autoencoder (CVAE)

## Quick Start Guide

### Prerequisites
- Linux operating system (recommended for optimal DeepSpeed compatibility)
- Python3 + torch-2.1.2+cu118
- Valid Hugging Face access token (required for Llama3 download)

### 1. Dataset Preparation

#### AGNews Dataset
```bash
git clone https://huggingface.co/datasets/fancyzhx/ag_news
```

#### DailyDialog Dataset
```bash
git clone https://huggingface.co/datasets/roskoN/dailydialog
```

#### Yelp Dataset
```bash
git clone https://github.com/fangleai/Implicit-LVM.git
mv Implicit-LVM/lang_model_yelp/data yelp
```

### 2. Model Download (Llama3-8B)
```bash
# Requires Meta Llama3 access approval: https://huggingface.co/settings/tokens
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B
mv Meta-Llama-3-8B llama3-8b # Simplified directory name for convenience
```

### 3. Environment Setup
```bash
# We suggest running on Linux
# DeepSpeed may be not well-suited on other OS
python3 -m pip install -r requirements.txt
```

### Training Pipeline

#### Supervised Fine-Tuning (SFT)
```bash
deepspeed train_sft.py --dataset agnews
```

#### One-to-Many Sampling
Note: The following code requires vLLM, but vLLM may be incompatible with DeepSpeed.
```bash
# Parallel execution recommended for different splits
num_splits=10
for ((index_split=1; index_split<=num_splits; index_split++))
do
    python3 vllm_one2many_inference.py \
        --num_splits ${num_splits} \
        --index_split ${index_split}
done
```

#### CVAE Fine-Tuning
```bash
# Default configuration uses our proposed DG-CVAE with G-Skip
deepspeed train_cvae.py --dataset agnews --vae_type DG-VAE --add_gskip_connection 1
```

#### CVAE Performance Testing
```bash
# Include identical fine-tuning arguments & test-specific options
deepspeed train_cvae.py --dataset agnews --test_prior_inference --small_test
```

We have also provided some demo scripts in ./scripts. Don't hesitate to contact me or open an issue when you have any problems!
