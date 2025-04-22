import json
import os
import time

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM

from config import root
from utils import get_basic_parser, get_trainer, get_last_checkpoint, check_output_dir
from dataset import SFTDataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def set_default_output_dir(args):
    if args.output_dir is None:
        args.output_dir = os.path.join(root, f"sft/{args.base_model_name}-{args.dataset_name}")
    if args.epochs != 3:
        args.output_dir = args.output_dir + f"-ep{args.epochs}"
    if (args.global_batch_size, args.mini_batch_size) != (64, 8):
        args.output_dir = args.output_dir + f"-gbs{args.global_batch_size}-mbs{args.mini_batch_size}"


if __name__ == "__main__":
    parser = get_basic_parser(epochs=3)

    args = parser.parse_args()
    time.sleep(args.local_rank * 10)
    set_default_output_dir(args)
    check_output_dir(args)
    set_seed(args)

    train_dataset = SFTDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name, usage="train")
    valid_dataset = SFTDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name, usage="validation")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name, device_map=f"cuda:{args.local_rank}")

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=train_dataset.tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        remove_unused_columns=False,
        collate_fn=train_dataset.collate_fn
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir))
    trainer.save_state()