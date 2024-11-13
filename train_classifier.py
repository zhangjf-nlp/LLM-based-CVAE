import os
import torch
import random
import numpy as np
from transformers import RobertaForSequenceClassification

from config import root
from utils import get_basic_parser, get_trainer, get_last_checkpoint, check_output_dir
from dataset import ClassificationDataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def set_default_output_dir(args):
    if args.output_dir is None:
        args.output_dir = os.path.join(root, f"cls/{args.base_model_name}-{args.dataset_name}")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    acc = (logits.argmax(axis=-1) == labels)
    return {
        "acc": acc.mean(),
    }


if __name__ == "__main__":
    parser = get_basic_parser(epochs=3, base_model_name="roberta-large")

    args = parser.parse_args()
    set_default_output_dir(args)
    check_output_dir(args)
    set_seed(args)

    train_dataset = ClassificationDataset(tokenizer_path="llama3-8b", dataset_name=args.dataset_name, usage="train")
    valid_dataset = ClassificationDataset(tokenizer_path="llama3-8b", dataset_name=args.dataset_name, usage="validation")
    model = RobertaForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=train_dataset.num_labels,
        type_vocab_size=2,
        ignore_mismatched_sizes=True
    )

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=train_dataset.classification_tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        remove_unused_columns=False,
        collate_fn=train_dataset.collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir))
    trainer.save_state()