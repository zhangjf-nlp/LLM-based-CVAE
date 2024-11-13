import os
import time

import torch
import random
import numpy as np

from config import root
from utils import get_basic_parser, get_trainer, get_last_checkpoint, check_output_dir, get_last_checkpoint_at_root
from dataset import One2ManyDataset
from models import AutoModelForCVAE
from transformers import TrainerCallback

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def compute_metrics(eval_preds):
    loss_kld, loss_lm, loss_contra = eval_preds.predictions
    result = {
        "loss_kld": loss_kld.mean(),
        "loss_lm": loss_lm.mean(),
        "loss_contra": loss_contra.mean(),
    }
    return result


def set_default_output_dir(args):
    if args.output_dir is None:
        model_feature = (f"{args.num_latent_encoder_layers}_enc-"
                         #f"1024_dim_z-"
                         f"{args.frozen_pretrained}_frozen-"
                         f"{args.use_standard_prior}_stdprior-"
                         f"{args.vae_type}-"
                         f"{args.total_many}_total_{args.mini_many}_mini-"
                         f"{args.add_contra_loss}_contra-"
                         f"{args.add_skip_connection}_skip-"
                         f"{args.add_ghost_skip_connection}_ghost-"
                         f"{args.epochs}_epochs-"
                         f"{args.learning_rate}_lr-all-checkpoints")
        if args.cvae_model_path is not None:
            from_steps = args.cvae_model_path.split("-")[-1]
            model_feature = model_feature + f"-ft-{from_steps}"
        else:
            model_feature = model_feature + "-pt"
        args.output_dir = os.path.join(root, f"cvae/{args.base_model_name}-{args.dataset_name}/{model_feature}")


if __name__ == "__main__":
    parser = get_basic_parser(epochs=2, global_batch_size=16, mini_batch_size=1, learning_rate=1e-5)
    parser.add_argument('--cvae_model_path', type=get_last_checkpoint_at_root, default=None)
    parser.add_argument('--num_latent_encoder_layers', type=int, default=2)
    parser.add_argument('--frozen_pretrained', type=int, default=1)
    parser.add_argument('--use_standard_prior', type=int, default=1)
    parser.add_argument('--vae_type', type=str, default="DG-VAE")
    parser.add_argument('--add_contra_loss', type=int, default=0)
    parser.add_argument('--add_skip_connection', type=int, default=1)
    parser.add_argument('--add_ghost_skip_connection', type=int, default=0)
    parser.add_argument('--total_many', type=int, default=32)
    parser.add_argument('--mini_many', type=int, default=16)

    args = parser.parse_args()
    assert args.epochs == args.total_many // args.mini_many
    assert args.n_devices * (args.total_many // args.mini_many) == args.global_batch_size
    time.sleep(args.local_rank * 10)
    set_default_output_dir(args)
    check_output_dir(args)
    set_seed(args)

    train_dataset = One2ManyDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name, usage="train", total_many=args.total_many, mini_many=args.mini_many)
    valid_dataset = One2ManyDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name, usage="validation", total_many=args.total_many, mini_many=args.mini_many)
    cvae_structure_args = {
        "num_latent_encoder_layers": args.num_latent_encoder_layers,
        "frozen_pretrained": args.frozen_pretrained,
        "use_standard_prior": args.use_standard_prior,
        "vae_type": args.vae_type,
        "add_contra_loss": args.add_contra_loss,
        "add_skip_connection": args.add_skip_connection,
        "add_ghost_skip_connection": args.add_ghost_skip_connection,
        #"dim_z": 1024,
    }
    if args.cvae_model_path is None:
        sft_model_path = get_last_checkpoint_at_root(f"sft/{args.base_model_name}-{args.dataset_name}")
        model = AutoModelForCVAE.from_pretrained(sft_model_path, cvae_structure_args, device_map=f"cuda:{args.local_rank}")
    else:
        model = AutoModelForCVAE.from_pretrained(args.cvae_model_path, cvae_structure_args, device_map=f"cuda:{args.local_rank}")
    if args.vae_type in ["Optimus_", "Anneal_"]:
        train_steps, eval_steps = {
            "yelp": (1085, 1079),
            "agnews": (475, 1499),
            "dailydialog": (884, 950),
            "plato_dailydialog": (884, 950),
        }[args.dataset_name]
        model.config.train_steps = train_steps
        model.config.eval_steps = eval_steps

    class EvalAtStartCallback(TrainerCallback):
        def __init__(self):
            self.should_evaluate_once = True
        def on_step_begin(self, args, state, control, **kwargs):
            if self.should_evaluate_once:
                control.should_evaluate = True
                self.should_evaluate_once = False

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=train_dataset.tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        remove_unused_columns=False,
        collate_fn=train_dataset.collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EvalAtStartCallback()],
        save_total_limit=1,
    )
    #resume_from = args.cvae_model_path if args.cvae_model_path is not None else get_last_checkpoint(args.output_dir)
    resume_from = get_last_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()