import os
import shutil
import time

from config import root
from utils import get_basic_parser, get_trainer, get_last_checkpoint, check_output_dir, get_last_checkpoint_at_root
from dataset import One2ManyDataset
from models import AutoModelForCVAE
from transformers import TrainerCallback, set_seed


class EvalAtStartCallback(TrainerCallback):
    def __init__(self):
        self.should_evaluate_once = True
    def on_step_begin(self, args, state, control, **kwargs):
        if self.should_evaluate_once:
            control.should_evaluate = True
            self.should_evaluate_once = False


class SaveModelFileCallback(TrainerCallback):
    def __init__(self, source_file="modeling_llama_cvae.py"):
        self.source_file = source_file
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if args.local_rank<=0 and os.path.exists(self.source_file):
            dest_file = os.path.join(checkpoint_dir, os.path.basename(self.source_file))
            shutil.copy2(self.source_file, dest_file)
            print(f"Copied {self.source_file} to {dest_file}")


def compute_metrics(eval_preds):
    loss_kld, loss_lm, mi = eval_preds.predictions
    result = {
        "loss_kld": loss_kld.mean(),
        "loss_lm": loss_lm.mean(),
        "mi": mi.mean(),
    }
    return result


def set_default_output_dir(args):
    if args.output_dir is None:
        model_feature = (f"total{args.total_many}-mini{args.mini_many}-"
                         f"epochs{args.epochs}-lr{args.learning_rate}-"
                         f"frozen{args.frozen_stage}-"
                         f"{args.vae_type}")
        if args.add_skip_connection:
            model_feature += "-skip"
        if args.add_gskip_connection:
            model_feature += "-gskip"
        args.output_dir = os.path.join(root, f"cvae-zero1/{args.base_model_name}-{args.dataset_name}/{model_feature}")


def get_parser():
    parser = get_basic_parser(epochs=2, global_batch_size=16, mini_batch_size=1, learning_rate=1e-5)
    parser.add_argument('--cvae_model_path', type=get_last_checkpoint_at_root, default=None)
    parser.add_argument('--frozen_stage', type=int, default=0, help="0: full parameter;\n"
                                                                    "1: frozen backbone llm;\n"
                                                                    "2: frozen adapter attn and encoder embed.")
    parser.add_argument('--vae_type', type=str, default="DG-VAE", help="different kinds of loss")
    parser.add_argument('--add_skip_connection', type=int, default=1, help="skip connection from latent code to lm logits")
    parser.add_argument('--add_gskip_connection', type=int, default=0, help="gradient-only skip connection from latent code to lm logits")
    parser.add_argument('--total_many', type=int, default=32, help="total outputs for the same prompt")
    parser.add_argument('--mini_many', type=int, default=16, help="number of outputs in the same mini-batch")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.epochs == args.total_many // args.mini_many
    assert args.n_devices * (args.total_many // args.mini_many) == args.global_batch_size
    time.sleep(args.local_rank * 10)
    set_default_output_dir(args)
    check_output_dir(args)
    set_seed(args.seed)

    train_dataset = One2ManyDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name,
                                    usage="train", total_many=args.total_many, mini_many=args.mini_many)
    valid_dataset = One2ManyDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name,
                                    usage="validation", total_many=args.total_many, mini_many=args.mini_many)
    cvae_structure_args = {
        "frozen_stage": args.frozen_stage,
        "vae_type": args.vae_type,
        "add_skip_connection": args.add_skip_connection,
        "add_gskip_connection": args.add_gskip_connection,
    }
    if args.cvae_model_path is None:
        sft_model_path = get_last_checkpoint_at_root(f"sft/{args.base_model_name}-{args.dataset_name}")
        model = AutoModelForCVAE.from_pretrained(sft_model_path, cvae_structure_args, device_map=f"cuda:{args.local_rank}")
    else:
        model = AutoModelForCVAE.from_pretrained(args.cvae_model_path, cvae_structure_args, device_map=f"cuda:{args.local_rank}")
    if args.local_rank <= 0:
        print(f"total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"tunable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if args.vae_type in ["Optimus_", "Anneal_"]:
        # steps per epoch
        train_steps, eval_steps = {
            "yelp": (1085, 1079),
            "agnews": (475, 1499),
            "dailydialog": (884, 950),
            "plato_dailydialog": (884, 950),
        }[args.dataset_name]
        model.config.train_steps = train_steps
        model.config.eval_steps = eval_steps

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=train_dataset.tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        remove_unused_columns=False,
        collate_fn=train_dataset.collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EvalAtStartCallback(), SaveModelFileCallback()],
        save_total_limit=1,
    )
    resume_from = get_last_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()