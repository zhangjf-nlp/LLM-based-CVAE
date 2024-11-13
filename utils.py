import datetime
import os
import re
import json
import time
import torch
import argparse
import inspect
from transformers import TrainingArguments, Trainer


def get_basic_parser(description='parse args for modeling and training', **kwargs):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--base_model_name', type=str, default=kwargs.get("base_model_name", "llama3-8b"))
    parser.add_argument('--dataset_name', type=str, default=kwargs.get("dataset_name", "dailydialog"))
    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--epochs', type=int, default=kwargs.get("epochs", 3))
    parser.add_argument('--learning_rate', type=float, default=kwargs.get("learning_rate", 1e-5))
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=kwargs.get("global_batch_size", 64))
    parser.add_argument('--mini_batch_size', type=int, default=kwargs.get("mini_batch_size", 8))

    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def check_args_in_output_dir(args):
    args_file = os.path.join(args.output_dir, "args.json")
    args_matched = True
    if os.path.exists(args_file):
        with open(args_file, 'r', encoding='utf-8') as f:
            saved_args = json.loads(f.read())
        for k,v in vars(args).items():
            saved_v = saved_args.get(k, None)
            if saved_v != v and k not in ["local_rank", "output_dir"]:
                args_matched = False
                if args.local_rank == 0:
                    print(f"mismatched {k}: present {v} -> saved {saved_v}")
    return args_matched


def rename_output_dir(args):
    if args.local_rank == 0:
        creation_time = os.path.getctime(args.output_dir)
        timestamp = datetime.datetime.fromtimestamp(creation_time).strftime('%Y%m%d-%H%M%S')
        os.rename(args.output_dir, args.output_dir + "-" + timestamp)
        make_output_dir(args)


def make_output_dir(args):
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))


def check_output_dir(args):
    if os.path.exists(args.output_dir):
        experiment_finished = os.path.exists(os.path.join(args.output_dir, "trainer_state.json"))
        experiment_args_matched = check_args_in_output_dir(args)
        time.sleep(30)
        if experiment_finished and experiment_args_matched:
            print(f"args matched and experiment is finished -> exit training: {args.output_dir}")
            quit()
        elif experiment_finished and not experiment_args_matched:
            print(f"args mismatched while experiment is finished -> rename output_dir: {args.output_dir}")
            rename_output_dir(args)
        elif not experiment_finished and experiment_args_matched:
            print(f"args matched but experiment is not finished -> continue training: {args.output_dir}")
        else:
            assert not experiment_finished and not experiment_args_matched
            rename_output_dir(args)
    else:
        make_output_dir(args)
    if args.local_rank == 0:
        print(json.dumps(vars(args), ensure_ascii=False, indent=4))

def get_trainer(args, model, tokenizer, train_dataset, valid_dataset, collate_fn, **kwargs):
    save_total_limit = kwargs.pop("save_total_limit", 1)
    gradient_accumulation_steps = args.global_batch_size // (args.n_devices * args.mini_batch_size)
    total_training_steps = args.epochs * len(train_dataset) // args.global_batch_size
    warmup_steps = int(total_training_steps * 0.1)
    eval_steps = int(total_training_steps * 0.1)
    save_steps = eval_steps

    deepspeed = None if args.no_deepspeed else {
        "train_micro_batch_size_per_gpu": args.mini_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {"enabled": True, "min_loss_scale": 1, "opt_level": "O2"},
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True}},
        "optimizer": {"type": "AdamW", "params": {"lr": args.learning_rate}},
        "scheduler": {"type": "WarmupLR", "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            }
        }
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.mini_batch_size,
        per_device_eval_batch_size=args.mini_batch_size,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=min(10, max(1, eval_steps // 10)),
        deepspeed=deepspeed,
        load_best_model_at_end=True,
        **{k:v for k,v in kwargs.items() if k in inspect.signature(TrainingArguments.__init__).parameters},
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        **{k:v for k,v in kwargs.items() if k in inspect.signature(Trainer.__init__).parameters}
    )
    return trainer


def get_last_checkpoint(folder):
    if not os.path.exists(folder):
        return None
    if os.path.basename(folder) == "best_model":
        return folder
    if os.path.basename(folder) == "huggingface_format":
        return folder
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    if _re_checkpoint.search(os.path.basename(folder)) is not None:
        return folder
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if (_re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(folder, path))
            and os.path.exists(os.path.join(folder, path, "trainer_state.json")))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def get_last_checkpoint_at_root(folder):
    from config import root
    folder = os.path.join(root, folder)
    return get_last_checkpoint(folder)


def load_sharded_prefix_checkpoint(model, folder, prefix="actor_model", strict=True, prefer_safe=True):
    """
    Mostly copied from transformers.modeling_utils.load_sharded_checkpoint
    only load the state_dict of a submodule, e.g., actor / latent_encoder
    ---------------------
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    from transformers.modeling_utils import (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME, is_safetensors_available,
                                             logger, is_torch_greater_or_equal_than_1_13, safe_load_file, partial, gc)
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    def contain_prefix(parameter_name):
        module_names = parameter_name.split(".")
        return prefix in module_names

    def remove_prefix(parameter_name):
        assert contain_prefix(parameter_name)
        module_names = parameter_name.split(".")
        module_names = module_names[module_names.index(prefix) + 1:]
        return ".".join(module_names)

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = [remove_prefix(k) for k in index["weight_map"].keys() if contain_prefix(k)]
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
    loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        state_dict = {remove_prefix(k):v for k,v in state_dict.items() if contain_prefix(k)}
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)