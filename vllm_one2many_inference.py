import os
import json
import argparse

from tqdm import tqdm

from dataset import One2ManyDataset, get_dataset_max_seq_length
from utils import get_last_checkpoint
from config import root


def run_vllm_one2many_inference(args, tasks, output_max_length):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.sft_model_path,
        trust_remote_code=True,
        swap_space=32,
    )
    params = SamplingParams(
        n=32,
        temperature=1.0,
        top_p=0.95,
        max_tokens=output_max_length,
        min_tokens=2,
        stop=["<|eot_id|>"],
        include_stop_str_in_output=True
    )
    usage2results = {}
    for usage in ["train", "validation", "test"]:
        usage_tasks = [task for task in tasks if task["usage"]==usage]
        if len(usage_tasks) == 0:
            continue
        prompts = [task["prompt"] for task in usage_tasks]
        results = []
        # generate and print the first case
        outputs = llm.generate(
            prompts=prompts[:1],
            sampling_params=params,
            use_tqdm=True
        )
        prompt = prompts[0]
        responses = [output_.text for output_ in outputs[0].outputs]
        nlls = [-output_.cumulative_logprob for output_ in outputs[0].outputs]
        results.append({
            "prompt": prompt,
            "responses": responses,
            "nlls": nlls,
        })
        print(f"{json.dumps(results, ensure_ascii=False, indent=4)}")
        # run ~
        outputs = llm.generate(
            prompts=prompts[1:],
            sampling_params=params,
            use_tqdm=True
        )
        for output in outputs:
            prompt = output.prompt
            responses = [output_.text for output_ in output.outputs]
            nlls = [-output_.cumulative_logprob for output_ in output.outputs]
            assert len(responses) == params.n
            results.append({
                "prompt": prompt,
                "responses": responses,
                "nlls": nlls,
            })
        usage2results[usage] = results
    return usage2results


def main():
    parser = argparse.ArgumentParser(description="parse args for vllm one-to-many inference")
    parser.add_argument('--sft_model_path', type=str, default=None)
    parser.add_argument('--base_model_name', type=str, default="llama3-8b")
    parser.add_argument('--dataset_name', type=str, default="plato_dailydialog")
    parser.add_argument('--index_split', type=int, default=1)
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--erase', type=int, default=1)
    args = parser.parse_args()

    if args.sft_model_path is None:
        print(f"get last checkout at: {os.path.join(root, f'sft/{args.base_model_name}-{args.dataset_name}')}")
        args.sft_model_path = get_last_checkpoint(os.path.join(root, f"sft/{args.base_model_name}-{args.dataset_name}"))

    tasks = []
    usage2output_file = {}
    for usage in ["train", "validation", "test"]:
        dataset = One2ManyDataset(
            tokenizer_path=args.base_model_name,
            dataset_name=args.dataset_name,
            usage=usage,
            only_load_prompts=True,
        )
        usage2output_file[usage] = dataset.cache_file(index_split=args.index_split, num_splits=args.num_splits)
        prompts = [item["prompt"] for item in dataset.preprocessed_data]
        print(f"{usage}: {len(prompts)}")
        split_length = len(prompts) // args.num_splits + 1
        prompts = prompts[(args.index_split-1)*split_length: args.index_split*split_length]
        print(f"{usage}: {len(prompts)}")
        for prompt in tqdm(prompts):
            task = {
                "prompt": prompt,
                "usage": usage,
            }
            tasks.append(task)
    print(json.dumps(usage2output_file, ensure_ascii=False, indent=4))
    for usage in ["train", "validation", "test"]:
        usage_tasks = [task for task in tasks if task["usage"] == usage]
        print(f"{usage}: {len(usage_tasks)}")
    output_max_length = get_dataset_max_seq_length(args.dataset_name)[1]
    usage2results = run_vllm_one2many_inference(args, tasks, output_max_length)
    for usage, results in usage2results.items():
        usage_output_file = usage2output_file[usage]
        print(f"{usage}: {len(results)} -> {usage_output_file}")
        os.makedirs(os.path.dirname(usage_output_file), exist_ok=True)
        with open(usage_output_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()