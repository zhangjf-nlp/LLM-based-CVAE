import os
import csv
import json
import random
import hashlib
from collections import Counter
from typing import Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM, RobertaForSequenceClassification
from config import root
from datasets import DownloadConfig
from dataset import SFTDataset, One2ManyDataset, eot_token, ClassificationDataset, right_padding_to_left_padding, \
    ComparisonDataset, get_dataset_indexed_labels, polish
from modeling_llama_cvae import LlamaForCVAE
from utils import get_last_checkpoint
from utils_latent import sampling, log_pdf, exp_mean_log
from train_cvae import set_default_output_dir

torch.set_printoptions(precision=3, sci_mode=False)

version = "release"
version_results = "release"

ks = [16]  # for latent manipulation

if not os.path.exists("evaluate") and os.path.exists(root):
    # link to the local evaluate library
    os.system(f"ln -s {root}/evaluate evaluate")


def round_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().tolist()
    if isinstance(tensor, np.ndarray):
        tensor = tensor.tolist()
    if isinstance(tensor[0], list):
        return [[round(x, 3) for x in line] for line in tensor]
    return [round(x, 3) for x in tensor]


@torch.no_grad()
def compute_mauve(responses, references, model_id="gpt2-large", n_clusters: Union[int, str] = "auto"):
    torch.cuda.empty_cache()
    mauve = load("evaluate/metrics/mauve")
    results = mauve.compute(
        predictions=responses, references=references,
        featurize_model_name=model_id, max_text_length=256,
        device_id=0, verbose=False, num_buckets=n_clusters
    )
    assert len(responses) == len(responses)
    return results.mauve, results.divergence_curve


@torch.no_grad()
def compute_perplexity(responses, model_id="gpt2-large"):
    torch.cuda.empty_cache()
    perplexity = load("evaluate/metrics/perplexity", module_type="metric")
    results = perplexity.compute(predictions=responses, model_id=model_id, device='cuda')
    # return results['mean_perplexity']
    # we exclude abnormal points through only analyzing results from 1% to 99%
    perplexities = results["perplexities"]
    lower_bound = np.percentile(perplexities, 1)
    upper_bound = np.percentile(perplexities, 99)
    perplexities = [ppl for ppl in perplexities if lower_bound <= ppl <= upper_bound]
    return np.mean(perplexities)


@torch.no_grad()
def compute_bert_score(responses, references):
    bert = load("evaluate/metrics/bertscore")
    results = bert.compute(predictions=responses, references=references, lang="en")
    # >>> bert_score.utils.lang2model['en']
    # 'roberta-large'
    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()
    return results


@torch.no_grad()
def compute_debertascore(responses, references):
    bert = load("evaluate/metrics/bertscore")
    results = bert.compute(predictions=responses, references=references, lang="en",
                           model_type='microsoft/deberta-xlarge-mnli')
    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()
    return results


def compute_bleu(responses, references):
    bleu = load("evaluate/metrics/bleu",
                download_config=DownloadConfig(cache_dir='evaluate/cache/bleu', local_files_only=True))
    results = {
        f"bleu-{max_order}": bleu.compute(
            predictions=responses,
            references=references,
            max_order=max_order,
            smooth=True,
        )['bleu']
        for max_order in [1, 2, 3, 4]
    }
    return results


def compute_self_bleu(responses):
    try:
        return compute_bleu(
            responses=responses,
            references=[
                [responses[i] for i in range(len(responses)) if i != j]
                for j in range(len(responses))
            ]
        )
    except ZeroDivisionError:
        return {
            f"bleu-{max_order}": 0
            for max_order in [1, 2, 3, 4]
        }


def compute_bleu_cross(results):
    bleu_scores = []
    bleu = load("evaluate/metrics/bleu",
                download_config=DownloadConfig(cache_dir='evaluate/cache/bleu', local_files_only=True))
    for result in results:
        responses = result["responses"]
        references = result["references"]
        try:
            bleu_precision = bleu.compute(
                predictions=responses,
                references=[references] * len(responses),
                max_order=2,
                smooth=True,
            )['bleu']
        except ZeroDivisionError:
            bleu_precision = 0
        try:
            bleu_recall = bleu.compute(
                predictions=references,
                references=[responses] * len(references),
                max_order=2,
                smooth=True,
            )['bleu']
        except ZeroDivisionError:
            bleu_recall = 0
        bleu_scores.append((bleu_precision, bleu_recall))
    return np.mean([bleu_score[0] for bleu_score in bleu_scores]), np.mean(
        [bleu_score[1] for bleu_score in bleu_scores])


def compute_bleu_cross_mp(results):
    from multiprocessing import Pool
    num_processes = 32
    chunk_size = len(results) // num_processes + 1
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

    with Pool(processes=num_processes) as pool:
        bleu_scores = list(tqdm(pool.imap(compute_bleu_cross, chunks), total=len(chunks)))

    total_precision = 0
    total_recall = 0
    total_length = len(results)

    for chunk, (precision, recall) in zip(chunks, bleu_scores):
        chunk_length = len(chunk)
        total_precision += precision * (chunk_length / total_length)
        total_recall += recall * (chunk_length / total_length)

    return total_precision, total_recall


def compute_rouge(responses, references, use_aggregator=True, use_stemmer=False):
    rouge = load("evaluate/metrics/rouge")
    results = rouge.compute(predictions=responses, references=references,
                            use_aggregator=use_aggregator, use_stemmer=use_stemmer)
    return results


import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from evaluate import load


def compute_single_rouge(response, reference, use_aggregator, use_stemmer):
    rouge = load("evaluate/metrics/rouge")
    return rouge.compute(
        predictions=[response], references=[reference],
        use_aggregator=use_aggregator, use_stemmer=use_stemmer
    )


def compute_rouge_mp(responses, references, use_aggregator=True, use_stemmer=False):
    keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    results = []

    with ProcessPoolExecutor() as executor:
        future_to_response = {
            executor.submit(compute_single_rouge, response, reference, use_aggregator, use_stemmer): (
                response, reference)
            for response, reference in zip(responses, references)
        }

        for future in tqdm(as_completed(future_to_response), total=len(future_to_response)):
            result = future.result()
            results.append(result)

    return {
        key: np.mean([result[key] for result in results])
        for key in keys
    }


def compute_distinct(responses):
    def distinct_n(responses, n):
        all_ngrams = Counter()
        for response in responses:
            tokens = response.split()
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.update(ngrams)
        total_ngrams = sum(all_ngrams.values())
        distinct_ngrams = len(all_ngrams)
        return distinct_ngrams / total_ngrams if total_ngrams > 0 else 0

    results = {
        f"distinct-{n}": distinct_n(responses, n)
        for n in [1, 2, 3, 4]
    }
    return results


@torch.no_grad()
def compute_mauve_for_results(results):
    from mauve.compute_mauve import get_features_from_input, compute_mauve
    torch.eye(3).cuda()  # ensure cuda available
    all_references = []
    all_responses = []
    for result in results:
        references = result["references"]
        responses = result["responses"]
        assert len(references) == len(responses) == 32
        all_references += references
        all_responses += responses
    assert len(all_references) == 32 * len(results)
    assert len(all_responses) == 32 * len(results)
    all_references_features = get_features_from_input(
        features=None,
        tokenized_texts=None,
        texts=all_references,
        featurize_model_name='gpt2-large',
        max_len=512,
        device_id=0,
        name="references",
        batch_size=32,
    )
    assert all_references_features.shape[0] == 32 * len(results)
    all_responses_features = get_features_from_input(
        features=None,
        tokenized_texts=None,
        texts=all_responses,
        featurize_model_name='gpt2-large',
        max_len=512,
        device_id=0,
        name="responses",
        batch_size=32,
    )
    assert all_responses_features.shape[0] == 32 * len(results)
    all_mauve_scores = []
    for i in tqdm(range(len(results))):
        mauve_score = compute_mauve(
            p_features=all_responses_features[32 * i:32 * (i + 1), :],
            q_features=all_references_features[32 * i:32 * (i + 1), :],
            num_buckets=8,
        ).mauve
        all_mauve_scores.append(mauve_score)
    return all_mauve_scores


def compute_rouge_for_chunk(chunk):
    responses, references = chunk
    return compute_rouge(responses=responses, references=references)


def compute_singular_rouge_for_results(results, chunk_size=1024):
    responses = [response for result in results for response in result["responses"]]
    references = [reference for result in results for reference in result["references"]]
    n = len(responses)
    chunks = [(responses[i:i + chunk_size], references[i:i + chunk_size])
              for i in range(0, n, chunk_size)]

    chunks_results = []
    with ProcessPoolExecutor() as executor:
        for chunk_results in tqdm(executor.map(compute_rouge_for_chunk, chunks), total=len(chunks)):
            chunks_results.append(chunk_results)

    rouge_results = {
        key: np.sum([
            chunk_results[key] * len(chunk[0]) / n
            for chunk, chunk_results in zip(chunks, chunks_results)
        ])
        for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    }

    return rouge_results


def compute_group_rouge_for_results(results, chunk_size=128):
    responses = ["\n".join(result["responses"]) for result in results]
    references = ["\n".join(result["references"]) for result in results]
    n = len(responses)
    chunks = [(responses[i:i + chunk_size], references[i:i + chunk_size])
              for i in range(0, n, chunk_size)]

    chunks_results = []
    with ProcessPoolExecutor() as executor:
        for chunk_results in tqdm(executor.map(compute_rouge_for_chunk, chunks), total=len(chunks)):
            chunks_results.append(chunk_results)

    rouge_results = {
        key: np.sum([
            chunk_results[key] * len(chunk[0]) / n
            for chunk, chunk_results in zip(chunks, chunks_results)
        ])
        for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    }

    return rouge_results


def compute_distinct_for_chunk(chunk):
    return [compute_distinct(responses=responses) for responses in chunk]


def compute_distinct_for_results(results, chunk_size=32):
    all_responses = [result["responses"] for result in results]
    n = len(all_responses)
    chunks = [all_responses[i:i + chunk_size] for i in range(0, n, chunk_size)]

    chunks_results = []
    with ProcessPoolExecutor() as executor:
        for chunk_results in tqdm(executor.map(compute_distinct_for_chunk, chunks), total=len(chunks)):
            chunks_results.append(chunk_results)

    distinct_results = {
        key: np.mean([
            result[key]
            for chunk_results in chunks_results
            for result in chunk_results
        ])
        for key in ['distinct-1', 'distinct-2', 'distinct-3', 'distinct-4']
    }

    return distinct_results


def compute_selfbleu_for_chunk(chunk):
    return [compute_self_bleu(responses=responses) for responses in chunk]


def compute_selfbleu_for_results(results, chunk_size=32):
    all_responses = [result["responses"] for result in results]
    n = len(all_responses)
    chunks = [all_responses[i:i + chunk_size] for i in range(0, n, chunk_size)]

    chunks_results = []
    with ProcessPoolExecutor() as executor:
        for chunk_results in tqdm(executor.map(compute_selfbleu_for_chunk, chunks), total=len(chunks)):
            chunks_results.append(chunk_results)

    selfbleu_results = {
        key: np.mean([
            result[key]
            for chunk_results in chunks_results
            for result in chunk_results
        ])
        for key in ["bleu-1", "bleu-2", "bleu-3", "bleu-4"]
    }

    return selfbleu_results


class TestGroundBase:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0"

    def load_model(self):
        raise NotImplementedError

    def strategy_name(self):
        raise NotImplementedError

    def save_results_to_csv(self, ell_results=None, prior_results=None, post_results=None):
        # Define the header for the CSV file
        header = [
            "model_name", "kl_mean", "joint_mi_mean", "marginal_mi_mean",
            "au_1", "au_2", "au_3", "cu_1", "cu_2", "cu_3",
            "prior-mauve_score", "prior-distinct", "prior-bleu", "prior-rouge", "prior-selfbleu",
            "post-mauve_score", "post-distinct", "post-bleu", "post-rouge", "post-selfbleu",
            "cvae_model_path", "finished_training",
        ]

        # ell_results: test_encoder_language_latent results
        if ell_results is not None:
            kl_mean = np.mean([result["kl"] for result in ell_results])
            joint_mi_mean = np.mean([result["joint_mi"] for result in ell_results])
            marginal_mi_mean = np.mean([result["marginal_mi"] for result in ell_results])
            au_mean = np.array([result["au"] for result in ell_results]).mean(axis=0)
            cu_mean = np.array([result["cu"] for result in ell_results]).mean(axis=0)
        else:
            kl_mean = np.nan
            joint_mi_mean = np.nan
            marginal_mi_mean = np.nan
            au_mean = [np.nan] * 3
            cu_mean = [np.nan] * 3

        # pri_results: test_prior_inference results
        if prior_results is None:
            prior_results = {}

        # post_results: test_posterior_inference results
        if post_results is None:
            post_results = {}

        # Prepare the row to be added to the CSV
        model_name = f"{self.args.dataset_name}-{self.strategy_name()}"
        if self.args.small_test:
            model_name += "-small_test"
        if self.args.cvae_model_path is not None and "1024_dim_z" in self.args.cvae_model_path:
            model_name += "-1024_dim_z"
        row = [
            model_name,
            kl_mean,
            joint_mi_mean,
            marginal_mi_mean,
            *au_mean,
            *cu_mean,
            prior_results.get("mauve_score", None),
            prior_results.get("distinct", None),
            prior_results.get("bleu", None),
            prior_results.get("rouge", None),
            prior_results.get("selfbleu", None),
            post_results.get("mauve_score", None),
            post_results.get("distinct", None),
            post_results.get("bleu", None),
            post_results.get("rouge", None),
            post_results.get("selfbleu", None),
            args.cvae_model_path,
            os.path.exists(os.path.join(args.output_dir, "trainer_state.json"))
        ]

        # Write the data to the CSV file
        csv_file_path = os.path.join(root, f"test_model_single_plus_greedy_results_{version_results}.csv")
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def save_prior_inference_with_different_temperature_results_to_csv(self, prior_results):
        csv_file_path = os.path.join(root, f"test_prior_temperature_inference_results_{version_results}.csv")

        # Define the header for the CSV file
        header = [
            "model_name",
            "mauve",
            "rougeL",
            "distinct",
            "selfbleu",
            "cvae_model_path", "finished_training",
        ]

        row = [
            f"{self.args.dataset_name}-{self.strategy_name()}" + ("-small_test" if self.args.small_test else ""),
            round_tensor(prior_results["final_result"]["mauve"]),
            round_tensor(prior_results["final_result"]["rougeL"]),
            round_tensor(prior_results["final_result"]["distinct"]),
            round_tensor(prior_results["final_result"]["selfbleu"]),
            args.cvae_model_path,
            os.path.exists(os.path.join(args.output_dir, "trainer_state.json")),
        ]

        # Write the data to the CSV file
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def save_manipulated_results_to_csv(self, manipulated_results):
        csv_file_path = os.path.join(root, f"test_model_manipulated_results_{version_results}.csv")

        # Define the header for the CSV file
        header = [
            "model_name", "k", "index_feature",
            "references_ppl",
            "references_probs-0", "1", "2", "3", "4",
            "reconstructed_ppl",
            "reconstructed_probs-0", "1", "2", "3", "4",
            "reconstructed_rouge-1", "2", "L", "Lsum",
            "reconstructed_debertascore-precision", "recall", "f1",
            "manipulated_ppl",
            "manipulated_probs-0", "1", "2", "3", "4",
            "manipulated_rouge-1", "2", "L", "Lsum",
            "manipulated_debertascore-precision", "recall", "f1",
            "cvae_model_path", "finished_training",
        ]

        # Prepare the row to be added to the CSV
        model_name = f"{self.args.dataset_name}-{self.strategy_name()}"

        def extend_probs(probs: list):
            return probs + [np.nan] * (5 - len(probs))

        for k in ks:
            for index_feature in range(len(get_dataset_indexed_labels(self.args.dataset_name))):
                row = [
                    model_name, k, index_feature,
                    manipulated_results[0]["references_ppl"],
                    *extend_probs(
                        np.array([result["references_probs"] for result in manipulated_results]).mean(axis=1).mean(
                            axis=0).tolist()),
                    manipulated_results[0]["reconstructed_ppl"],
                    *extend_probs(
                        np.array([result["reconstructed_probs"] for result in manipulated_results]).mean(axis=1).mean(
                            axis=0).tolist()),
                    *[np.mean([result["reconstructed_rouge"][key] for result in manipulated_results])
                      for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]],
                    *[np.mean([result["reconstructed_debertascore"][key] for result in manipulated_results])
                      for key in ["precision", "recall", "f1"]],
                    manipulated_results[0][f"manipulated_ppl-{index_feature}@{k}"],
                    *extend_probs(np.array(
                        [result[f"manipulated_probs-{index_feature}@{k}"] for result in manipulated_results]).mean(
                        axis=1).mean(axis=0).tolist()),
                    *[np.mean([result[f"manipulated_rouge-{index_feature}@{k}"][key] for result in manipulated_results])
                      for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]],
                    *[np.mean([result[f"manipulated_debertascore-{index_feature}@{k}"][key] for result in
                               manipulated_results])
                      for key in ["precision", "recall", "f1"]],
                    args.cvae_model_path,
                    os.path.exists(os.path.join(args.output_dir, "trainer_state.json"))
                ]

                # Write the data to the CSV file
                file_exists = os.path.isfile(csv_file_path)
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(header)
                    writer.writerow(row)

    def save_most_likely_results_to_csv(self, most_likely_results):
        csv_file_path = os.path.join(root, f"test_model_most_likely_results_{version_results}.csv")

        # Define the header for the CSV file
        header = [
            "model_name", "mauve_array", "mauve_array_trace", "mauve_array_trace_ratio",
            "cvae_model_path", "finished_training",
        ]

        row = [
            f"{self.args.dataset_name}-{self.strategy_name()}",
            str(most_likely_results[0]["mauve_array"]),
            most_likely_results[0]["mauve_array_trace"],
            most_likely_results[0]["mauve_array_trace_ratio"],
            args.cvae_model_path,
            os.path.exists(os.path.join(args.output_dir, "trainer_state.json")),
        ]

        # Write the data to the CSV file
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def save_interpolation_results_to_csv(self, interpolation_results, postfix=""):
        csv_file_path = os.path.join(root, f"test_model_interpolation_results_{version_results}{postfix}.csv")

        # Define the header for the CSV file
        header = [
            "model_name",
            "rouge_a-1", "2", "L", "Lsum",
            "rouge_b-1", "2", "L", "Lsum",
            "rouge_avg-1", "2", "L", "Lsum",
            "cvae_model_path", "finished_training",
        ]

        row = [
            f"{self.args.dataset_name}-{self.strategy_name()}" + ("-small_test" if self.args.small_test else ""),
            *[round_tensor([interpolation_results[i]["rouge_a"][key] for i in range(11)])
              for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]],
            *[round_tensor([interpolation_results[i]["rouge_b"][key] for i in range(11)])
              for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]],
            *[round_tensor([(interpolation_results[i]["rouge_a"][key] + interpolation_results[i]["rouge_b"][key]) / 2
                            for i in range(11)])
              for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]],
            args.cvae_model_path,
            os.path.exists(os.path.join(args.output_dir, "trainer_state.json")),
        ]

        # Write the data to the CSV file
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def test_dailydialog(self):
        self.load_model()
        dataset = SFTDataset(
            tokenizer_path=self.args.base_model_name,
            dataset_name=self.args.dataset_name,
            usage="test",
        )
        generation_config = {
            "num_return_sequences": 1,
            "do_sample": False if self.args.temperature == 0.0 else True,
            "temperature": self.args.temperature,
            "max_new_tokens": dataset.max_output_length,
            "min_new_tokens": 2,
        }
        indexes = range(0, len(dataset), 10 if self.args.small_test else 1)
        prompts, responses, references = [], [], []
        for index in tqdm(indexes, desc="testing prior mean greedy inference"):
            data = dataset[index]
            prompt_ids, reference_ids = data["prompt_ids"], data["response_ids"]
            attention_mask = torch.ones_like(prompt_ids)
            inputs = {
                "input_ids": prompt_ids.cuda(),
                "attention_mask": attention_mask.cuda(),
            }
            with torch.no_grad():
                response_ids = self.model.generate(
                    **inputs,
                    **generation_config,
                )
            prompt_length = inputs["input_ids"].shape[1]
            response = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:], skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)[0]
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            response = cut_tail_after(response, dataset.tokenizer.eos_token)
            response = cut_tail_after(response, eot_token)
            reference = dataset.preprocessed_data[index]["response"]
            reference = cut_tail_after(reference, eot_token)
            prompt = dataset.preprocessed_data[index]["prompt"]
            print(f"prompt: {prompt}")
            print(f"reference: {reference}")
            print(f"response: {response}")
            prompts.append(prompt)
            responses.append(response)
            references.append(reference)

        from evaluation_dior import bleu
        from evaluation_dior import distinct
        bleu(hyps=responses, refs=references)
        distinct(hyps=responses)
        bertscore = compute_bert_score(responses, references)
        print(f"bertscore: {bertscore}")

        import fed
        # Load model
        model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
        # Evaluate
        keys = ["relevant", "understandable", "coherent", "diverse"]
        all_scores = []
        for prompt, response in zip(tqdm(prompts), responses):
            conversation = polish(prompt).replace(eot_token, " <|endoftext|> ") + polish(response)
            scores = fed.evaluate(conversation, model, tokenizer, aspects=" ".join(keys))
            all_scores.append(scores)
        for key in keys:
            print(f"response-{key}: {np.mean([scores[key] for scores in all_scores]):.3f}")
        print(f"response-overall: {np.mean([scores[key] for scores in all_scores for key in keys]):.3f}")


class TestGroundForSFT(TestGroundBase):
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.args.sft_model_path
        if not hasattr(self, "model"):
            load_kwargs = {"torch_dtype": torch.float16, "device_map": self.device}
            model = LlamaForCausalLM.from_pretrained(model_path, **load_kwargs)
            self.model = model
        if not hasattr(self, "tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(self.args.sft_model_path)
            self.tokenizer = tokenizer

    def strategy_name(self):
        return self.args.vae_type

    def test_dailydialog(self):
        temperature = self.args.temperature
        cache_file = os.path.join(self.args.sft_model_path,
                                  f"test_sampling_single_temp{int(temperature * 10):02d}_{version}.json")
        dataset = SFTDataset(
            tokenizer_path=self.args.base_model_name,
            dataset_name=self.args.dataset_name,
            usage="test",
        )
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model=self.args.sft_model_path,
                trust_remote_code=True,
                swap_space=32,
            )
            if temperature >= 0:
                params = SamplingParams(
                    n=1,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=dataset.max_output_length,
                    min_tokens=2,
                    stop=[eot_token, dataset.tokenizer.eos_token],
                    include_stop_str_in_output=False
                )
            else:
                params = SamplingParams(
                    n=1,
                    temperature=temperature,
                    use_beam_search=True,
                    best_of=16,
                    top_p=1.0,
                    max_tokens=dataset.max_output_length,
                    min_tokens=2,
                    stop=[eot_token, dataset.tokenizer.eos_token],
                    include_stop_str_in_output=False
                )
            results = []
            for index in range(len(dataset)):
                prompt = dataset.data[index]["prompt"]
                reference = dataset.data[index]["response"]
                results.append({
                    "prompt": prompt,
                    "reference": reference,
                    "temperature": temperature,
                })
            outputs = llm.generate(
                prompts=[result["prompt"] for result in results],
                sampling_params=params,
                use_tqdm=True
            )
            for result, output in zip(results, outputs):
                assert result["prompt"] == output.prompt
                # result["responses"] = [output.outputs[i].text for i in range(32)]
                result["response"] = \
                    dataset.tokenizer.batch_decode([output.outputs[0].token_ids], clean_up_tokenization_spaces=False)[0]
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        if args.small_test:
            results = [result for i, result in enumerate(results) if i % 10 == 0]

        cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
        for result in results:
            result["response"] = cut_tail_after(result["response"], dataset.tokenizer.eos_token)
            result["response"] = cut_tail_after(result["response"], eot_token)
            result["reference"] = cut_tail_after(result["reference"], eot_token)

        print(f"dataset: {self.args.dataset_name}")
        print(f"temperature: {temperature}")
        print(self.args.sft_model_path.split("-")[-1])

        from evaluation_dior import bleu, distinct, entropy
        prompts = [result["prompt"] for result in results]
        responses = [result["response"] for result in results]
        references = [result["reference"] for result in results]
        for i in range(3):
            print(f'prompt: {results[i]["prompt"]}')
            print(f'reference: {results[i]["reference"]}')
            print(f'response: {results[i]["response"]}')
            print('-' * 20)
        bleu(hyps=responses, refs=references)
        distinct(hyps=responses)
        entropy(hyps=responses)
        bertscore = compute_bert_score(responses, references)
        print(f"bertscore: {bertscore}")

        import fed
        # Load model
        model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
        # Evaluate
        keys = ["relevant", "understandable", "coherent", "diverse"]
        all_scores = []
        for prompt, response in zip(tqdm(prompts), responses):
            conversation = polish(prompt).replace(eot_token, " <|endoftext|> ") + polish(response)
            scores = fed.evaluate(conversation, model, tokenizer, aspects=" ".join(keys))
            all_scores.append(scores)
        for key in keys:
            print(f"response-{key}: {np.mean([scores[key] for scores in all_scores]):.3f}")
        print(f"response-overall: {np.mean([scores[key] for scores in all_scores for key in keys]):.3f}")

    def test_sampling(self, temperature=1.0):
        cache_file = os.path.join(self.args.sft_model_path,
                                  f"test_sampling_temp{int(temperature * 10):02d}_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            from vllm import LLM, SamplingParams
            llm = LLM(
                model=self.args.sft_model_path,
                trust_remote_code=True,
                swap_space=32,
            )
            params = SamplingParams(
                n=32,
                temperature=temperature,
                top_p=0.95,
                max_tokens=dataset.max_output_length,
                min_tokens=2,
                stop=[eot_token, dataset.tokenizer.eos_token],
                include_stop_str_in_output=False
            )
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            results = []
            for index in range(len(dataset)):
                prompt = dataset.one2many_inference[index]["prompt"]
                references = dataset.one2many_inference[index]["responses"]
                results.append({
                    "prompt": prompt,
                    "references": references,
                    "temperature": temperature,
                })
            outputs = llm.generate(
                prompts=[result["prompt"] for result in results],
                sampling_params=params,
                use_tqdm=True
            )
            for result, output in zip(results, outputs):
                assert result["prompt"] == output.prompt
                # result["responses"] = [output.outputs[i].text for i in range(32)]
                result["responses"] = [
                    dataset.tokenizer.batch_decode([output.outputs[i].token_ids], clean_up_tokenization_spaces=False)
                    for i in range(32)]
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        if args.small_test:
            results = [result for i, result in enumerate(results) if i % 10 == 0]

        cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
        for result in results:
            result["responses"] = [cut_tail_after(response, eot_token) for response in result["responses"]]
            result["references"] = [cut_tail_after(response, eot_token) for response in result["references"]]

        print(f"dataset: {self.args.dataset_name}")
        print(f"temperature: {temperature}")
        mauve_scores = compute_mauve_for_results(results)
        print(f"mauve_scores: {np.mean(mauve_scores) * 100:.2f} +/- {np.std(mauve_scores) * 100:.2f}")
        singular_rouge_scores = compute_singular_rouge_for_results(results)
        print(f"singular_rouge_scores:\n{json.dumps(singular_rouge_scores, ensure_ascii=False, indent=4)}")
        group_rouge_scores = compute_group_rouge_for_results(results)
        print(f"group_rouge_scores:\n{json.dumps(group_rouge_scores, ensure_ascii=False, indent=4)}")
        distinct_scores = compute_distinct_for_results(results)
        print(f"distinct_scores:\n{json.dumps(distinct_scores, ensure_ascii=False, indent=4)}")
        selfbleu_scores = compute_selfbleu_for_results(results)
        print(f"selfbleu_scores:\n{json.dumps(selfbleu_scores, ensure_ascii=False, indent=4)}")

    def test_posterior_inference(self):
        cache_file = os.path.join(self.args.sft_model_path, ("small_" if self.args.small_test else "") + (
            "sampled_" if "sampling" in self.args.vae_type else "") + f"test_prior_unbiased_inference_{version}.json")
        if os.path.exists(cache_file) and not self.args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not self.args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": False,
                # "temperature": self.args.temperature,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            results = {}
            indexes = range(0, len(dataset), 10 if args.small_test else 1)
            for index in tqdm(indexes, desc="greedy search"):
                prompt = dataset.one2many_inference[index]["prompt"]
                reference = dataset.preprocessed_data[index]["response"]
                references = dataset.one2many_inference[index]["responses"]
                if str(index) in cached_results:
                    result: dict = cached_results[str(index)]
                    if not all([result.get(k, None) == v for k, v in {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references
                    }.items()]):
                        result = {}
                else:
                    result = {}
                if result == {}:
                    inputs = dataset.tokenizer([prompt], return_tensors="pt")
                    inputs = {k: v.cuda().repeat(len(references), 1) for k, v in inputs.items()}
                    prompt_length = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        response_ids = self.model.generate(
                            **inputs,
                            **generation_config,
                        )
                    responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                               skip_special_tokens=False)
                    cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
                    responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in responses]
                    responses = [cut_tail_after(response, eot_token) for response in responses]
                    responses = ["." if len(response) == 0 else response for response in responses]
                    references = [cut_tail_after(reference, dataset.tokenizer.eos_token) for reference in references]
                    references = [cut_tail_after(reference, eot_token) for reference in references]
                    # mauve_score, _ = compute_mauve(responses, references, n_clusters=8)
                    # distinct = compute_distinct(responses=responses)
                    # bleu = compute_bleu(responses=responses, references=[references for _ in responses])
                    # rouge = compute_rouge(responses=responses, references=[references for _ in responses])
                    # selfbleu = compute_self_bleu(responses=responses)
                    result = {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references,
                        "responses": responses,
                        # "mauve_score": mauve_score,
                        # "distinct": distinct,
                        # "bleu": bleu,
                        # "rouge": rouge,
                        # "selfbleu": selfbleu,
                    }
                results[index] = result
                if len(results) % (len(indexes) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        results = list(results.values())
        # mauve_score = np.mean([result["mauve_score"] for result in results])
        # distinct = {k:np.mean([result["distinct"][k] for result in results])
        #            for k in ['distinct-1', 'distinct-2', 'distinct-3', 'distinct-4']}
        # bleu = {k:np.mean([result["bleu"][k] for result in results])
        #        for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}
        # rouge = {k:np.mean([result["rouge"][k] for result in results])
        #         for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
        # selfbleu = {k:np.mean([result["selfbleu"][k] for result in results])
        #             for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}
        # update in v6.2
        # re-compute rouge scores
        # rouge = compute_rouge_mp(
        #    responses = [response for result in results for response in result["responses"]],
        #    references = [reference for result in results for reference in result["references"]],
        # )
        mauve_score = np.mean(compute_mauve_for_results(results))
        rouge = compute_singular_rouge_for_results(results)
        distinct = compute_distinct_for_results(results)
        selfbleu = compute_selfbleu_for_results(results)

        results = {
            "mauve_score": mauve_score,
            "distinct": distinct,
            # "bleu": bleu,
            "selfbleu": selfbleu,
            "rouge": rouge,
        }
        return results

    def test_prior_inference_with_different_temperature(self):
        temperatures = [0.1, 0.4, 0.7, 1.0]
        cache_file = os.path.join(self.args.sft_model_path, (
            "small_" if args.small_test else "") + f"test_prior_temperature_inference_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 32,
                "do_sample": True,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            results = {}
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            indexes = range(0, len(dataset), 10 if args.small_test else 1)
            for temperature in temperatures:
                temperature_repr = f"{int(temperature * 10):02d}"
                results[temperature_repr] = {}
                for index in tqdm(indexes, desc=f"test_prior_temperature_inference_{temperature_repr}"):
                    prompt = dataset.one2many_inference[index]["prompt"]
                    reference = dataset.preprocessed_data[index]["response"]
                    references = dataset.one2many_inference[index]["responses"]
                    references = [cut_tail_after(reference, dataset.tokenizer.eos_token) for reference in references]
                    references = [cut_tail_after(reference, eot_token) for reference in references]
                    result = {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references
                    }
                    if temperature_repr in cached_results and str(index) in cached_results[temperature_repr]:
                        cached_result = cached_results[temperature_repr][str(index)]
                        if all([cached_result.get(k, None) == v for k, v in result.items()]):
                            result["responses"] = cached_result["responses"]
                    if "responses" not in result:
                        inputs = dataset.tokenizer([prompt], return_tensors="pt")
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        prompt_length = inputs["input_ids"].shape[1]
                        with torch.no_grad():
                            response_ids = self.model.generate(
                                **inputs,
                                **generation_config,
                                temperature=temperature,
                            )
                        responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                                   skip_special_tokens=False)
                        responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in responses]
                        responses = [cut_tail_after(response, eot_token) for response in responses]
                        responses = ["." if len(response) == 0 else response for response in responses]
                        result["responses"] = responses
                    results[temperature_repr][index] = result
                with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                    f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        results["final_result"] = {
            "mauve": [np.mean(compute_mauve_for_results(results[f"{int(temperature * 10):02d}"].values())) for
                      temperature in temperatures],
            "rougeL": [compute_group_rouge_for_results(results[f"{int(temperature * 10):02d}"].values())["rougeLsum"]
                       for temperature in temperatures],
            "distinct": [compute_distinct_for_results(results[f"{int(temperature * 10):02d}"].values())["distinct-4"]
                         for temperature in temperatures],
            "selfbleu": [compute_selfbleu_for_results(results[f"{int(temperature * 10):02d}"].values())["bleu-4"] for
                         temperature in temperatures],
        }
        return results

    def test_prior_inference(self):
        cache_file = os.path.join(self.args.sft_model_path, (
            "small_" if args.small_test else "") + f"test_posterior_inference_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": True,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            results = {}
            indexes = range(0, len(dataset), 10 if args.small_test else 1)
            for index in tqdm(indexes, desc="sampling"):
                prompt = dataset.one2many_inference[index]["prompt"]
                reference = dataset.preprocessed_data[index]["response"]
                references = dataset.one2many_inference[index]["responses"]
                if str(index) in cached_results:
                    result: dict = cached_results[str(index)]
                    if not all([result.get(k, None) == v for k, v in {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references
                    }.items()]):
                        result = {}
                else:
                    result = {}
                if result == {}:
                    prior_inputs = dataset.tokenizer([prompt], return_tensors="pt")
                    inputs = {
                        "input_ids": prior_inputs["input_ids"].cuda().repeat(len(references), 1),
                        "attention_mask": prior_inputs["attention_mask"].cuda().repeat(len(references), 1),
                    }
                    prompt_length = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        response_ids = self.model.generate(
                            **inputs,
                            **generation_config,
                        )
                    responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                               skip_special_tokens=False)
                    cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
                    responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in responses]
                    responses = [cut_tail_after(response, eot_token) for response in responses]
                    responses = ["." if len(response) == 0 else response for response in responses]
                    references = [cut_tail_after(reference, dataset.tokenizer.eos_token) for reference in references]
                    references = [cut_tail_after(reference, eot_token) for reference in references]
                    # mauve_score, _ = compute_mauve(responses, references, n_clusters=8)
                    # distinct = compute_distinct(responses=responses)
                    # bleu = compute_bleu(responses=responses, references=[[reference] for reference in references])
                    # rouge = compute_rouge(responses=responses, references=[[reference] for reference in references])
                    # debertascore = compute_debertascore(responses=responses,
                    #                                    references=[[reference] for reference in references])
                    # selfbleu = compute_self_bleu(responses=responses)
                    result = {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references,
                        "responses": responses,
                        # "mauve_score": mauve_score,
                        # "distinct": distinct,
                        # "bleu": bleu,
                        # "rouge": rouge,
                        # "debertascore": debertascore,
                        # "selfbleu": selfbleu
                    }
                results[index] = result
                if index % (len(indexes) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        results = list(results.values())
        # mauve_score = np.mean([result["mauve_score"] for result in results])
        # distinct = {k: np.mean([result["distinct"][k] for result in results])
        #            for k in ['distinct-1', 'distinct-2', 'distinct-3', 'distinct-4']}
        # bleu = {k: np.mean([result["bleu"][k] for result in results])
        #        for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}
        # rouge = {k: np.mean([result["rouge"][k] for result in results])
        #         for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
        # debertascore = {k: np.mean([result["debertascore"][k] for result in results])
        #                for k in ['precision', 'recall', 'f1']}
        # selfbleu = {k: np.mean([result["selfbleu"][k] for result in results])
        #            for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}

        # update in v6.2
        # re-compute rouge scores
        # rouge = compute_rouge_mp(
        #    responses = ["\n".join(result["responses"]) for result in results],
        #    references = ["\n".join(result["references"]) for result in results],
        # )

        mauve_score = np.mean(compute_mauve_for_results(results))
        rouge = compute_group_rouge_for_results(results)
        distinct = compute_distinct_for_results(results)
        selfbleu = compute_selfbleu_for_results(results)

        results = {
            "mauve_score": mauve_score,
            "distinct": distinct,
            # "bleu": bleu,
            "rouge": rouge,
            # "debertascore": debertascore,
            "selfbleu": selfbleu,
        }
        return results

    def test_encoder_language_latent(self, n_samples=16, verbose=1):
        return None

    def test_interpolation(self, num_test_shots="all"):
        cache_file = os.path.join(
            self.args.sft_model_path + f"test_interpolation_{num_test_shots}_{version}.json"
        )
        if os.path.exists(cache_file) and not self.args.erase and False:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model="Meta-Llama-3-8B-Instruct",
                trust_remote_code=True,
                swap_space=32,
            )
            params = SamplingParams(
                n=1,
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024,
                min_tokens=2,
                stop=["<|eot_id|>"],
                include_stop_str_in_output=False
            )
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            prompts = []
            results = []
            for index in range(len(dataset)):
                prompt = dataset.one2many_inference[index]["prompt"]
                references = dataset.one2many_inference[index]["responses"]
                random.seed(int(hashlib.sha256(prompt.encode()).hexdigest(), 16))
                a, b = random.sample(range(32), 2)
                context = prompt.replace("<|eot_id|>", "\n")
                response_a = references[a].replace("<|eot_id|>", "\n")
                response_b = references[b].replace("<|eot_id|>", "\n")
                prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "You are a helpful AI assistant to perform interpolated generation between two sentences.\n"
                    "Good interpolated sentences should be consistent in context, and representing semantics at a middle position of the two given sentences.<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    f"Context:\n{context}\n\n"
                    f"Response A:\n{response_a}\n\n"
                    f"Response B:\n{response_b}\n\n"
                    "Please perform continuous interpolation between Responses A and B in the semantic space, "
                    "and precisely generate Response C for the interpolated semantic, "
                    "s.t. $ C = A * \\lambda + B * (1-\\lambda) $ "
                    "for $ \\lambda \\in \\{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\\}$."
                    "Return your outputs in the following format strictly:\n"
                    "---start of interpolation---\n"
                    "C($\\lambda=0.0$): xxx\n"
                    "C($\\lambda=0.1$): xxx\n"
                    "C($\\lambda=0.2$): xxx\n"
                    "C($\\lambda=0.3$): xxx\n"
                    "C($\\lambda=0.4$): xxx\n"
                    "C($\\lambda=0.5$): xxx\n"
                    "C($\\lambda=0.6$): xxx\n"
                    "C($\\lambda=0.7$): xxx\n"
                    "C($\\lambda=0.8$): xxx\n"
                    "C($\\lambda=0.9$): xxx\n"
                    "C($\\lambda=1.0$): xxx\n"
                    "---end of interpolation---\n"
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
                prompts.append(prompt)
                results.append({
                    "prompt": prompt,
                    "reference_a": response_a,
                    "reference_b": response_b,
                })
            outputs = llm.generate(
                prompts=prompts,
                sampling_params=params,
                use_tqdm=True
            )
            num_unformatted = 0
            for result, output in zip(results, outputs):
                prompt = output.prompt
                formatted_output = output.outputs[0].text
                print(f"prompt:\n{prompt}")
                print(f"formatted_output:\n{formatted_output}")
                print("-" * 100)
                formatted_output = cut_tail_after(formatted_output, "<|eot_id|>")
                heads = [f"C($\\lambda={lambda_value}$):" for lambda_value in
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
                heads.append("---end of interpolation---")
                if not all(head in formatted_output for head in heads):
                    interpolated_responses = ["ERROR"] * 11
                    num_unformatted += 1
                else:
                    interpolated_responses = [
                        formatted_output[formatted_output.index(heads[i]) + len(heads[i]):
                                         formatted_output.index(heads[i + 1])].strip()
                        for i in range(11)
                    ]
                print(f"num_unformatted = {num_unformatted}")
                result.update({
                    "interpolated_responses": interpolated_responses,
                })
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        for i in range(11):
            results[i]["rouge_a"] = compute_rouge(
                responses=[result["interpolated_responses"][i] for result in results],
                references=[result["reference_a"] for j, result in enumerate(results)],
            )
            results[i]["rouge_b"] = compute_rouge(
                responses=[result["interpolated_responses"][i] for result in results],
                references=[result["reference_b"] for j, result in enumerate(results)],
            )
        return results


class TestGroundForVAE(TestGroundBase):
    def load_model(self):
        if not hasattr(self, "model"):
            load_kwargs = {"torch_dtype": torch.float16, "device_map": self.device}
            model = LlamaForCVAE.from_pretrained(self.args.cvae_model_path, **load_kwargs).eval()
            self.model = model
        if not hasattr(self, "tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(self.args.cvae_model_path)
            self.tokenizer = tokenizer

    def strategy_name(self):
        return (f"{self.args.vae_type}-{self.args.add_skip_connection}-{self.args.add_gskip_connection}")

    def visualize_all_latent(self):
        self.load_model()
        dataset = One2ManyDataset(
            tokenizer_path=self.args.base_model_name,
            dataset_name=self.args.dataset_name,
            usage="test", total_many=32, mini_many=32
        )
        print(f"dataset: {dataset}")

        def compute_grid_logprobs(posterior_mean, posterior_logvar, bound=5, pixels=300):
            # posterior_mean, posterior_logvar 均为 [batch_size, dim_z]
            dim_z = posterior_mean.shape[1]
            zs = (torch.arange(pixels).float() / pixels * 2 - 1) * bound
            zs = zs.unsqueeze(1).repeat(1, dim_z).to(device=posterior_mean.device,
                                                     dtype=posterior_mean.dtype)  # [pixels, dim_z]
            logprobs = log_pdf(posterior_mean[None, :, :], posterior_logvar[None, :, :],
                               zs[:, None, :])  # [pixels, batch_size, dim_z]
            logprobs = torch.where(logprobs < -100, -100, logprobs)
            logprobs = exp_mean_log(logprobs, dim=1)  # [pixels, dim_z]
            return logprobs

        def visualize_logprobs(logprobs, dimx=0, dimy=1, bound=5, pixels=300, figname=None):
            # Convert log-probabilities to 2D probabilities
            probs2d = (logprobs[:, dimx, None] + logprobs[None, :, dimy]).exp()
            # Create grid coordinates
            zs = (torch.arange(pixels).float() / pixels * 2 - 1) * bound
            xx, yy = torch.meshgrid(zs, zs, indexing='ij')
            # Convert to numpy for plotting
            probs2d_np = probs2d.cpu().numpy()
            xx_np = xx.cpu().numpy()
            yy_np = yy.cpu().numpy()
            # Plot
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            cmap = plt.get_cmap('jet')
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                'truncated_jet',
                cmap(np.linspace(0.2, 0.8, 256))
            )
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(xx_np, yy_np, probs2d_np, shading='auto', vmin=0, vmax=0.25, cmap=new_cmap)
            plt.colorbar(label='Probability Density')
            plt.xlabel(f'Latent Dimension {dimx}')
            plt.ylabel(f'Latent Dimension {dimy}')
            plt.title('2D Probability Density in Latent Space')
            plt.grid(True, alpha=0.25)
            # Save to file
            plt.savefig(figname or f'latent-{self.args.dataset_name}-{self.args.vae_type}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

        all_posterior_mean = []
        all_grid_logprobs = []
        for index in tqdm(range(0, len(dataset), 10 if args.small_test else 1),
                          desc="test_encoder_language_latent"):
            batch_item = dataset[index]
            input_ids = batch_item["input_ids"]
            posterior_input_ids = input_ids
            # [batch_size, seq_len]
            posterior_input_ids, posterior_attention_mask = right_padding_to_left_padding(
                posterior_input_ids,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            )
            # [batch_size, dim_z]
            with torch.no_grad():
                posterior_mean, posterior_logvar = self.model.latent_encoder(
                    posterior_input_ids=posterior_input_ids.to(self.device),
                    posterior_attention_mask=posterior_attention_mask.to(self.device),
                )
                all_posterior_mean.append(posterior_mean)
                grid_logprobs = compute_grid_logprobs(posterior_mean, posterior_logvar)
                all_grid_logprobs.append(grid_logprobs)
        all_posterior_mean = torch.cat(all_posterior_mean, dim=0)  # [N * batch_size, dim_z]
        variances = torch.var(all_posterior_mean, dim=0)
        dimx, dimy = torch.topk(variances, k=2).indices
        visualize_logprobs(exp_mean_log(torch.stack(all_grid_logprobs, dim=0), dim=0), dimx, dimy)

    def test_encoder_language_latent(self, n_samples=16, verbose=1):
        cache_file = os.path.join(self.args.cvae_model_path, (
            "small_" if args.small_test else "") + f"test_encoder_language_latent_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            au_thresh = [0.03, 0.1, 0.3]  # [0.01, 0.03, 0.1]
            cu_thresh = [0.03, 0.1, 0.3]  # [0.01, 0.03, 0.1]
            results = []
            for index in tqdm(range(0, len(dataset), 10 if args.small_test else 1),
                              desc="test_encoder_language_latent"):
                batch_item = dataset[index]
                input_ids = batch_item["input_ids"]
                posterior_input_ids = input_ids
                # [batch_size, seq_len]
                posterior_input_ids, posterior_attention_mask = right_padding_to_left_padding(
                    posterior_input_ids,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id,
                )
                # [batch_size, dim_z]
                posterior_mean, posterior_logvar = self.model.latent_encoder(
                    posterior_input_ids=posterior_input_ids.to(self.device),
                    posterior_attention_mask=posterior_attention_mask.to(self.device),
                )
                # [batch_size, dim_z, n_samples]
                zs = sampling(posterior_mean, posterior_logvar, n_samples)

                # KL(q(z|x,y)||p(z|x))
                kl = 0.5 * (posterior_mean.pow(2) + posterior_logvar.exp() - posterior_logvar - 1).sum(dim=1).mean(
                    dim=0)

                # MI(y,z;q) =
                # E_{y \sim LLM(y|x)} E_{z \sim q(z|x,y)} [\log{q(y,z|x)} - \log{q(y|x)} - \log{q(z|x)}]
                # \log{q(y,z|x)} - \log{q(y|x)} - \log{q(z|x)}
                # joint_mi
                log_q_yz_x = log_pdf(posterior_mean, posterior_logvar, zs).sum(dim=1) - np.log(dataset.total_many)
                log_q_y_x = - np.log(dataset.total_many)
                log_q_z_x = exp_mean_log(
                    log_pdf(posterior_mean[None, :, :], posterior_logvar[None, :, :], zs[:, None, :, :]).sum(dim=2),
                    dim=1)
                joint_mi = (log_q_yz_x - log_q_y_x - log_q_z_x).mean(dim=0).mean(dim=-1)

                log_q_yz_x = log_pdf(posterior_mean, posterior_logvar, zs) - np.log(dataset.total_many)
                log_q_y_x = - np.log(dataset.total_many)
                log_q_z_x = exp_mean_log(
                    log_pdf(posterior_mean[None, :, :], posterior_logvar[None, :, :], zs[:, None, :, :]), dim=1)
                marginal_mi = (log_q_yz_x - log_q_y_x - log_q_z_x).sum(dim=1).mean(dim=0).mean(dim=-1)

                # AU and CU
                au = [(torch.var(posterior_mean, dim=0) > thresh).sum().item() for thresh in au_thresh]
                marginal_kl = (log_q_z_x - log_pdf(torch.zeros_like(posterior_mean), torch.zeros_like(posterior_logvar),
                                                   zs)).mean(dim=-1).mean(dim=0)
                cu = [(marginal_kl < thresh).sum().item() for thresh in cu_thresh]

                results.append({
                    "index": index,
                    "kl": kl.cpu().item(),
                    "joint_mi": joint_mi.cpu().item(),
                    "marginal_mi": marginal_mi.cpu().item(),
                    "au": au,
                    "cu": cu
                })
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        if verbose:
            all_values = [result["kl"] for result in results]
            print(f"encoder language-latent kl: {np.mean(all_values):.2f} +/- {np.std(all_values):.2f}")
            all_values = [result["joint_mi"] for result in results]
            print(f"encoder language-latent joint_mi: {np.mean(all_values):.2f} +/- {np.std(all_values):.2f}")
            all_values = [result["marginal_mi"] for result in results]
            print(f"encoder language-latent marginal_mi: {np.mean(all_values):.2f} +/- {np.std(all_values):.2f}")
            all_values = np.array([result["au"] for result in results])
            all_values_mean = "\t".join([f"{value:.4f}" for value in all_values.mean(axis=0)])
            all_values_std = "\t".join([f"{value:.4f}" for value in all_values.std(axis=0)])
            print(f"encoder language-latent au: {all_values_mean} +/- {all_values_std}")
            all_values = np.array([result["cu"] for result in results])
            all_values_mean = "\t".join([f"{value:.4f}" for value in all_values.mean(axis=0)])
            all_values_std = "\t".join([f"{value:.4f}" for value in all_values.std(axis=0)])
            print(f"encoder language-latent cu: {all_values_mean} +/- {all_values_std}")
        return results

    def test_prior_inference(self):
        cache_file = os.path.join(self.args.cvae_model_path,
                                  ("small_" if args.small_test else "") + f"test_prior_greedy_inference_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": False,
                # "temperature": self.args.temperature,
                "latent_sampling": True,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            results = {}
            indexes = range(0, len(dataset), 10 if args.small_test else 1)
            for index in tqdm(indexes, desc="test_prior_inference"):
                prompt = dataset.one2many_inference[index]["prompt"]
                reference = dataset.preprocessed_data[index]["response"]
                references = dataset.one2many_inference[index]["responses"]
                if str(index) in cached_results:
                    result: dict = cached_results[str(index)]
                    if not all([result.get(k, None) == v for k, v in {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references
                    }.items()]):
                        result = {}
                else:
                    result = {}
                if result == {}:
                    inputs = dataset.tokenizer([prompt], return_tensors="pt")
                    inputs = {k: v.cuda().repeat(len(references), 1) for k, v in inputs.items()}
                    prompt_length = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        response_ids = self.model.generate(
                            **inputs,
                            **generation_config,
                        )
                    responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                               skip_special_tokens=False)
                    cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
                    responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in responses]
                    responses = [cut_tail_after(response, eot_token) for response in responses]
                    responses = ["." if len(response) == 0 else response for response in responses]
                    references = [cut_tail_after(reference, dataset.tokenizer.eos_token) for reference in references]
                    references = [cut_tail_after(reference, eot_token) for reference in references]
                    # mauve_score, _ = compute_mauve(responses, references, n_clusters=8)
                    # distinct = compute_distinct(responses=responses)
                    # bleu = compute_bleu(responses=responses, references=[references for _ in responses])
                    # rouge = compute_rouge(responses=responses, references=[references for _ in responses])
                    # selfbleu = compute_self_bleu(responses=responses)
                    result = {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references,
                        "responses": responses,
                        # "mauve_score": mauve_score,
                        # "distinct": distinct,
                        # "bleu": bleu,
                        # "rouge": rouge,
                        # "selfbleu": selfbleu,
                    }
                results[index] = result
                if len(results) % (len(indexes) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        results = list(results.values())
        # mauve_score = np.mean([result["mauve_score"] for result in results])
        # distinct = {k:np.mean([result["distinct"][k] for result in results])
        #            for k in ['distinct-1', 'distinct-2', 'distinct-3', 'distinct-4']}
        # bleu = {k:np.mean([result["bleu"][k] for result in results])
        #        for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}
        # rouge = {k:np.mean([result["rouge"][k] for result in results])
        #         for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
        # selfbleu = {k:np.mean([result["selfbleu"][k] for result in results])
        #             for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}

        # update in v6.2
        # re-compute rouge scores
        # rouge = compute_rouge_mp(
        #    responses = ["\n".join(result["responses"]) for result in results],
        #    references = ["\n".join(result["references"]) for result in results],
        # )

        mauve_score = np.mean(compute_mauve_for_results(results))
        rouge = compute_group_rouge_for_results(results)
        distinct = compute_distinct_for_results(results)
        selfbleu = compute_selfbleu_for_results(results)
        results = {
            "mauve_score": mauve_score,
            "distinct": distinct,
            # "bleu": bleu,
            "selfbleu": selfbleu,
            "rouge": rouge,
        }
        return results

    def test_prior_inference_with_different_temperature(self):
        temperatures = [0.1, 0.4, 0.7, 1.0]
        cache_file = os.path.join(self.args.cvae_model_path, (
            "small_" if args.small_test else "") + f"test_prior_temperature_inference_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 32,
                "do_sample": True,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
                "latent_sampling": True,
            }
            results = {}
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            indexes = range(0, len(dataset), 10 if args.small_test else 1)
            for temperature in temperatures:
                temperature_repr = f"{int(temperature * 10):02d}"
                results[temperature_repr] = {}
                for index in tqdm(indexes, desc=f"test_prior_temperature_inference_{temperature_repr}"):
                    prompt = dataset.one2many_inference[index]["prompt"]
                    reference = dataset.preprocessed_data[index]["response"]
                    references = dataset.one2many_inference[index]["responses"]
                    references = [cut_tail_after(reference, dataset.tokenizer.eos_token) for reference in references]
                    references = [cut_tail_after(reference, eot_token) for reference in references]
                    result = {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references
                    }
                    if temperature_repr in cached_results and str(index) in cached_results[temperature_repr]:
                        cached_result = cached_results[temperature_repr][str(index)]
                        if all([cached_result.get(k, None) == v for k, v in result.items()]):
                            result["responses"] = cached_result["responses"]
                    if "responses" not in result:
                        inputs = dataset.tokenizer([prompt], return_tensors="pt")
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        prompt_length = inputs["input_ids"].shape[1]
                        with torch.no_grad():
                            response_ids = self.model.generate(
                                **inputs,
                                **generation_config,
                                temperature=temperature,
                            )
                        responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                                   skip_special_tokens=False)
                        responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in responses]
                        responses = [cut_tail_after(response, eot_token) for response in responses]
                        responses = ["." if len(response) == 0 else response for response in responses]
                        result["responses"] = responses
                    results[temperature_repr][index] = result
                with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                    f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        results["final_result"] = {
            "mauve": [np.mean(compute_mauve_for_results(results[f"{int(temperature * 10):02d}"].values())) for
                      temperature in temperatures],
            "rougeL": [compute_group_rouge_for_results(results[f"{int(temperature * 10):02d}"].values())["rougeLsum"]
                       for temperature in temperatures],
            "distinct": [compute_distinct_for_results(results[f"{int(temperature * 10):02d}"].values())["distinct-4"]
                         for temperature in temperatures],
            "selfbleu": [compute_selfbleu_for_results(results[f"{int(temperature * 10):02d}"].values())["bleu-4"] for
                         temperature in temperatures],
        }
        return results

    def test_posterior_inference(self):
        cache_file = os.path.join(self.args.cvae_model_path, (
            "small_" if args.small_test else "") + f"test_posterior_greedy_inference_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": False,
                # "temperature": self.args.temperature,
                "latent_sampling": False,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            results = {}
            indexes = range(0, len(dataset), 10 if args.small_test else 1)
            for index in tqdm(indexes, desc="test_posterior_inference"):
                prompt = dataset.one2many_inference[index]["prompt"]
                reference = dataset.preprocessed_data[index]["response"]
                references = dataset.one2many_inference[index]["responses"]
                if str(index) in cached_results:
                    result: dict = cached_results[str(index)]
                    if not all([result.get(k, None) == v for k, v in {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references
                    }.items()]):
                        result = {}
                else:
                    result = {}
                if result == {}:
                    prior_inputs = dataset.tokenizer([prompt], return_tensors="pt")
                    posterior_inputs = dataset.collate_fn([dataset[index]])
                    inputs = {
                        "input_ids": prior_inputs["input_ids"].cuda().repeat(len(references), 1),
                        "attention_mask": prior_inputs["attention_mask"].cuda().repeat(len(references), 1),
                        "posterior_input_ids": posterior_inputs["posterior_input_ids"].cuda(),
                        "posterior_attention_mask": posterior_inputs["posterior_attention_mask"].cuda(),
                    }
                    prompt_length = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        response_ids = self.model.generate(
                            **inputs,
                            **generation_config,
                        )
                    responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                               skip_special_tokens=False)
                    cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
                    responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in responses]
                    responses = [cut_tail_after(response, eot_token) for response in responses]
                    responses = ["." if len(response) == 0 else response for response in responses]
                    references = [cut_tail_after(reference, dataset.tokenizer.eos_token) for reference in references]
                    references = [cut_tail_after(reference, eot_token) for reference in references]
                    # mauve_score, _ = compute_mauve(responses, references, n_clusters=8)
                    # distinct = compute_distinct(responses=responses)
                    # bleu = compute_bleu(responses=responses, references=[[reference] for reference in references])
                    # rouge = compute_rouge(responses=responses, references=[[reference] for reference in references])
                    # debertascore = compute_debertascore(responses=responses, references=[[reference] for reference in references])
                    # selfbleu = compute_self_bleu(responses=responses)
                    result = {
                        "prompt": prompt,
                        "reference": reference,
                        "references": references,
                        "responses": responses,
                        # "mauve_score": mauve_score,
                        # "distinct": distinct,
                        # "bleu": bleu,
                        # "rouge": rouge,
                        # "debertascore": debertascore,
                        # "selfbleu": selfbleu
                    }
                results[index] = result
                if index % (len(indexes) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        results = list(results.values())
        # mauve_score = np.mean([result["mauve_score"] for result in results])
        # distinct = {k:np.mean([result["distinct"][k] for result in results])
        #            for k in ['distinct-1', 'distinct-2', 'distinct-3', 'distinct-4']}
        # bleu = {k:np.mean([result["bleu"][k] for result in results])
        #        for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}
        # rouge = {k:np.mean([result["rouge"][k] for result in results])
        #         for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
        # debertascore = {k:np.mean([result["debertascore"][k] for result in results])
        #         for k in ['precision', 'recall', 'f1']}
        # selfbleu = {k:np.mean([result["selfbleu"][k] for result in results])
        #             for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']}

        mauve_score = np.mean(compute_mauve_for_results(results))
        rouge = compute_singular_rouge_for_results(results)
        distinct = compute_distinct_for_results(results)
        selfbleu = compute_selfbleu_for_results(results)
        results = {
            "mauve_score": mauve_score,
            "distinct": distinct,
            # "bleu": bleu,
            "rouge": rouge,
            # "debertascore": debertascore,
            "selfbleu": selfbleu,
        }
        return results

    def get_posterior_latent_and_probs(self, usage="validation", num_shots="all"):
        cache_file = os.path.join(self.args.cvae_model_path, (
            "small_" if args.small_test else "") + f"get_posterior_latent_and_probs_{usage}_{num_shots}_{version}.json")
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = {}
            self.load_model()
            dataset = ComparisonDataset(tokenizer_path=args.base_model_name, dataset_name=args.dataset_name,
                                        usage=usage)
            results = {}
            if num_shots == "all":
                indexes = range(len(dataset.one2many_inference_with_logits))
            elif num_shots == "active_100":
                dataset.compute_index_label_mutual_information()
                indexes = [(mi, index) for index, mi in enumerate(dataset.index_label_mi)]
                indexes = [index for mi, index in sorted(indexes)[-100:]]
            elif num_shots == "active_10":
                dataset.compute_index_label_mutual_information()
                indexes = [(mi, index) for index, mi in enumerate(dataset.index_label_mi)]
                indexes = [index for mi, index in sorted(indexes)[-10:]]
            else:
                indexes = list(range(0, len(dataset.one2many_inference_with_logits),
                                     len(dataset.one2many_inference_with_logits) // int(num_shots)))[:int(num_shots)]
            for index in tqdm(indexes):
                data = dataset.one2many_inference_with_logits[index]
                prompt = data["prompt"]
                responses = data["responses"]

                if str(index) in cached_results:
                    result: dict = cached_results[str(index)]
                    if not all([result.get(k, None) == v for k, v in {
                        "prompt": prompt,
                        "responses": responses,
                    }.items()]):
                        result = {}
                else:
                    result = {}

                if result == {}:
                    prompt_ids = dataset.tokenizer(
                        [prompt], return_tensors="pt"
                    ).input_ids.long()
                    dataset.tokenizer.truncation_side = "right"
                    dataset.tokenizer.padding_side = "right"
                    input_ids = dataset.tokenizer(
                        [prompt + response for response in responses],
                        truncation=True, return_tensors="pt", padding="longest",
                        max_length=prompt_ids.shape[1] + dataset.max_output_length
                    ).input_ids.long()
                    exceeded_prompt_length = prompt_ids.shape[1] - dataset.max_input_length
                    if exceeded_prompt_length > 0:
                        input_ids = input_ids[:, exceeded_prompt_length:]
                        prompt_ids = prompt_ids[:, exceeded_prompt_length:]
                    responses_ids = dataset.tokenizer(
                        responses,
                        truncation=True, return_tensors="pt", padding="longest",
                        max_length=dataset.max_output_length
                    ).input_ids.long()
                    if responses_ids.shape[1] < dataset.max_output_length:
                        eos_token_ids = torch.LongTensor([[dataset.tokenizer.eos_token_id]] * input_ids.shape[0])
                        input_ids = torch.cat([input_ids, eos_token_ids], dim=1)

                    posterior_input_ids = input_ids
                    # [batch_size, seq_len]
                    posterior_input_ids, posterior_attention_mask = right_padding_to_left_padding(
                        posterior_input_ids,
                        self.tokenizer.eos_token_id,
                        self.tokenizer.pad_token_id,
                    )
                    # [batch_size, dim_z]
                    posterior_mean, posterior_logvar = self.model.latent_encoder(
                        posterior_input_ids=posterior_input_ids.to(self.device),
                        posterior_attention_mask=posterior_attention_mask.to(self.device),
                    )
                    # [batch_size, num_labels]
                    logits = torch.FloatTensor(data["logits"])
                    probs = F.softmax(logits, dim=1)
                    result = {
                        "prompt": prompt,
                        "responses": responses,
                        "probs": round_tensor(probs),
                        "logits": round_tensor(logits),
                        "posterior_mean": round_tensor(posterior_mean),
                        "posterior_logvar": round_tensor(posterior_logvar),
                    }
                results[index] = result
                if index % (len(indexes) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        return results

    def test_latent_guided_generation(self, num_valid_shots="10", num_test_shots="active_100"):
        cache_file = os.path.join(self.args.cvae_model_path, (
            "small_" if args.small_test else "") + f"test_latent_guided_generation_{num_valid_shots}_{num_test_shots}_{version}.json")
        num_labels = len(get_dataset_indexed_labels(self.args.dataset_name))
        if os.path.exists(cache_file) and not self.args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            # Step 1. Validation Cases
            valid_posterior_latent_and_logits = self.get_posterior_latent_and_probs(usage="validation", num_shots=num_valid_shots)
            test_posterior_latent_and_logits = self.get_posterior_latent_and_probs(usage="test", num_shots=num_test_shots)

            # Step 2. Latent Vector Encoding
            all_corrcoef = []
            for posterior_result in valid_posterior_latent_and_logits.values():
                logits = torch.FloatTensor(posterior_result["logits"]) # response labels - [batch_size, num_labels]
                mean = torch.FloatTensor(posterior_result["posterior_mean"]) # latent vectors - [batch_size, dim_z]
                inputs = torch.cat([F.softmax(logits, dim=1), mean], dim=1).T
                corrcoef = torch.corrcoef(inputs)
                dim_z = mean.shape[1]
                assert corrcoef.shape == (num_labels + dim_z, num_labels + dim_z), corrcoef.shape
                all_corrcoef.append(corrcoef[:num_labels, -dim_z:]) # Pearson correlation coefficient between response labels and latent dimensions
            all_corrcoef = torch.stack(all_corrcoef, dim=0)
            all_corrcoef = torch.where(torch.isnan(all_corrcoef), 0.0, all_corrcoef)
            avg_corrcoef = all_corrcoef.mean(dim=0)
            unbiased_corrcoef = avg_corrcoef - avg_corrcoef.mean(dim=0, keepdim=True)
            normalized_corrcoef = unbiased_corrcoef / unbiased_corrcoef.pow(2).sum(dim=-1, keepdim=True).sqrt()
            print(f"normalized_corrcoef:\n{normalized_corrcoef}")

            # Step 3. Latent Vector Normalizing and Decoding & Evaluation
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": True,
                "temperature": self.args.temperature,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            results = []
            for posterior_result in tqdm(list(test_posterior_latent_and_logits.values()),
                                         desc="performing manipulated generation"):
                prompt = posterior_result["prompt"]
                references = posterior_result["responses"]
                references = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in references]
                references = [cut_tail_after(response, eot_token) for response in references]
                references_logits = torch.FloatTensor(posterior_result["logits"])
                references_probs = F.softmax(references_logits, dim=1)
                references_posterior_mean = torch.FloatTensor(posterior_result["posterior_mean"])
                references_posterior_logvar = torch.zeros_like(references_posterior_mean).fill_(-100)

                inputs = dataset.tokenizer([prompt], return_tensors="pt")
                inputs = {k: v.repeat(32, 1).cuda() for k, v in inputs.items()}
                prompt_length = inputs["input_ids"].shape[1]

                inputs["latent_mean_logvar"] = (references_posterior_mean.cuda(), references_posterior_logvar.cuda())
                with torch.no_grad():
                    response_ids = self.model.generate(**inputs, **generation_config)
                reconstructed_responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:], skip_special_tokens=False)
                reconstructed_responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in reconstructed_responses]
                reconstructed_responses = [cut_tail_after(response, eot_token) for response in reconstructed_responses]
                reconstructed_rouge = compute_rouge(responses=reconstructed_responses, references=[references for _ in reconstructed_responses])
                reconstructed_debertascore = compute_debertascore(responses=reconstructed_responses,
                                                                  references=[references for _ in
                                                                              reconstructed_responses])
                reconstructed_logits = self.test_logits(prompt, reconstructed_responses)
                reconstructed_probs = F.softmax(reconstructed_logits, dim=-1)
                result = {
                    "prompt": prompt,
                    "references": references,
                    "references_probs": round_tensor(references_probs),
                    "reconstructed_responses": reconstructed_responses,
                    "reconstructed_rouge": reconstructed_rouge,
                    "reconstructed_debertascore": reconstructed_debertascore,
                    "reconstructed_probs": round_tensor(reconstructed_probs),
                    "reconstructed_probs_delta": round_tensor(reconstructed_probs - references_probs),
                }
                for k in ks:
                    for index_feature in range(num_labels):
                        inputs["latent_mean_logvar"] = (
                            (references_posterior_mean + normalized_corrcoef[index_feature, :] * k).cuda(),
                            references_posterior_logvar.cuda()
                        )
                        with torch.no_grad():
                            response_ids = self.model.generate(
                                **inputs,
                                **generation_config,
                            )
                        manipulated_responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                                               skip_special_tokens=False)
                        manipulated_responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in
                                                 manipulated_responses]
                        manipulated_responses = [cut_tail_after(response, eot_token) for response in
                                                 manipulated_responses]
                        manipulated_logits = self.test_logits(prompt, manipulated_responses)
                        manipulated_probs = F.softmax(manipulated_logits, dim=1)
                        result[f"manipulated_responses-{index_feature}@{k}"] = manipulated_responses
                        result[f"manipulated_rouge-{index_feature}@{k}"] = compute_rouge(
                            responses=manipulated_responses,
                            references=[references for _ in manipulated_responses]
                        )
                        result[f"manipulated_debertascore-{index_feature}@{k}"] = compute_debertascore(
                            responses=manipulated_responses,
                            references=[references for _ in manipulated_responses]
                        )
                        result[f"manipulated_probs-{index_feature}@{k}"] = round_tensor(manipulated_probs)
                        result[f"manipulated_win_rate-{index_feature}@{k}"] = round_tensor(
                            (manipulated_probs > reconstructed_probs).float().mean(dim=0))
                        result[f"manipulated_probs_delta-{index_feature}@{k}"] = round_tensor(
                            (manipulated_probs - reconstructed_probs).mean(dim=0))
                results.append(result)

            for index_feature in range(num_labels):
                for k in ks:
                    manipulated_responses = [result["prompt"] + response for result in results
                                             for response in result[f"manipulated_responses-{index_feature}@{k}"]]
                    results[0][f"manipulated_ppl-{index_feature}@{k}"] = compute_perplexity(
                        responses=manipulated_responses)
            # ppl
            references = [result["prompt"] + reference for result in results for reference in result["references"]]
            results[0]["references_ppl"] = compute_perplexity(responses=references)
            reconstructed_responses = [result["prompt"] + response for result in results
                                       for response in result["reconstructed_responses"]]
            results[0]["reconstructed_ppl"] = compute_perplexity(responses=reconstructed_responses)
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))
        references_probs = torch.Tensor([result[f"references_probs"] for result in results]).mean(dim=1).mean(dim=0)
        print(f"references_probs:\n{references_probs}")
        reconstructed_probs = torch.Tensor([result[f"reconstructed_probs"] for result in results]).mean(dim=1).mean(
            dim=0)
        print(f"reconstructed_probs:\n{reconstructed_probs}")
        for index_feature in range(num_labels):
            print(f"feature-{index_feature}:")
            print(f"ks: {ks}")
            manipulated_probs = torch.Tensor([
                [result[f"manipulated_probs-{index_feature}@{k}"] for result in results]
                for k in ks
            ]).mean(dim=2).mean(dim=1)
            manipulated_win_rate = torch.Tensor([
                [result[f"manipulated_win_rate-{index_feature}@{k}"] for result in results]
                for k in ks
            ]).mean(dim=1)
            manipulated_probs_delta = torch.Tensor([
                [result[f"manipulated_probs_delta-{index_feature}@{k}"] for result in results]
                for k in ks
            ]).mean(dim=1)
            print(f"manipulated_probs:\n{manipulated_probs}")
            print(f"manipulated_win_rate:\n{manipulated_win_rate}")
            print(f"manipulated_probs_delta:\n{manipulated_probs_delta}")

        for k in ks:
            maximized_probs = [
                np.mean([probs[index_feature] for result in results
                         for probs in result[f"manipulated_probs-{index_feature}@{k}"]])
                for index_feature in range(num_labels)
            ]
            ppl = np.mean([
                results[0][f'manipulated_ppl-{index_feature}@{k}']
                for index_feature in range(num_labels)
            ])
            print(f"maximized probs @ {k}: {round_tensor(maximized_probs)}")
            print(f"maximized probs @ {k} sum: {sum(maximized_probs):.3f}")
            print(f"ppl @ {k}: {ppl}")

        return results

    def test_interpolation(self, num_test_shots="all"):
        cache_file = os.path.join(
            self.args.cvae_model_path,
            (
                "small_" if args.small_test else "") + f"test_interpolation_{num_test_shots}_{version}_temp{int(self.args.temperature * 10):02d}.json"
        )
        if os.path.exists(cache_file) and not self.args.erase and False:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not self.args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = []
            test_posterior_latent_and_logits = self.get_posterior_latent_and_probs(
                usage="test", num_shots=num_test_shots)
            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": True,
                "temperature": self.args.temperature,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
                "latent_sampling": False,
            }
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            results = []
            posterior_results = list(test_posterior_latent_and_logits.values())
            for index, posterior_result in enumerate(tqdm(posterior_results, desc="performing interpolation")):
                prompt = posterior_result["prompt"]
                references = posterior_result["responses"]
                references = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in references]
                references = [cut_tail_after(response, eot_token) for response in references]
                references_logits = torch.FloatTensor(posterior_result["logits"])

                random.seed(int(hashlib.sha256(prompt.encode()).hexdigest(), 16))
                a, b = random.sample(range(32), 2)
                reference_a, reference_b = references[a], references[b]
                z_a = torch.FloatTensor(posterior_result["posterior_mean"])[a, :]
                z_b = torch.FloatTensor(posterior_result["posterior_mean"])[b, :]
                lamda = torch.arange(0, 1.01, 0.1)
                zs_lamda = z_a[None, :] * (1 - lamda[:, None]) + z_b[None, :] * lamda[:, None]

                logvar_a = torch.FloatTensor(posterior_result["posterior_logvar"])[a, :]
                logvar_b = torch.FloatTensor(posterior_result["posterior_logvar"])[b, :]
                logvar_lamda = logvar_a[None, :] * (1 - lamda[:, None]) + logvar_b[None, :] * lamda[:, None]

                if (len(cached_results) > 0
                        and cached_results[0].get("prompt", None) == prompt
                        and cached_results[0].get("reference_a", None) == reference_a
                        and cached_results[0].get("reference_b", None) == reference_b):
                    result = cached_results.pop(0)
                else:
                    inputs = dataset.tokenizer([prompt], return_tensors="pt")
                    inputs = {k: v.repeat(lamda.shape[0], 1).cuda() for k, v in inputs.items()}
                    manipulated_posterior_mean = zs_lamda.cuda()
                    manipulated_posterior_logvar = logvar_lamda.cuda()
                    inputs["latent_mean_logvar"] = (manipulated_posterior_mean, manipulated_posterior_logvar)

                    with torch.no_grad():
                        response_ids = self.model.generate(
                            **inputs,
                            **generation_config,
                        )
                    prompt_length = inputs["input_ids"].shape[1]
                    interpolated_responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                                            skip_special_tokens=False)
                    interpolated_responses = [cut_tail_after(response, dataset.tokenizer.eos_token) for response in
                                              interpolated_responses]
                    interpolated_responses = [cut_tail_after(response, eot_token) for response in
                                              interpolated_responses]
                    assert len(interpolated_responses) == lamda.shape[0]

                    result = {
                        "prompt": prompt,
                        "reference_a": reference_a,
                        "reference_b": reference_b,
                        "interpolated_responses": interpolated_responses,
                    }
                    if index % 100 == 0:
                        print("-" * 100)
                        print(json.dumps(result, ensure_ascii=False, indent=4))
                results.append(result)
                if len(results) % (len(posterior_results) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        for result in tqdm(results):
            logits = self.test_logits(
                prompt=result["prompt"],
                responses=[result["reference_a"], result["reference_b"]] + result["interpolated_responses"]
            )
            probs = F.softmax(logits, dim=-1)
            probs_a, probs_b = probs[0, :], probs[1, :]
            probs_avg = (probs_a + probs_b) / 2
            entropy = lambda probs: (probs * probs.log()).sum()
            jsd = entropy(probs_avg) - entropy(probs_a) / 2 - entropy(probs_b) / 2
            result["probs"] = round_tensor(probs)
            result["jsd"] = round(jsd.item(), 3)

        with open(cache_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))

        for i in range(11):
            results[i]["rouge_a"] = compute_rouge(
                responses=[result["interpolated_responses"][i] for result in results],
                references=[result["reference_a"] for j, result in enumerate(results)],
            )
            results[i]["rouge_b"] = compute_rouge(
                responses=[result["interpolated_responses"][i] for result in results],
                references=[result["reference_b"] for j, result in enumerate(results)],
            )

        return results

    def test_yelp_interpolation(self):
        if not self.args.dataset_name == "yelp":
            return None
        num_shots = "all" if not self.args.small_test else "100"

        cache_file = os.path.join(
            self.args.cvae_model_path, f"test_yelp_interpolation_{num_shots}_{version}.json"
        )
        if os.path.exists(cache_file) and not self.args.erase and False:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            if os.path.exists(cache_file + ".cache") and not self.args.erase:
                with open(cache_file + ".cache", "r", encoding='utf-8') as f:
                    cached_results = json.loads(f.read())
            else:
                cached_results = []

            self.load_model()
            dataset = One2ManyDataset(
                tokenizer_path=self.args.base_model_name,
                dataset_name=self.args.dataset_name,
                usage="test", total_many=32, mini_many=32
            )
            generation_config = {
                "num_return_sequences": 1,
                "do_sample": True,
                "temperature": self.args.temperature,
                "max_new_tokens": dataset.max_output_length,
                "min_new_tokens": 2,
            }
            cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
            results = []
            test_posterior_latent_and_logits = list(
                self.get_posterior_latent_and_probs(usage="test", num_shots=num_shots).values())
            for result in test_posterior_latent_and_logits:
                probs = np.array(result["probs"])
                averaged_ratings = (probs * (np.arange(5) + 1)).sum(axis=1)
                if not averaged_ratings.std() > np.linspace(1, 5, 32).std():
                    continue
                lowest_rated_index = averaged_ratings.argmin()
                highest_rated_index = averaged_ratings.argmax()

                prompt = result["prompt"]
                references = result["responses"]
                reference_a, reference_b = references[lowest_rated_index], references[highest_rated_index]
                z_a = torch.FloatTensor(result["posterior_mean"])[lowest_rated_index, :]
                z_b = torch.FloatTensor(result["posterior_mean"])[highest_rated_index, :]
                lamda = torch.arange(0, 1.01, 0.1)
                zs_lamda = z_a[None, :] * (1 - lamda[:, None]) + z_b[None, :] * lamda[:, None]

                if (len(cached_results) > 0
                        and cached_results[0].get("prompt", None) == prompt
                        and cached_results[0].get("reference_a", None) == reference_a
                        and cached_results[0].get("reference_b", None) == reference_b):
                    result = cached_results.pop(0)
                else:
                    inputs = dataset.tokenizer([prompt], return_tensors="pt")
                    inputs = {k: v.repeat(lamda.shape[0], 1).cuda() for k, v in inputs.items()}
                    manipulated_posterior_mean = zs_lamda.cuda()
                    manipulated_posterior_logvar = torch.empty_like(manipulated_posterior_mean).fill_(-100)
                    inputs["latent_mean_logvar"] = (manipulated_posterior_mean, manipulated_posterior_logvar)

                    with torch.no_grad():
                        response_ids = self.model.generate(
                            **inputs,
                            **generation_config,
                        )
                    prompt_length = inputs["input_ids"].shape[1]
                    interpolated_responses = dataset.tokenizer.batch_decode(response_ids[:, prompt_length:],
                                                                            skip_special_tokens=False)
                    interpolated_responses = [cut_tail_after(response, dataset.tokenizer.eos_token)
                                              for response in interpolated_responses]
                    interpolated_responses = [cut_tail_after(response, eot_token)
                                              for response in interpolated_responses]
                    assert len(interpolated_responses) == lamda.shape[0]

                    interpolated_logits = self.test_logits(prompt=prompt, responses=interpolated_responses)
                    interpolated_probs = F.softmax(interpolated_logits, dim=1)
                    interpolated_ratings = round_tensor((interpolated_probs.numpy() * (np.arange(5) + 1)).sum(axis=1))

                    result = {
                        "prompt": prompt,
                        "reference_a": reference_a,
                        "reference_b": reference_b,
                        "interpolated_responses": interpolated_responses,
                        "interpolated_ratings": interpolated_ratings,
                    }
                results.append(result)
                if len(results) % (len(test_posterior_latent_and_logits) // 10) == 0:
                    with open(cache_file + ".cache", "w", encoding='utf-8') as f:
                        f.write(json.dumps(results, ensure_ascii=False, indent=4))
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        print(f"number of results: {len(results)}")

        interpolated_ratings_array = np.array([result["interpolated_ratings"] for result in results])
        mean_ratings = np.mean(interpolated_ratings_array, axis=0)
        std_ratings = np.std(interpolated_ratings_array, axis=0)
        x_ticks = np.linspace(0.0, 1.0, 11)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(x_ticks, mean_ratings, marker='o', label='Mean Ratings', color='b')
        plt.errorbar(x_ticks, mean_ratings, yerr=std_ratings, fmt='o', color='r', label='Standard Deviation', capsize=5)
        plt.title('Ratings of Interpolated Responses')
        plt.xlabel('Interpolated Positions')
        plt.ylabel('Ratings')
        plt.xticks(x_ticks)  # 设置横坐标刻度
        plt.legend()
        plt.grid()
        plt.savefig(cache_file + ".png")

    def test_logits(self, prompt, responses):
        if type(responses) is str:
            responses = [responses]

        classifier_output_dir = os.path.join(root, f"cls/roberta-large-{args.dataset_name}")
        classifier_output_dir = get_last_checkpoint(classifier_output_dir)
        if not hasattr(self, "classifier") or not hasattr(self, "classification_tokenizer"):
            self.classifier = RobertaForSequenceClassification.from_pretrained(classifier_output_dir).cuda().eval()
            self.classification_tokenizer = AutoTokenizer.from_pretrained(classifier_output_dir)
        if not hasattr(self, "classification_dataset_for_api"):
            self.classification_dataset_for_api = ClassificationDataset(
                tokenizer_path=self.args.base_model_name, dataset_name=self.args.dataset_name, usage="test",
                classification_tokenizer_path=classifier_output_dir,
            )

        with torch.no_grad():
            batch_items = []
            for response in responses:
                batch_item = self.classification_dataset_for_api.__getitem__(
                    index=None, data={"prompt": prompt, "response": response}
                )
                batch_items.append(batch_item)
            inputs = self.classification_dataset_for_api.collate_fn(batch_items=batch_items)
            inputs = {k: v.cuda() if v is not None else v for k, v in inputs.items()}
            logits = self.classifier(**inputs, return_dict=True).logits
        return logits.cpu()

    def test_sampling_single(self):
        temperature = self.args.temperature
        cache_file = os.path.join(self.args.cvae_model_path,
                                  f"test_sampling_single_temp{int(temperature * 10):02d}_{version}.json")
        dataset = SFTDataset(
            tokenizer_path=self.args.base_model_name,
            dataset_name=self.args.dataset_name,
            usage="test",
        )
        if os.path.exists(cache_file) and not args.erase:
            with open(cache_file, "r", encoding='utf-8') as f:
                results = json.loads(f.read())
        else:
            self.load_model()
            if temperature >= 0:
                generation_config = {
                    "num_return_sequences": 1,
                    "do_sample": temperature > 0.0,
                    "temperature": temperature,
                    "max_new_tokens": dataset.max_output_length,
                    "min_new_tokens": 2,
                }
            else:
                generation_config = {
                    "num_return_sequences": 1,
                    "do_sample": False,
                    "num_beams": 16,
                    "max_new_tokens": dataset.max_output_length,
                    "min_new_tokens": 2,
                }
            results = []
            for index in tqdm(range(len(dataset)), desc="test sampling single"):
                prompt = dataset.data[index]["prompt"]
                reference = dataset.data[index]["response"]
                prompt_ids = dataset.preprocessed_data[index]["prompt_ids"]
                attention_mask = torch.ones_like(prompt_ids)
                response_ids = self.model.generate(
                    input_ids=prompt_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    **generation_config
                )
                response = self.tokenizer.batch_decode(response_ids, clean_up_tokenization_spaces=False)[0]
                results.append({
                    "prompt": prompt,
                    "reference": reference,
                    "temperature": temperature,
                    "response": response,
                })
            with open(cache_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))

        if args.small_test:
            results = [result for i, result in enumerate(results) if i % 10 == 0]

        cut_tail_after = lambda string, eos: string if eos not in string else string[:string.index(eos)]
        for result in results:
            assert result["response"].startswith("<|begin_of_text|>" + result["prompt"])
            result["response"] = result["response"][len("<|begin_of_text|>" + result["prompt"]):]
            result["response"] = cut_tail_after(result["response"], dataset.tokenizer.eos_token)
            result["response"] = cut_tail_after(result["response"], eot_token)
            result["reference"] = cut_tail_after(result["reference"], eot_token)

        print(f"dataset: {self.args.dataset_name}")
        print(f"temperature: {temperature}")

        from evaluation_dior import bleu, distinct, entropy
        prompts = [result["prompt"] for result in results]
        responses = [result["response"] for result in results]
        references = [result["reference"] for result in results]
        bleu(hyps=responses, refs=references)
        distinct(hyps=responses)
        entropy(hyps=responses)
        bertscore = compute_bert_score(responses, references)
        print(f"bertscore: {bertscore}")

        import fed
        # Load model
        model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
        # Evaluate
        keys = ["relevant", "understandable", "coherent", "diverse"]
        all_scores = []
        for prompt, response in zip(tqdm(prompts), responses):
            conversation = polish(prompt).replace(eot_token, " <|endoftext|> ") + polish(response)
            scores = fed.evaluate(conversation, model, tokenizer, aspects=" ".join(keys))
            all_scores.append(scores)
        for key in keys:
            print(f"response-{key}: {np.mean([scores[key] for scores in all_scores]):.3f}")
        print(f"response-overall: {np.mean([scores[key] for scores in all_scores for key in keys]):.3f}")


def initialize_judger(args) -> Union[TestGroundForVAE, TestGroundForSFT]:
    if args.vae_type == "SFT":
        if args.cvae_model_path is not None:
            args.sft_model_path = args.cvae_model_path  # get_last_checkpoint(args.cvae_model_path)
            args.output_dir = args.sft_model_path
        else:
            args.output_dir = os.path.join(root, f"sft/{args.base_model_name}-{args.dataset_name}")
            args.sft_model_path = get_last_checkpoint(args.output_dir)
        print(json.dumps(vars(args), ensure_ascii=False, indent=4))
        judger = TestGroundForSFT(args)
    else:
        if args.cvae_model_path is None:
            set_default_output_dir(args)
            args.cvae_model_path = get_last_checkpoint(args.output_dir)
        else:
            args.output_dir = args.cvae_model_path
        print(json.dumps(vars(args), ensure_ascii=False, indent=4))
        judger = TestGroundForVAE(args)
    return judger


if __name__ == "__main__":
    from train_cvae import get_parser

    parser = get_parser()
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--latent_sampling', type=int, default=0)
    parser.add_argument('--small_test', type=int, default=0)
    parser.add_argument('--erase', type=int, default=1)
    parser.add_argument('--test_latent', action="store_true")
    parser.add_argument('--test_prior_inference', action="store_true")
    parser.add_argument('--test_prior_inference_temperature', action="store_true")
    parser.add_argument('--test_posterior_inference', action="store_true")
    parser.add_argument('--test_interpolated_inference', action="store_true")
    parser.add_argument('--test_manipulated_inference', action="store_true")
    parser.add_argument('--test_dailydialog_benchmark', action="store_true")
    parser.add_argument('--visualize_latent_and_prob', action="store_true")
    parser.add_argument('--visualize_latent_and_position', action="store_true")
    args = parser.parse_args()

    judger = initialize_judger(args)

    if args.test_latent:
        ell_results = judger.test_encoder_language_latent()
        print(f"ell_results: {ell_results}")

    if args.test_prior_inference:
        prior_results = judger.test_prior_inference()
        print(f"prior_results: {prior_results}")

    if args.test_prior_inference_temperature:
        prior_temperature_results = judger.test_prior_inference_with_different_temperature()
        print(f"prior_temperature_results: {prior_temperature_results}")

    if args.test_posterior_inference:
        post_results = judger.test_posterior_inference()
        print(f"post_results: {post_results}")

    if args.test_interpolated_inference:
        interpolation_results = judger.test_interpolation(num_test_shots="all")
        print(f"interpolation_results: {interpolation_results}")

    if args.test_manipulated_inference:
        manipulated_results = judger.test_latent_guided_generation()
        print(f"manipulated_results: {manipulated_results}")

    if args.test_dailydialog_benchmark:
        judger.test_dailydialog()

    if args.visualize_latent_and_prob:
        pass
        # this visualization is done by another project: https://github.com/zhangjf-nlp/LatentDPO
        # we will migrate it into here immediately

    if args.visualize_latent_and_position:
        judger.visualize_all_latent()