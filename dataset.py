import json
import os.path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DownloadConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig

from config import root, workspace

def json_dumps_plus(data, **kwargs):
    data = {k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
    return json.dumps(data, **kwargs)


def right_padding_to_left_padding(input_ids, eos_token_id, pad_token_id):
    batch_size, seq_len = input_ids.shape
    is_pad_ids = input_ids.eq(pad_token_id)
    is_pad_ids_at_tail = is_pad_ids * (torch.cumsum(~is_pad_ids, dim=1) == torch.sum(~is_pad_ids, dim=1, keepdim=True))
    num_pad_ids_at_tail = is_pad_ids_at_tail.sum(dim=1)
    if eos_token_id == pad_token_id:
        num_pad_ids_at_tail = torch.where(num_pad_ids_at_tail>0, num_pad_ids_at_tail-1, num_pad_ids_at_tail)
    input_ids_left_padding = torch.stack(
        [torch.roll(input_ids[i, :], shifts=num_pad_ids_at_tail[i].item(), dims=0)
         for i in range(batch_size)], dim=0
    ).long()
    attention_mask_left_padding = (torch.arange(seq_len)[None, :] >= num_pad_ids_at_tail[:, None]).long()
    return input_ids_left_padding, attention_mask_left_padding


def create_attention_mask_after_first_eos(input_ids, eos_token_id=0):
    batch_size, seq_len = input_ids.shape
    is_not_eos = ~input_ids.eq(eos_token_id)
    is_tail_eos = is_not_eos.cumsum(dim=1).eq(is_not_eos.sum(dim=1, keepdim=True))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    for i in range(batch_size):
        is_eos_token_id = input_ids[i].eq(eos_token_id)
        if any(is_eos_token_id):
            eos_index = is_tail_eos[i].nonzero(as_tuple=True)[0].min().item()
            attention_mask[i, eos_index + 2:] = 0
    return attention_mask


def longest_reproducible_prefix_decoding(input_ids, tokenizer, max_offset=5):
    for offset in range(max_offset):
        input_ids_offset = input_ids[:, :input_ids.shape[1]-offset]
        text = tokenizer.decode(input_ids_offset[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        text = text.replace('<|begin_of_text|>', '')
        text = text.replace('<|end_of_text|>', '')
        input_ids_reproduced = tokenizer([text], return_tensors='pt').input_ids
        if input_ids_reproduced.shape[1] == input_ids_offset.shape[1] + 1:
            assert input_ids_reproduced[0, 0] == tokenizer.bos_token_id
            input_ids_reproduced = input_ids_reproduced[:, 1:]
        elif input_ids_reproduced.shape[1] < input_ids_offset.shape[1]:
            continue
        assert input_ids_reproduced.shape == input_ids_offset.shape
        if torch.all(input_ids_reproduced.eq(input_ids_offset)):
            return text, offset
    #raise ValueError("...")
    return None, None


def get_dataset_max_seq_length(dataset_name):
    return {
        "yelp": (32, 128), # 5% ~ 95%: 30 ~ 200
        "agnews": (8, 32), # 5% ~ 95%: 31 ~ 79
        "dailydialog": (256, 36), # 5% ~ 95%: 21 ~ 237 # output改为50
        "plato_dailydialog": (256, 50),
    }[dataset_name]


def get_dataset_indexed_labels(dataset_name):
    return {
        "yelp": {0: "rated_0", 1: "rated_1", 2: "rated_2", 3: "rated_3", 4: "rated_4"},
        "agnews": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "dailydialog": {0: "inform", 1: "question", 2: "directive", 3: "commissive"},
        "plato_dailydialog": {0: "unk"},
    }[dataset_name]


def polish(string):
    return PreTrainedTokenizerBase.clean_up_tokenization(string).strip()


eot_token = "<|eot_id|>"
def read_data(dataset_name, usage):
    download_config = DownloadConfig()
    download_config.local_files_only = True
    if dataset_name == "yelp":
        # train / validation / test
        # 100k / 10k / 10k
        # downloaded from https://github.com/OpenVLG/DELLA/tree/main/data
        dataset = load_dataset("./yelp", download_config=download_config)
        dataset = dataset[usage]
        dataset = [
            {
                "text": polish(data["text"].split("\t")[1]),
                "label": int(data["text"].split("\t")[0]), # 0~4
            }
            for data in dataset
        ]
    elif dataset_name == "agnews":
        # train / validation / test
        # 120k / 3.8k / 3.8k
        # dataset = load_dataset("ag_news") # from huggingface
        dataset = load_dataset("./ag_news", download_config=download_config) # from local
        #
        if usage == "train":
            dataset = dataset["train"] # num_rows: 120000
        else:
            dataset = dataset["test"] # num_rows: 7600
            if usage == "validation":
                dataset = [data for i,data in enumerate(dataset) if i%2==0]
            elif usage == "test":
                dataset = [data for i,data in enumerate(dataset) if i%2==1]
            else:
                raise NotImplementedError(usage)
        dataset = [
            {
                "text": polish(data["text"]),
                "label": data["label"]
            }
            for data in dataset
        ]
    elif dataset_name in ["dailydialog", "dailydialog_emotion", "dailydialog_intent"]:
        # train / validation / test
        # 87k / 8k / 7.7k
        dataset = load_dataset("./dailydialog", download_config=download_config, trust_remote_code=True)
        dataset = dataset[usage]
        dataset_ = []
        for item in dataset:
            context = ""
            for utterance, emotion, intent in zip(
                item["utterances"], item["emotions"], item["acts"]
            ):
                if len(context) > 0:
                    dataset_.append({
                        "text": context + polish(utterance) + eot_token,
                        "prompt": context,
                        "response": polish(utterance) + eot_token,
                        "label": (emotion if "emotion" in dataset_name else intent - 1),
                    })
                context = context + polish(utterance) + eot_token
        dataset = dataset_
    elif dataset_name =="plato_dailydialog":
        dataset = []
        file = os.path.join(workspace, f"datasets/PLATO-Dataset/DailyDialog/dial.{usage[:5]}")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                prompt, response = line.strip().split("\t")
                prompt = prompt.replace(" __eou__ ", eot_token).strip() + eot_token
                response = response.strip() + eot_token
                data = {
                    "text": prompt + response,
                    "prompt": prompt,
                    "response": response,
                    "label": 0,
                }
                if False and usage != "test" and len(dataset) > 0 and dataset[-1]["text"] == prompt:
                    data = {
                        "text": prompt + response,
                        "prompt": dataset[-1]["prompt"],
                        "response": dataset[-1]["response"] + response,
                        "label": 0
                    }
                    dataset[-1] = data
                else:
                    dataset.append(data)
    elif dataset_name == "infinity":
        dataset = load_from_disk(os.path.join(workspace, f"datasets/Infinity-Instruct"))
        dataset = dataset["train"]
        label2count = defaultdict(int)
        for data in dataset:
            for label in data['label']['cate_ability_zh']:
                label2count[label] += 1
        dataset = sorted(list(dataset), key=lambda data:" ".join(
            sorted(data["label"]['cate_ability_zh'])
        ))

    else:
        raise NotImplementedError(dataset_name)

    return dataset


class BaseDataset:
    def __init__(self, tokenizer_path="gpt2", dataset_name="dailydialog", usage="train"):
        self.tokenizer_path = tokenizer_path
        self.dataset_name = dataset_name
        self.usage = usage
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token_id is None:
            config = AutoConfig.from_pretrained(tokenizer_path)
            if config.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = config.pad_token_id
        self.data = read_data(dataset_name, usage)
        self.max_input_length, self.max_output_length = get_dataset_max_seq_length(dataset_name)
        self.preprocess()

    def preprocess(self):
        preprocessed_data = []
        if "prompt" not in self.data[0] or "response" not in self.data[0]:
            assert "text" in self.data[0]
            self.tokenizer.truncation_side = "right"
            for item in self.data:
                text_ids = self.tokenizer([item["text"]], truncation=True, return_tensors="pt",
                                          max_length=self.max_input_length+self.max_output_length).input_ids
                if not text_ids.shape[1] >= self.max_input_length + self.max_output_length // 10:
                    continue
                prompt_ids = text_ids[:, :self.max_input_length]
                response_ids = text_ids[:, self.max_input_length:]
                prompt, response = self.detokenize(prompt_ids=prompt_ids, response_ids=response_ids)
                if prompt is None or response is None:
                    continue
                preprocessed_data.append({
                    "text": item["text"],
                    "prompt": prompt,
                    "response": response,
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "label": item["label"],
                })
        elif self.dataset_name == "plato_dailydialog":
            for item in self.data:
                prompt = item["prompt"]
                response = item["response"]
                prompt_ids = self.tokenizer.encode(prompt)
                response_ids = self.tokenizer.encode(response)[1:] # ignore '<|begin_of_text|>'
                data = {
                    "text": prompt+response,
                    "prompt": prompt,
                    "response": response,
                    "prompt_ids": torch.LongTensor([prompt_ids]),
                    "response_ids": torch.LongTensor([response_ids]),
                    "label": 0,
                }
                preprocessed_data.append(data)
        else:
            for item in self.data:
                prompt_ids, response_ids = self.tokenize(prompt=item["prompt"], response=item["response"])
                prompt_, response_ = self.detokenize(prompt_ids=prompt_ids, response_ids=response_ids)
                if prompt_ is None or response_ is None:
                    continue
                assert item["prompt"].endswith(prompt_)
                assert item["response"].startswith(response_)
                preprocessed_data.append({
                    "text": item["text"],
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "label": item["label"],
                })
        self.preprocessed_data = preprocessed_data

    def tokenize(self, prompt, response, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        input_ids = tokenizer([prompt + response], truncation=True, return_tensors="pt",
                               max_length=1024).input_ids
        prompt_ids = tokenizer([prompt], truncation=True, return_tensors="pt",
                               max_length=1024).input_ids
        if prompt_ids[0, -1] == tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_ids_length = prompt_ids.shape[1]
        assert torch.all(prompt_ids.eq(input_ids[:, :prompt_ids_length]))
        response_ids = input_ids[:, prompt_ids_length:]
        prompt_ids = input_ids[:, :prompt_ids_length]
        prompt_ids = prompt_ids[:, -self.max_input_length:]
        response_ids = response_ids[:, :self.max_output_length]
        return prompt_ids, response_ids

    def detokenize(self, prompt_ids, response_ids, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        prompt, offset = longest_reproducible_prefix_decoding(input_ids=prompt_ids, tokenizer=tokenizer)
        text, _ = longest_reproducible_prefix_decoding(input_ids=input_ids, tokenizer=tokenizer)
        if prompt is None or text is None:
            return None, None
        assert text.startswith(prompt)
        response = text[len(prompt):]
        return prompt, response

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, index):
        return self.preprocessed_data[index]

    def __repr__(self):
        return f"{self.__class__.__name__} - {self.dataset_name} - {self.usage}: {len(self.preprocessed_data)} / {len(self.data)}"

    def pad_and_concat(self, list_ids, truncation_side, padding_side, pad_token_id, max_length):
        if not truncation_side in ["right", "left"]:
            raise NotImplementedError(truncation_side)
        if not padding_side in ["right", "left"]:
            raise NotImplementedError(truncation_side)
        if not all(len(ids.shape)==2 for ids in list_ids):
            raise ValueError([ids.shape for ids in list_ids])
        list_input_ids, list_attention_mask = [], []
        if max_length == "longest":
            max_length = max(ids.shape[1] for ids in list_ids)
        for ids in list_ids:
            if ids.shape[1] >= max_length:
                if truncation_side == "left":
                    ids = ids[:, -max_length:]
                else:
                    ids = ids[:, :max_length]
                attention_mask = torch.ones_like(ids)
            else:
                padding_ids = torch.LongTensor(size=(ids.shape[0], max_length-ids.shape[1])).fill_(pad_token_id)
                if padding_side == "left":
                    attention_mask = torch.cat([padding_ids.clone().fill_(0), ids.clone().fill_(1)], dim=1)
                    ids = torch.cat([padding_ids, ids], dim=1)
                else:
                    attention_mask = torch.cat([ids.clone().fill_(1), padding_ids.clone().fill_(0)], dim=1)
                    ids = torch.cat([ids, padding_ids], dim=1)
            list_input_ids.append(ids)
            list_attention_mask.append(attention_mask)
        return {"input_ids": torch.concat(list_input_ids, dim=0).long(),
                "attention_mask": torch.concat(list_attention_mask, dim=0).long()}


class SFTDataset(BaseDataset):
    def __init__(self, tokenizer_path="gpt2", dataset_name="dailydialog", usage="train"):
        super().__init__(tokenizer_path=tokenizer_path, dataset_name=dataset_name, usage=usage)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # keys: {"text", "prompt", "response", "prompt_ids", "response_ids", "label"}
        prompt_ids = data["prompt_ids"].long()
        response_ids = data["response_ids"].long()
        assert prompt_ids.shape[1] > 0
        assert response_ids.shape[1] > 0
        response_ids = torch.cat([response_ids, torch.LongTensor([[self.tokenizer.eos_token_id]])], dim=1)
        return {"prompt_ids": prompt_ids, "response_ids": response_ids}

    def collate_fn(self, batch_items):
        batch_size = len(batch_items)
        prompt_inputs = self.pad_and_concat(
            list_ids=[item["prompt_ids"] for item in batch_items],
            truncation_side="left",
            padding_side="left",
            pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest",
        )
        response_inputs = self.pad_and_concat(
            list_ids=[item["response_ids"] for item in batch_items],
            truncation_side="right",
            padding_side="right",
            pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest",
        )
        input_ids = torch.cat([prompt_inputs["input_ids"], response_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_inputs["attention_mask"], response_inputs["attention_mask"]], dim=1)
        labels = torch.cat([
            prompt_inputs["input_ids"].clone().fill_(-100),
            torch.where(response_inputs["attention_mask"].eq(1), response_inputs["input_ids"], -100)
        ], dim=1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ClassificationDataset(BaseDataset):
    def __init__(self, tokenizer_path="gpt2", dataset_name="dailydialog", usage="train",
                 classification_tokenizer_path="roberta-large"):
        super().__init__(tokenizer_path=tokenizer_path, dataset_name=dataset_name, usage=usage)
        labels = [data["label"] for data in self.preprocessed_data]
        labels_uniq = sorted(set(labels))
        self.label2count = {label: labels.count(label) for label in labels_uniq}
        assert len(labels_uniq) == max(labels_uniq) + 1, f"unexpected labels set: {self.label2count}"
        self.num_labels = len(labels_uniq)
        self.classification_tokenizer = AutoTokenizer.from_pretrained(classification_tokenizer_path)
        self.classification_tokenizer.truncation_side = "right"
        self.classification_tokenizer.padding_side = "right"

        self.label2indexes = {label: [] for label in range(self.num_labels)}
        for index, data in enumerate(self.preprocessed_data):
            self.label2indexes[data["label"]].append(index)

    def __getitem__(self, index=None, data=None):
        if index is not None:
            data = super().__getitem__(index)
        else:
            assert data is not None
        # keys: {"prompt", "response"} + (optional) {"text", "prompt_ids", "response_ids", "label"}
        max_length = self.max_input_length+self.max_output_length
        inputs = self.classification_tokenizer(
            [data["prompt"]+data["response"]],
            truncation=True, padding="max_length",
            return_tensors="pt",
            max_length=max_length
        )
        prompt_length = self.classification_tokenizer([data["prompt"]], return_tensors='pt').input_ids.shape[1]
        inputs = {
            "input_ids": inputs.input_ids,
            "token_type_ids": (torch.arange(max_length) < prompt_length).long().unsqueeze(0),
            "attention_mask": inputs.attention_mask,
        }
        if "label" in data:
            inputs["labels"] = torch.LongTensor([data["label"]])
        return inputs

    def collate_fn(self, batch_items):
        keys = batch_items[0].keys()
        return {
            key: torch.cat([batch_item[key] for batch_item in batch_items], dim=0)
            for key in keys
        }


class One2ManyDataset(BaseDataset):
    def __init__(self, tokenizer_path="gpt2", dataset_name="dailydialog", usage="train",
                 total_many=32, mini_many=4, only_load_prompts=False, inner_shuffle=False):
        super().__init__(tokenizer_path=tokenizer_path, dataset_name=dataset_name, usage=usage)
        self.total_many = total_many
        self.mini_many = mini_many
        if not only_load_prompts:
            self.load_one2many_inference()
            self.inner_iter_steps = total_many // mini_many
            self.iter_queue = []
            self.inner_shuffle = inner_shuffle

    def cache_file(self, index_split=1, num_splits=10):
        return os.path.join(root, f"one2many_inference/{self.tokenizer_path}-{self.dataset_name}/{self.usage}-{index_split}-of-{num_splits}.json")

    def load_one2many_inference(self, num_splits=10):
        self.one2many_inference = []
        for index_split in range(1, num_splits+1):
            with open(self.cache_file(index_split=index_split, num_splits=num_splits), "r", encoding='utf-8') as f:
                self.one2many_inference += json.loads(f.read())
        assert len(self.one2many_inference) == len(self.preprocessed_data)
        for index, (pd, oi) in enumerate(zip(self.preprocessed_data, self.one2many_inference)):
            assert pd["prompt"] == oi["prompt"], f"prompt mismatch at index: {index}"

    def __len__(self):
        return len(self.one2many_inference)

    def __getitem__(self, item):
        for inner_bias in range(self.inner_iter_steps):
            self.iter_queue.append((item, inner_bias))

        item, inner_bias = self.iter_queue.pop(0)
        if inner_bias == 0 and self.inner_shuffle:
            # inner shuffle, to avoid contrastive overfitting across multiple epochs
            self.one2many_inference[item]["responses"] = list(
                np.random.choice(self.one2many_inference[item]["responses"],
                                 self.total_many, replace=False)
            )
        prompt = self.one2many_inference[item]["prompt"]
        responses = [self.one2many_inference[item]["responses"][inner_item]
                     for inner_item in range(inner_bias, self.total_many, self.inner_iter_steps)]
        #responses = self.one2many_inference[item]["responses"]
        #responses = responses[inner_bias*self.mini_many:] + responses[:inner_bias*self.mini_many]
        prompt_ids = self.tokenizer(
            [prompt], return_tensors="pt"
        ).input_ids.long()
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "right"
        input_ids = self.tokenizer(
            [prompt+response for response in responses],
            truncation=True, return_tensors="pt", padding="longest",
            max_length=prompt_ids.shape[1]+self.max_output_length
        ).input_ids.long()
        exceeded_prompt_length = prompt_ids.shape[1] - self.max_input_length
        if exceeded_prompt_length > 0:
            input_ids = input_ids[:, exceeded_prompt_length:]
            prompt_ids = prompt_ids[:, exceeded_prompt_length:]
        responses_ids = self.tokenizer(
            responses,
            truncation=True, return_tensors="pt", padding="longest",
            max_length=self.max_output_length
        ).input_ids.long()
        if responses_ids.shape[1] < self.max_output_length:
            eos_token_ids = torch.LongTensor([[self.tokenizer.eos_token_id]] * input_ids.shape[0])
            input_ids = torch.cat([input_ids, eos_token_ids], dim=1)
        return {"prompt_ids": prompt_ids, "input_ids": input_ids}

    def collate_fn(self, batch_items):
        # for latent_encoder
        #     posterior_input_ids: torch.LongTensor = None,
        #     posterior_attention_mask: torch.FloatTensor = None,
        # for latent_decoder
        #     input_ids: torch.LongTensor = None,
        #     attention_mask: torch.FloatTensor = None,
        #     labels: torch.LongTensor = None,
        batch_size = len(batch_items)
        assert batch_size == 1, (f"batch_size ({batch_size}) > 1 is currently"
                                 f" not supported in one2many CVAE training.")
        prompt_ids, input_ids = batch_items[0]["prompt_ids"], batch_items[0]["input_ids"]
        #posterior_input_ids = input_ids[:, prompt_ids.shape[1]:]
        posterior_input_ids = input_ids
        posterior_input_ids, posterior_attention_mask = right_padding_to_left_padding(
            posterior_input_ids,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        group_posterior_input_ids, group_posterior_attention_mask = posterior_input_ids.clone(), posterior_attention_mask.clone()
        posterior_input_ids = posterior_input_ids[:self.mini_many, :]
        posterior_attention_mask = posterior_attention_mask[:self.mini_many, :]
        input_ids = input_ids[:self.mini_many, :]

        assert not torch.any(prompt_ids.eq(self.tokenizer.pad_token_id))
        attention_mask = create_attention_mask_after_first_eos(input_ids, self.tokenizer.eos_token_id)
        labels = torch.where(attention_mask.eq(1), input_ids, -100)
        labels[:, :prompt_ids.shape[1]] = -100

        return {
            "posterior_input_ids": posterior_input_ids,
            "posterior_attention_mask": posterior_attention_mask,
            "group_posterior_input_ids": group_posterior_input_ids,
            "group_posterior_attention_mask": group_posterior_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ComparisonDataset(BaseDataset):
    def __init__(self, tokenizer_path="gpt2", dataset_name="dailydialog", usage="train",
                 pairs_per_prompt=2, preference_label_index=0):
        super().__init__(tokenizer_path=tokenizer_path, dataset_name=dataset_name, usage=usage)
        self.pairs_per_prompt = pairs_per_prompt
        self.preference_label_index = preference_label_index
        self.preference_label = get_dataset_indexed_labels(dataset_name)[preference_label_index]
        self.load_one2many_inference_with_logits()

    def __repr__(self):
        return f"{self.__class__.__name__} - {self.dataset_name} - {self.preference_label} - {self.usage}: {len(self.preprocessed_data)} / {len(self.data)}"

    def cache_file(self, index_split=1, num_splits=10):
        return os.path.join(root, f"one2many_inference/{self.tokenizer_path}-{self.dataset_name}/{self.usage}-{index_split}-of-{num_splits}-with-logits.json")

    def load_one2many_inference_with_logits(self, num_splits=10):
        self.one2many_inference_with_logits = []
        for index_split in range(1, num_splits+1):
            with open(self.cache_file(index_split=index_split, num_splits=num_splits), "r", encoding='utf-8') as f:
                self.one2many_inference_with_logits += json.loads(f.read())
        assert len(self.one2many_inference_with_logits) == len(self.preprocessed_data)
        for index, (pd, oi) in enumerate(zip(self.preprocessed_data, self.one2many_inference_with_logits)):
            assert pd["prompt"] == oi["prompt"], f"prompt mismatch at index: {index}"

    def compute_index_label_mutual_information(self):
        self.index_label_mi = []
        for index in range(len(self.one2many_inference_with_logits)):
            data = self.one2many_inference_with_logits[index]
            logits = torch.FloatTensor(data["logits"])
            p_xy = torch.nn.functional.softmax(logits, dim=1) / logits.shape[0]
            p_y = p_xy.sum(dim=0)
            h_x = np.log(logits.shape[0])
            h_y = (-p_y * (p_y + 1e-5).log()).sum()
            h_xy = (-p_xy * (p_xy + 1e-5).log()).sum()
            mi_xy = h_x + h_y - h_xy
            self.index_label_mi.append(mi_xy.item())

    def __len__(self):
        return len(self.one2many_inference_with_logits) * self.pairs_per_prompt

    def __getitem__(self, item):
        index, bias = item // self.pairs_per_prompt, item % self.pairs_per_prompt
        data = self.one2many_inference_with_logits[index]
        prompt = data["prompt"]
        responses = data["responses"]
        logits = torch.FloatTensor(data["logits"])[:, self.preference_label_index]

        best_of_n_values, best_of_n_indices = torch.topk(logits, self.pairs_per_prompt)
        worst_of_n_values, worst_of_n_indices = torch.topk(-logits, self.pairs_per_prompt)
        chosen_index = best_of_n_indices[bias].item()
        rejected_index = worst_of_n_indices[bias].item()
        chosen_response = responses[chosen_index]
        rejected_response = responses[rejected_index]

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        prompt_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        chosen_ids = self.tokenizer(
            [prompt+chosen_response], return_tensors="pt", truncation=True,
            max_length=self.max_input_length+self.max_output_length
        ).input_ids.long()
        if chosen_ids.shape[1] < self.max_input_length+self.max_output_length:
            eos_token_ids = torch.LongTensor([[self.tokenizer.eos_token_id]])
            chosen_ids = torch.cat([chosen_ids, eos_token_ids], dim=1)
        rejected_ids = self.tokenizer(
            [prompt+rejected_response], return_tensors="pt", truncation=True,
            max_length=self.max_input_length+self.max_output_length
        ).input_ids.long()
        if rejected_ids.shape[1] < self.max_input_length+self.max_output_length:
            eos_token_ids = torch.LongTensor([[self.tokenizer.eos_token_id]])
            rejected_ids = torch.cat([rejected_ids, eos_token_ids], dim=1)
        return {
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids, "rejected_ids": rejected_ids,
            "chosen_logits": best_of_n_values, "rejected_logits": worst_of_n_values
        }

    def dpo_collate_fn(self, batch_items):
        chosen_ids, _ = self.pad_and_concat(
            list_ids=[batch_item["chosen_ids"] for batch_item in batch_items],
            truncation_side="right", padding_side="right", pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest"
        ).values()
        chosen_attention_mask = create_attention_mask_after_first_eos(chosen_ids, self.tokenizer.eos_token_id)
        rejected_ids, _ = self.pad_and_concat(
            list_ids=[batch_item["rejected_ids"] for batch_item in batch_items],
            truncation_side="right", padding_side="right", pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest"
        ).values()
        rejected_attention_mask = create_attention_mask_after_first_eos(rejected_ids, self.tokenizer.eos_token_id)
        return {
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

    def lpo_collate_fn(self, batch_items):
        input_ids, attention_mask = self.pad_and_concat(
            list_ids=[batch_item["prompt_ids"] for batch_item in batch_items],
            truncation_side="left", padding_side="left", pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest",
        ).values()
        chosen_ids, _ = self.pad_and_concat(
            list_ids=[batch_item["chosen_ids"] for batch_item in batch_items],
            truncation_side="right", padding_side="right", pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest",
        ).values()
        chosen_ids, chosen_attention_mask = right_padding_to_left_padding(
            chosen_ids,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        rejected_ids, _ = self.pad_and_concat(
            list_ids=[batch_item["rejected_ids"] for batch_item in batch_items],
            truncation_side="right", padding_side="right", pad_token_id=self.tokenizer.pad_token_id,
            max_length="longest",
        ).values()
        rejected_ids, rejected_attention_mask = right_padding_to_left_padding(
            rejected_ids,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }


def convert_style(hyp_text):
    import re
    # 替换标点符号后的空格
    gts_text = re.sub(r"(\S)([,.!?'])", r'\1 \2', hyp_text)
    gts_text = re.sub(r"([,.!?'])(\S)", r'\1 \2', gts_text)
    gts_text = gts_text.replace("  ", " ")
    # 替换省略号
    gts_text = gts_text.replace(". . .", "...")
    # 将缩写中间的空格去掉
    gts_text = gts_text.replace(" ’ ", " ' ")
    return gts_text.strip()

if __name__ == "__main__":
    for usage in ["train", "validation", "test"]:
        plato_dailydialog = BaseDataset(tokenizer_path="llama3-8b", dataset_name="plato_dailydialog", usage=usage)
        print(plato_dailydialog)
        plato_dailydialog = SFTDataset(tokenizer_path="llama3-8b", dataset_name="plato_dailydialog", usage=usage)
        print(plato_dailydialog)
        for i in tqdm(range(len(plato_dailydialog))):
            prompt = plato_dailydialog.data[i]["prompt"]
            prompt_reconstructed = plato_dailydialog.tokenizer.batch_decode(plato_dailydialog[i]["prompt_ids"], clean_up_tokenization_spaces=False)[0]
            assert prompt_reconstructed.startswith("<|begin_of_text|>")
            prompt_reconstructed = prompt_reconstructed[len("<|begin_of_text|>"):]
            assert prompt_reconstructed == prompt

            response = plato_dailydialog.data[i]["response"]
            response_reconstructed = plato_dailydialog.tokenizer.batch_decode(plato_dailydialog[i]["response_ids"], clean_up_tokenization_spaces=False)[0]
            assert response_reconstructed.endswith("<|end_of_text|>")
            response_reconstructed = response_reconstructed[:-len("<|end_of_text|>")]
            assert response_reconstructed == response