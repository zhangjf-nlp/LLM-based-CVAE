import os
import gc
import json
import torch
from tqdm import tqdm
from config import root
from dataset import One2ManyDataset, ClassificationDataset
from utils import get_basic_parser, get_last_checkpoint
from transformers import RobertaForSequenceClassification, AutoTokenizer

parser = get_basic_parser(base_model_name="roberta-large")
parser.add_argument("--num_splits", type=int, default=10)
parser.add_argument("--index_split", type=int, default=1)
args = parser.parse_args()
tokenizer_path = "llama3-8b" # tokenizer for LLMs training
device = "cuda:0"
classifier_output_dir = os.path.join(root, f"cls/{args.base_model_name}-{args.dataset_name}")
print(f"classifier_output_dir: {classifier_output_dir}")
classifier_output_dir = get_last_checkpoint(classifier_output_dir)
classifier = RobertaForSequenceClassification.from_pretrained(classifier_output_dir).to(device).eval()
classification_tokenizer = AutoTokenizer.from_pretrained(classifier_output_dir)
classification_dataset_for_api = ClassificationDataset(
    tokenizer_path=tokenizer_path, dataset_name=args.dataset_name, usage="test",
    classification_tokenizer_path=classifier_output_dir,
)

@torch.no_grad()
def process(prompt, responses):
    batch_items = []
    for response in responses:
        batch_item = classification_dataset_for_api.__getitem__(
            index=None, data={"prompt": prompt, "response": response}
        )
        batch_items.append(batch_item)
    inputs = classification_dataset_for_api.collate_fn(batch_items=batch_items)
    inputs = {k: v.to(device) if v is not None else v for k, v in inputs.items()}
    logits = classifier(**inputs, return_dict=True).logits
    return logits.cpu().tolist()


usages = ["validation", "train", "test"]
for usage in usages:
    dataset = One2ManyDataset(
        tokenizer_path=tokenizer_path,
        dataset_name=args.dataset_name,
        usage=usage, only_load_prompts=True,
    )
    output_file = dataset.cache_file(index_split=args.index_split, num_splits=args.num_splits)
    with open(output_file, "r", encoding='utf-8') as f:
        one2many_inference = json.loads(f.read())
    results = []
    for item in tqdm(range(len(one2many_inference))):
        verbose = (item == 0)
        prompt = one2many_inference[item]["prompt"]
        responses = one2many_inference[item]["responses"]
        if verbose:
            print(f"prompt:\n{prompt}")
            print(f"responses:")
            print(f"\n".join(responses))

        logits = process(prompt=prompt, responses=responses)
        if verbose:
            print(f"logits: {logits}")

        results.append({
            "prompt": prompt,
            "responses": responses,
            "logits": logits
        })

    with open(output_file.replace(".json", "-with-logits.json"), "w", encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))