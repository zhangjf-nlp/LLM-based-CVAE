import os

workspace = "/Users/zhangjianfei/projects"
root = os.path.join(workspace, "CLaP")

base_model_names = ["gpt2", "gpt2-large", "gpt2-xl", "llama3-8b"]#, "llama2-7b", "llama2-13b"]
dataset_names = ["imdb", "yelp", "dailydialog", "agnews", "alpaca"]


def create_links():
    # create links to support loading local models and datasets
    for model_name in [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "llama3-8b",
        "roberta-large",
        "microsoft/deberta-xlarge-mnli",
        "microsoft/DialoGPT-large",
    ]:
        if len(model_name.split("/")) > 1:
            os.makedirs("/".join(model_name.split("/")[:-1]), exist_ok=True)
        if not os.path.exists(model_name):
            os.system(f"ln -s {workspace}/models/{model_name} {model_name}")
    for dataset_name in ["imdb", "yelp", "dailydialog", "ag_news", "alpaca"]:
        if not os.path.exists(dataset_name):
            os.system(f"ln -s {workspace}/datasets/{dataset_name} {dataset_name}")
    if not os.path.exists("PLATO-Dataset"):
        os.system(f"ln -s {workspace}/datasets/PLATO-Dataset PLATO-Dataset")

if __name__ == "__main__":
    create_links()