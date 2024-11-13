import json
import math

import numpy as np
import pandas as pd


method_mapping = {
    "AE-0-0-small_test": "+ $\\beta$-VAE ($\\beta=0.0$)",
    "Beta-VAE-0-0-small_test": "+ $\\beta$-VAE ($\\beta=0.1$)",
    "Beta02-VAE-0-0-small_test": "+ $\\beta$-VAE ($\\beta=0.2$)",
    "Beta05-VAE-0-0-small_test": "+ $\\beta$-VAE ($\\beta=0.5$)",
    "BOW-VAE-0-0-small_test": "+ BoW-VAE",
    "DG-VAE-0-0-small_test": "DG-VAE w/ Memory",
    "DG-VAE-0-1-small_test": "DG-VAE w/ Memory + G-Skip (ours)",
    "DG-VAE-1-0-small_test": "DG-VAE w/ Memory + Skip",
    "EMB-VAE-0-0-small_test": "VAE w/ Memory + Embedding",
    "FB-VAE-0-0-small_test": "+ FB-VAE ($\\lambda=0.5$)",
    "Vanilla-0-0-small_test": "VAE w/ Memory",
    "Vanilla-0-1-small_test": "VAE w/ Memory + G-Skip",
    "Vanilla-1-0-small_test": "VAE w/ Memory + Skip",
    "Anneal_-0-0-small_test": "+ Cyclic-VAE",
    "Optimus_-0-0-small_test": "+ Cyclic-VAE + FB-VAE ($\\lambda=0.5$) $^\\dagger$",
    "SFT-small_test": "Llama3-8B-small_test",
}

def extract(metrics):
    # 读取 CSV 文件
    df = pd.read_csv('/Users/zhangjianfei/Downloads/test_model_single_plus_greedy_results_v7.0 (5).csv')  # 替换为你的文件路径

    result = {}
    for index, row in df.iterrows():
        model_name = row['model_name']
        dataset = model_name.split('-')[0]
        method = '-'.join(model_name.split('-')[1:])  # 提取 dataset 和 method
        if not method in method_mapping:
            continue

        method = method_mapping[method]

        if method not in result:
            result[method] = {}

        # 将指标值添加到对应的 dataset
        result[method][dataset] = {metric: row[metric] for metric in metrics}

    return result


#results = extract(["post-mauve_score","post-bleu","post-selfbleu","post-distinct","post-rouge"])
results = extract(["prior-mauve_score","prior-rouge","prior-distinct","prior-selfbleu"])
print(json.dumps(results, ensure_ascii=False, indent=4))
for model_name in [
    r"VAE w/ Memory",
    r"+ Cyclic-VAE",
    r"+ FB-VAE ($\lambda=0.5$)",
    r"+ Cyclic-VAE + FB-VAE ($\lambda=0.5$) $^\dagger$",
    r"+ BoW-VAE",
    r"+ $\beta$-VAE ($\beta=0.5$)",
    r"+ $\beta$-VAE ($\beta=0.2$)",
    r"+ $\beta$-VAE ($\beta=0.1$)",
    r"+ $\beta$-VAE ($\beta=0.0$)",
    r"VAE w/ Memory + Embedding",
    r"VAE w/ Memory + Skip",
    r"VAE w/ Memory + G-Skip",
    r"DG-VAE w/ Memory",
    r"DG-VAE w/ Memory + Skip",
    r"DG-VAE w/ Memory + G-Skip (ours)",
    r"Llama3-8B",
]:
    if model_name not in results:
        print(f"{model_name} \\\\")
    else:
        dataset2values = results[model_name]
        output = []
        output.append(model_name)
        for dataset in ["agnews", "yelp", "dailydialog"]:
            if dataset in dataset2values:
                values = dataset2values[dataset].values()
                values = [str(_).replace("'", '"') for _ in values]
                try:
                    output.append(f"{float(values[0])*100:.2f}") # "post-mauve_score"
                except:
                    output.append("nan")
                try:
                    output.append(f"{float(json.loads(values[1])['rougeLsum'])*100:.2f}") # "post-rouge"
                except:
                    output.append("nan")
                try:
                    output.append(f"{float(json.loads(values[2])['distinct-4'])*100:.2f}") # "post-distinct"
                except:
                    output.append("nan")
                try:
                    output.append(f"{float(list(json.loads(values[3]).values())[-1])*100:.2f}") # "post-selfbleu"
                except:
                    output.append("nan")
            else:
                output += ["unk"] * 4
        print(" & ".join(output) + "\\\\")


