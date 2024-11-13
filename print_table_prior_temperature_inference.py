import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据集和模型
datasets = ["yelp", "agnews", "dailydialog"]
models = [
    "Vanilla-0-0", "Vanilla-1-0",
    "Beta05-VAE-0-0", "Beta02-VAE-0-0", "Beta-VAE-0-0", "AE-0-0",
    "DG-VAE-0-0", "DG-VAE-1-0", "DG-VAE-0-1",
    #"Vanilla-2-0", "Vanilla-1-0", "Vanilla-0-1",
    "SFT",
]
translation_model = {
    "Vanilla-0-0": r"VAE ($\beta=1.0$)",
    "Beta05-VAE-0-0": r"VAE ($\beta=0.5$)",
    "Beta02-VAE-0-0": r"VAE ($\beta=0.2$)",
    "Beta-VAE-0-0": r"VAE ($\beta=0.1$)",
    "AE-0-0": r"VAE ($\beta=0.0$)",
    "Vanilla-1-0": r"VAE w. Skip",
    "DG-VAE-0-0": "DG-VAE",
    "DG-VAE-1-0": "DG-VAE w. Skip",
    "DG-VAE-0-1": "DG-VAE w. G-Skip (ours)",
    "SFT": "Llama3-8B"
}
translation_dataset = {
    "dailydialog": "DailyDialog",
    "agnews": "AGNews",
    "yelp": "Yelp",
}
indexed_metric = {
    "B": "mauve",
    "C": "dinstinct",
    #"D": "bleu",
    "E": "rouge",
}

# 读取数据
file_path = '/Users/zhangjianfei/Downloads/test_prior_inference_with_different_temperature_results_v6.0 (3).csv'
df = pd.read_csv(file_path)
print(df)

# 创建图形和子图
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(len(translation_dataset), len(indexed_metric),
                        figsize=(7*len(indexed_metric), 5*len(translation_dataset)))
fig.subplots_adjust(right=0.9)

# 存放每个子图的图例
lines = []
labels = []

for i, dataset in enumerate(translation_dataset.keys()):
    for j, (index, metric) in enumerate(indexed_metric.items()):
        for model in translation_model:
            model_name = f"{dataset}-{model}-small_test"
            metric_index = ord(index) - ord("A")

            # 横坐标
            x = [0.1, 0.4, 0.7, 1.0]
            # 纵坐标
            try:
                model_row = df[df.iloc[:, 0] == model_name]
                y = eval(model_row.iloc[0, metric_index])
            except:
                if j == 0:
                    print(f"unfound: {dataset} - {model}")
                y = [0] * len(x)  # unfound数据填充为[0] * len(x)
            # 绘制线条
            if len(indexed_metric) > 1:
                line, = axs[i, j].plot(x, y, label=translation_model[model])
            else:
                line, = axs[i].plot(x, y, label=translation_model[model])
            if i == 0 and j == 0:
                lines.append(line)
                labels.append(translation_model[model])

        if len(indexed_metric) > 1:
            axs[i, j].set_title(f"{translation_dataset[dataset]} - {metric}")
            axs[i, j].set_xlabel("temperature")
            axs[i, j].set_xticks([0.1, 0.4, 0.7, 1.0])  # 设置 x 轴刻度
            axs[i, j].set_ylim(0, 1)  # 设置 y 轴范围
        else:
            axs[i].set_title(f"{translation_dataset[dataset]} - {metric}")
            axs[i].set_xlabel("temperature")
            axs[i].set_xticks([0.1, 0.4, 0.7, 1.0])  # 设置 x 轴刻度
            axs[i].set_ylim(0, 1)  # 设置 y 轴范围


# 添加公共图例
fig.legend(lines, labels, loc='center right', fontsize='small')

# 保存图形
plt.savefig('results_prior_temperature.png')
plt.close()
