import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据集和模型
translation_model = {
    "Vanilla-0-0": "VAE w/ Memory",
    "Beta05-VAE-0-0": "+ $\\beta$-VAE ($\\beta=0.5$)",
    "Beta02-VAE-0-0": "+ $\\beta$-VAE ($\\beta=0.2$)",
    "Beta-VAE-0-0": "+ $\\beta$-VAE ($\\beta=0.1$)",
    "AE-0-0": "+ $\\beta$-VAE ($\\beta=0.0$)",
    "Vanilla-1-0": "VAE w/ Memory + Skip",
    "DG-VAE-0-0": "DG-VAE w/ Memory",
    "DG-VAE-1-0": "DG-VAE w/ Memory + Skip",
    "DG-VAE-0-1": "DG-VAE w/ Memory + G-Skip (ours)",
}
translation_model = {
    "Beta05-VAE-0-0": r"$\beta$-CVAE ($\beta=0.5$)",
    "Beta02-VAE-0-0": r"$\beta$-CVAE ($\beta=0.2$)",
    "Beta-VAE-0-0": r"$\beta$-CVAE ($\beta=0.1$)",
    "AE-0-0": r"$\beta$-CVAE ($\beta=0.0$)",
    "FB-VAE-0-0": r"FB-CVAE ($\lambda=0.5$)",
    #"Optimus_-0-0": r"Cyclic-FB-CVAE ($\lambda=0.5$)",
    "Vanilla-1-0": "Baseline CVAE w/ Skip",
    "DG-VAE-0-0": "DG-CVAE (ablation)",
    "DG-VAE-1-0": "DG-CVAE w/ Skip (ablation)",
    "Vanilla-0-0": "Baseline CVAE",
    "DG-VAE-0-1": "DG-CVAE w/ G-Skip (ours)",
}
translation_dataset = {
    "dailydialog": "DailyDialog",
    "yelp": "Yelp",
    "agnews": "AGNews",
}
indexed_metric = {
    #"D": "rougeL-a",
    #"H": "rougeL-b",
    "L": "rougeL",
    #"N": "length"
}

# 读取数据
#file_path = '/Users/zhangjianfei/Downloads/test_model_interpolation_results_v6.0 (5).csv'
#file_path = "test_model_interpolation_results_v7.0-1000-shots (2).csv"
file_path = "test_model_interpolation_results_v7.0-all-shots (2).csv"
df = pd.read_csv(file_path)

# 创建图形和子图
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(len(indexed_metric), len(translation_dataset),
                        figsize=(7*len(translation_dataset), 4*len(indexed_metric)))
fig.subplots_adjust(left=0.3, right=0.98, bottom=0.2, top=0.90, hspace=0.18, wspace=0.12)


# 存放每个子图的图例
lines = []
labels = []

for i, dataset in enumerate(translation_dataset.keys()):
    for j, (index, metric) in enumerate(indexed_metric.items()):
        for model in translation_model:
            #"D": "rougeL-a",
            #"H": "rougeL-b",
            model_name = f"{dataset}-{model}"
            metric_index = ord(index) - ord("A")
            try:
                model_row = df[df.iloc[:, 0] == model_name]
                value = eval(model_row.iloc[-1, metric_index])
                value_a = eval(model_row.iloc[-1, ord("D")-ord("A")])
                value_b = eval(model_row.iloc[-1, ord("H")-ord("A")])
                value = [a*(1-lamda/10)+b*lamda/10 for lamda,(a,b) in enumerate(zip(value_a,value_b))]
            except Exception as e:
                print(e)
                if j == 0:
                    print(f"unfound: {dataset} - {model}")
                value = [0] * 11  # unfound数据填充为[0] * 11

            # 横坐标
            x = np.linspace(0, 1, 11)
            # 纵坐标
            y = value
            # 绘制线条
            if not any(value):
                pass
            elif len(indexed_metric) > 1:
                if model == "DG-VAE-0-1":
                    line, = axs[j, i].plot(x, y, label=translation_model[model], marker='*', markersize=10, color='r')
                elif model in ["Vanilla-1-0", "DG-VAE-1-0", "DG-VAE-0-0"]:
                    line, = axs[j, i].plot(x, y, label=translation_model[model], marker='o', markersize=5)
                else:
                    line, = axs[j, i].plot(x, y, label=translation_model[model])
            else:
                if model == "DG-VAE-0-1":
                    line, = axs[i].plot(x, y, label=translation_model[model], marker='*', markersize=10, color='r')
                elif "DG" in model:
                    line, = axs[i].plot(x, y, label=translation_model[model], markerfacecolor='none', marker='*', markersize=10)
                elif model == "Vanilla-0-0" or model == "Vanilla-1-0":
                    line, = axs[i].plot(x, y, label=translation_model[model], markerfacecolor='none', marker='o', markersize=12)
                else:
                    line, = axs[i].plot(x, y, label=translation_model[model], markerfacecolor='none', marker='s')
            if i == 0 and j == 0:
                if model == "Vanilla-0-0" or model == "Vanilla-1-0" or model == "SFT":
                    lines.insert(0, line)
                    labels.insert(0, translation_model[model])
                else:
                    lines.append(line)
                    labels.append(translation_model[model])

        if len(indexed_metric) > 1:
            axs[i, j].set_title(translation_dataset[dataset])
            axs[i, j].set_xlabel(r"$\lambda$", fontsize=20)
            if i == 0:
                axs[i, j].set_ylabel("RougeL")
            #axs[i, j].set_ylim(0, 0.4)
        else:
            axs[i].set_title(translation_dataset[dataset])
            axs[i].set_xlabel(r"$\lambda$", fontsize=18)
            if i == 0:
                axs[i].set_ylabel("RougeL", fontsize=18)
            #axs[i].set_ylim(0, 0.4)



# 添加公共图例
fig.legend(lines, labels, loc='upper left', fontsize=18)

# 保存图形
plt.savefig('results_interpolation.pdf', dpi=300)
plt.close()
