import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 假设你已经有一个JSON文件路径
json_file_path = 'posterior_latent_and_probs_test_all_v7.json'

# 1. 读取JSON文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 2. 提取特征和标签
features = data["latent"]
labels = data["probs"]

X, Y = np.array(features), np.array(labels)
order = np.random.permutation(X.shape[0])
X = X[order, :]
Y = Y[order, :]
print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")

label_names = ["Inform", "Questions", "Directives", "Commissive"]

# 创建图形和子图
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(1, len(label_names), figsize=(13*63/59, 3))
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9, hspace=0.3, wspace=0.3)

for idx, label in enumerate(label_names):
    correlations = []
    for i in range(32):
        corr, _ = pearsonr(X[:, i], Y[:, idx])
        correlations.append(corr)
    correlations_abs = np.abs(correlations)
    top_two_indices = np.argsort(correlations_abs)[-2:]
    print(np.sort(correlations_abs)[-2:])

    # 使用散点图绘制每个点，颜色由标签决定
    axs[idx].scatter(
        X[:, top_two_indices[0]],
        X[:, top_two_indices[1]],
        c=Y[:, idx], cmap='Reds', edgecolor=None, s=3)
    axs[idx].set_xlabel(f'Dim = {top_two_indices[0]}', fontsize=8)
    axs[idx].set_ylabel(f'Dim = {top_two_indices[1]}', fontsize=8)
    axs[idx].set_title(label, fontsize=10)
    axs[idx].set_xlim(-3.5, 3.5)
    axs[idx].set_ylim(-3.5, 3.5)

# 添加公共色条
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), ax=axs, orientation='vertical')
cbar.set_label('Intention Probability', fontsize=10)

# 显示图像
plt.savefig("latent-probs-dailydialog.png", dpi=300)