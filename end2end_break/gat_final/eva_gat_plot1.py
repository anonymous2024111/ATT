import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = ['GitHub', 'blog', 'ognb-arxiv']

# 读取 CSV 文件（确保文件路径正确）
dgl = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/dgl.csv")
pyg = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/pyg.csv")
mgat16 = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/mgat16.csv")
mgat32 = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/mgat32.csv")

# 定义 a 系列和 s 系列的列名
a_cols = ['a0', 'a1', 'a2']
s_cols = ['s0', 's1', 's2']

# 计算加速比（DGL / MGATXX），数值越大表示 MGAT 相对 DGL 更快
sum_dgl_a = dgl[a_cols].sum(axis=1)
sum_gat16_a = mgat16[a_cols].sum(axis=1)
sum_gat32_a = mgat32[a_cols].sum(axis=1)
sum_pyg_a = pyg[a_cols].sum(axis=1)

speedup_a_dgl = sum_dgl_a / sum_gat16_a
speedup_a_pyg = sum_pyg_a / sum_gat16_a

sum_dgl_s = dgl[s_cols].sum(axis=1)
sum_gat16_s = mgat16[s_cols].sum(axis=1)
sum_gat32_s = mgat32[s_cols].sum(axis=1)
sum_pyg_s = pyg[s_cols].sum(axis=1)

speedup_s_dgl = sum_dgl_s / sum_gat16_s
speedup_s_pyg = sum_pyg_s / sum_gat16_s

# 配色
colors = sns.color_palette("Purples", 3)
colors1 = sns.color_palette("Greens", 3)


# 🔹 图 1：DGL / MGAT16 - Attention
df_gat_dgl = pd.DataFrame({
    'Speedup': speedup_a_dgl,
    'Group': ['DGL'] * len(speedup_a_dgl)
})

plt.figure(figsize=(3, 3))
sns.violinplot(data=df_gat_dgl, x='Group', y='Speedup', inner=None, linewidth=1.2, width=0.2, color='plum')
sns.stripplot(data=df_gat_dgl, x='Group', y='Speedup', color='black', size=5, jitter=True, alpha=0.9)
plt.hlines(df_gat_dgl['Speedup'].mean(), -0.3, 0.3, colors='k', linestyles='--', linewidth=1)

plt.gca().axes.get_xaxis().set_visible(False)  # 隐藏 X 轴
plt.tight_layout()
plt.savefig("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/gat_dgl.png", dpi=800)
plt.close()

# 🔹 图 2：PyG / MGAT16 - Attention
df_gat_pyg = pd.DataFrame({
    'Speedup': speedup_a_pyg,
    'Group': ['PyG'] * len(speedup_a_pyg)
})

plt.figure(figsize=(3, 3))
sns.violinplot(data=df_gat_pyg, x='Group', y='Speedup', inner=None, linewidth=1.2, width=0.2, color='lightgreen')
sns.stripplot(data=df_gat_pyg, x='Group', y='Speedup', color='black', size=5, jitter=True, alpha=0.9)
plt.hlines(df_gat_pyg['Speedup'].mean(), -0.3, 0.3, colors='k', linestyles='--', linewidth=1)

plt.gca().axes.get_xaxis().set_visible(False)  # 隐藏 X 轴
plt.tight_layout()
plt.savefig("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/gat_pyg.png", dpi=800)
plt.close()

# 🔹 图 3：DGL / MGAT16 - SpMM
df_spmm_dgl = pd.DataFrame({
    'Speedup': speedup_s_dgl,
    'Group': ['DGL'] * len(speedup_s_dgl)
})

plt.figure(figsize=(3, 3))
sns.violinplot(data=df_spmm_dgl, x='Group', y='Speedup', inner=None, linewidth=1.2, width=0.2, color='plum')
sns.stripplot(data=df_spmm_dgl, x='Group', y='Speedup', color='black', size=5, jitter=True, alpha=0.9)
plt.hlines(df_spmm_dgl['Speedup'].mean(), -0.3, 0.3, colors='k', linestyles='--', linewidth=1)

plt.gca().axes.get_xaxis().set_visible(False)  # 隐藏 X 轴
plt.tight_layout()
plt.savefig("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/spmm_dgl.png", dpi=800)
plt.close()

# 🔹 图 4：PyG / MGAT16 - SpMM
df_spmm_pyg = pd.DataFrame({
    'Speedup': speedup_s_pyg,
    'Group': ['PyG'] * len(speedup_s_pyg)
})

plt.figure(figsize=(3, 3))
sns.violinplot(data=df_spmm_pyg, x='Group', y='Speedup', inner=None, linewidth=1.2, width=0.2, color='lightgreen')
sns.stripplot(data=df_spmm_pyg, x='Group', y='Speedup', color='black', size=5, jitter=True, alpha=0.9)
plt.hlines(df_spmm_pyg['Speedup'].mean(), -0.3, 0.3, colors='k', linestyles='--', linewidth=1)

plt.gca().axes.get_xaxis().set_visible(False)  # 隐藏 X 轴
plt.tight_layout()
plt.savefig("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/spmm_pyg.png", dpi=800)
plt.close()