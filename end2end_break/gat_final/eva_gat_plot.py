import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

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

speedup_a_mgat16 = sum_dgl_a / sum_gat16_a
speedup_a_mgat32 = sum_dgl_a / sum_gat32_a
speedup_a_pyg = sum_dgl_a / sum_pyg_a

geo = round(stats.gmean(speedup_a_mgat16),2)
print("Attention: ")
print("geo-Fp16: ", geo )
print("max-Fp16: ", max(speedup_a_mgat16) )

sum_dgl_s = dgl[s_cols].sum(axis=1)
sum_gat16_s = mgat16[s_cols].sum(axis=1)
sum_gat32_s = mgat32[s_cols].sum(axis=1)
sum_pyg_s = pyg[s_cols].sum(axis=1)

speedup_s_mgat16 = sum_dgl_s / sum_gat16_s
speedup_s_mgat32 = sum_dgl_s / sum_gat32_s
speedup_s_pyg = sum_dgl_s / sum_pyg_s

geo = round(stats.gmean(speedup_s_mgat16),2)
print("Aggregation: ")
print("geo-Fp16: ", geo )
print("max-Fp16: ", max(speedup_s_mgat16) )

colors1 = sns.color_palette("Purples", 6)  # 获取五个蓝色的渐变颜色
colors = sns.color_palette("Greens", 3)

# 创建新的图像
plt.figure(figsize=(4, 2))

# 创建主 y 轴（柱状图）
ax1 = plt.gca()  # 获取当前轴
bar_width = 0.2
x_positions = np.arange(len(speedup_a_mgat16))

ax1.bar(x_positions, speedup_a_pyg, width=bar_width, label="DGL", align='center', alpha=1, color='lightseagreen',  edgecolor='black')
ax1.bar(x_positions + bar_width, speedup_a_mgat16, width=bar_width, label="GAT16", align='center', alpha=1, color=colors1[3], edgecolor='black')
ax1.bar(x_positions + 2*bar_width, speedup_a_mgat32, width=bar_width, label="GAT32", align='center', alpha=1, color=colors1[2], edgecolor='black')
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, zorder=0)
ax1.get_xaxis().set_visible(False)
# 显示图形
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/gat_att.png', dpi=800)
plt.close()

plt.figure(figsize=(4, 2))
ax1 = plt.gca()  # 获取当前轴
bar_width = 0.2
x_positions = np.arange(len(speedup_a_mgat16))
# 设置每个柱子的 x 轴标签
ax1.bar(x_positions, speedup_s_pyg, width=bar_width, label="DGL", align='center', alpha=1, color='lightseagreen', edgecolor='black')
ax1.bar(x_positions + bar_width, speedup_s_mgat16, width=bar_width, label="GAT16", align='center', alpha=1, color=colors1[3], edgecolor='black')
ax1.bar(x_positions + 2*bar_width, speedup_s_mgat32, width=bar_width, label="GAT32", align='center', alpha=1, color=colors1[2], edgecolor='black')
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8,zorder=0)
ax1.get_xaxis().set_visible(False)
# 显示图形
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/gat_spmm.png', dpi=800)
