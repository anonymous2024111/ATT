import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

# 组标签
labels = ['GitHub', 'Blog', 'ogbn-arxiv']
y = np.arange(len(labels))  # [0, 1, 2]

bar_height = 0.28

gat16_pre = [0.019, 0.08, 0.01]
gat16_train= [1.3973, 4.7865, 6.0214]
gat16_ = []
gat16_1 = []
for pre, train in zip(gat16_pre, gat16_train):
    ratio = pre / (pre + train)
    gat16_.append(round(ratio*100,2))
    gat16_1.append(100-round(ratio*100,2))

gat32_pre = [0.019, 0.08, 0.01]
gat32_train= [1.7696, 5.3727, 6.9271]
gat32_ = []
gat32_1 = []
for pre, train in zip(gat32_pre, gat32_train):
    ratio = pre / (pre + train)
    gat32_.append(round(ratio*100,2))
    gat32_1.append(100-round(ratio*100,2))

print(gat16_)
print(gat16_1)
print(gat32_)
print(gat32_1)

plt.figure(figsize=(5, 2.3))

colors = sns.color_palette("Purples", 6)
colors1 = sns.color_palette("Greens", 6)
patterns = ['#', '\\', '/', '\\', '|-']
# GAT16 堆叠两层（左边）
plt.barh(y - bar_height / 2, gat16_, height=bar_height, label='GAT16-Forward', color=colors[2])
plt.barh(y - bar_height / 2, gat16_1, height=bar_height, left=gat16_pre, label='GAT16-Backward', color=colors1[1])

# GAT32 堆叠两层（右边）
plt.barh(y + bar_height / 2, gat32_, height=bar_height, label='GAT32-Forward', color=colors[2])
plt.barh(y + bar_height / 2, gat32_1, height=bar_height, left=gat32_pre, label='GAT32-Backward', color=colors1[2])

# plt.xscale('log')
# 坐标轴和标签
plt.yticks(y, labels)
plt.xlabel("Execution Time (ms)")
plt.title("GAT16 vs GAT32: Forward + Backward Stacked Horizontal Bars")
# plt.legend()
plt.tight_layout()

plt.savefig('/home/shijinliang/module/tpds/ATT/end2end_ori_overhead/overhead.png', dpi=800)

