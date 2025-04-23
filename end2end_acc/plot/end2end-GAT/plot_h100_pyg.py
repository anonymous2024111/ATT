import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
result = {}
hidden = [64, 128]
head = [1, 8]
layer = [3]
dataset = ['artist', 'Coauthor_Physics',  'FacebookPagePage', 'GitHub', 'blog', 'pubmed', 'email-Enron', 'loc-Brightkite']
# 遍历每个数据集
for data in dataset:
    result[data] = {'dgl': [], 'pyg': []}
    dgl = []
    gat16 = []
    pyg = []
    #读取dgl的数据
    df_att = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/result/gat16/gat16_new_'+ data + '.csv')
    for index, row in df_att.iterrows():
        gat16.append(row['time'])
        
    df_pyg = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/result/pyg/pyg_'+ data + '.csv')
    for index, row in df_pyg.iterrows():
        pyg.append(row['time'])
        
    #计算加速比
    for pyg_, gat16_ in zip(pyg, gat16):
        result[data]['pyg'].append(round((pyg_/gat16_),2)) 
 # 绘制DGL
ind = np.arange(8)  # 柱状图的 x 坐标位置
width = 0.2  # 柱状图的宽度
# 绘制柱状图
fig, ax = plt.subplots(figsize=(16, 5))

# 每组柱状图的纹理样式
patterns = ['/', 'x', '-', '\\', '|-']
plot_list = []

# 第一组柱状图
for data in dataset:
    plot_list.append(result[data]['pyg'][0])
bar1 = ax.bar(ind - 1.5*width, plot_list, width, label='Libra-TF32', color='lightgreen', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.06, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12, zorder=3)

# 第二组柱状图
plot_list = []
for data in dataset:
    plot_list.append(result[data]['pyg'][2])
bar2 = ax.bar(ind - 0.5*width, plot_list, width, label='Libra-TF32', color='khaki', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.06, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

# 第三组柱状图
plot_list = []
for data in dataset:
    plot_list.append(result[data]['pyg'][1])
bar3 = ax.bar(ind + 0.5*width, plot_list, width, label='Libra-TF32', hatch='//', color='lightskyblue', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar3):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.06, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

# 第四组柱状图
plot_list = []
for data in dataset:
    plot_list.append(result[data]['pyg'][3])
bar4 = ax.bar(ind + 1.5*width, plot_list, width, label='Libra-TF32', hatch=patterns[1], color='lightcoral', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar4):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.06, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

ax.tick_params(axis='y', which='major', labelsize=18, width=1)  # 设置刻度大小和宽度
plt.axhline(y=1, color='black', linestyle='--', linewidth=2, zorder=-1)

# 设置 y 轴的刻度范围
ax.set_ylim(0, 9)  # 设置 y 轴从 0 到 9 的范围，确保 "1" 显示出来

# 显示 y 轴上的所有刻度，包括 "1"
ax.set_yticks(np.arange(0, 10, 1))  # 设置 y 轴的刻度为 [0, 1, 2, ..., 9]

# 显示网格和保存图像
# plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
ax.xaxis.set_visible(False)
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end/plot/end2end/h100—pyg.png', dpi=800)

# 清空图形
plt.clf()
