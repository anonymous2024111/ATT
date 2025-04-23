import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
result = {}
hidden = [64, 128]
head = [1, 8]
layer = [3]
dataset = ['IGB_small',  'yeast', 'GitHub', 'ell', 'Reddit2', 'amazon', 'ogb', 'FacebookPagePage']
# 遍历每个数据集
for data in dataset:
    result[data] = {'dgl': [], 'pyg': []}
    dgl = []
    gat16 = []
    pyg = []
    #读取dgl的数据
    df_dgl = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/dgl/dgl_'+ data + '.csv')
    for index, row in df_dgl.iterrows():
        dgl.append(row['time'])
        
    df_att = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/gat16/gat16_new_'+ data + '.csv')
    for index, row in df_att.iterrows():
        gat16.append(row['time'])
        
    # df_pyg = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/pyg/pyg_'+ data + '.csv')
    # for index, row in df_pyg.iterrows():
    #     pyg.append(row['time'])
        
    #计算加速比
    for dgl_, gat16_ in zip(dgl, gat16):
        result[data]['dgl'].append(round((dgl_/gat16_),2)) 
        # result[data]['pyg'].append(round((pyg_/gat16_),2)) 
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
    plot_list.append(result[data]['dgl'][0])
bar1 = ax.bar(ind - 1.5*width, plot_list, width, label='Libra-TF32', color='lightgreen', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.03, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

# 第二组柱状图
plot_list = []
for data in dataset:
    plot_list.append(result[data]['dgl'][2])
bar2 = ax.bar(ind - 0.5*width, plot_list, width, label='Libra-TF32', color='khaki', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.03, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

# 第三组柱状图
plot_list = []
for data in dataset:
    plot_list.append(result[data]['dgl'][1])
bar3 = ax.bar(ind + 0.5*width, plot_list, width, label='Libra-TF32', hatch='//', color='lightskyblue', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar3):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.03, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

# 第四组柱状图
plot_list = []
for data in dataset:
    plot_list.append(result[data]['dgl'][3])
bar4 = ax.bar(ind + 1.5*width, plot_list, width, label='Libra-TF32', hatch=patterns[1], color='lightcoral', edgecolor='black', linewidth=1.5,  zorder=2)

# 在每个柱子顶部显示值（竖着显示）
for i, rect in enumerate(bar4):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height+0.03, f'{plot_list[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12,  zorder=3)

ax.tick_params(axis='y', which='major', labelsize=18, width=1)  # 设置刻度大小和宽度
plt.axhline(y=1, color='black', linestyle='--', linewidth=2, zorder=-1)
# 显示网格和保存图像
# plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
ax.xaxis.set_visible(False)
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end/plot/end2end/a800—dgl.png', dpi=800)

# 清空图形
plt.clf()
