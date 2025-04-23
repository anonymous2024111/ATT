import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
result = {}
hidden = [64, 128]
head = [1, 8]
layer = [3]
dataset = []

gat16_sp_64 = []
gat32_sp_64 = []
pyg_sp_64 = []

gat16_sp_128 = []
gat32_sp_128 = []
pyg_sp_128 = []

# 各个表格的结果进行内连接

#读取dgl的数据
df_dgl = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_ori/gsage_final/result/dgl.csv')
df_att_fp16 = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_ori/gsage_final/result/mgat16.csv')
df_att_tf32 = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_ori/gsage_final/result/mgat32.csv')
df_pyg = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_ori/gsage_final/result/pyg.csv')

temp = pd.merge(df_dgl, df_att_fp16, on=['data', 'config'], how='inner')  
temp = pd.merge(temp, df_att_tf32, on=['data', 'config'], how='inner')  
temp = pd.merge(temp, df_pyg, on=['data', 'config'], how='inner')  
# 提取数据集名称
datasets = []
filter= ['ogb', 'AmazonProducts', 'IGB_medium']
gat16_sp_64_time = []
for index, row in temp.iterrows():
    if row['config'] == '3-64':
        gat16_sp_64.append(round((row['dgl']/row['gat16']),2)) 
        gat32_sp_64.append(round((row['dgl']/row['gat32']),2)) 
        pyg_sp_64.append(round((row['dgl']/row['pyg']),2)) 
        datasets.append(row['data'])
        gat16_sp_64_time.append(row['gat16'])
        
    if row['config'] == '3-128':
        gat16_sp_128.append(round((row['dgl']/row['gat16']),2)) 
        gat32_sp_128.append(round((row['dgl']/row['gat32']),2)) 
        if row['pyg']== 10000 :
            pyg_sp_128.append(0)
        else:
            pyg_sp_128.append(round((row['dgl']/row['pyg']),2)) 

gat16_avg = np.mean([gat16_sp_64, gat16_sp_128], axis=0)
gat32_avg = np.mean([gat32_sp_64, gat32_sp_128], axis=0)
filtered_gat16_avg = [x for x in gat16_avg if x != 0]
filtered_gat32_avg = [x for x in gat32_avg if x != 0]
geo = round(stats.gmean(filtered_gat16_avg),2)
print("geo-FP16: ", geo )
print("max-FP16: ", max(filtered_gat16_avg) )
geo = round(stats.gmean(filtered_gat32_avg),2)
print("geo-TF32: ", geo )
print("max-TF32: ", max(filtered_gat32_avg) )
print(datasets)
ind = np.arange(len(gat16_sp_64))  # 每个索引间隔变大
width = 0.25  # 保证宽度不会过大或过小
# 绘制柱状图
fig, ax = plt.subplots(figsize=(16, 4))
colors = sns.color_palette("Purples", 6)  # 获取五个蓝色的渐变颜色
# 每组柱状图的纹理样式
patterns = ['/', 'x', '-', '\\', '|-']
plot_list = []
print(dataset)
# 第一组柱状图
bar1 = ax.bar(ind - width, gat16_avg, width, label='TCGAT-FP16', color=colors[3], edgecolor='black', linewidth=1.5, zorder=2)
for i, rect in enumerate(bar1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 0.03, f'{gat16_avg[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12, zorder=3)

# 第二组柱状图
bar2 = ax.bar(ind, gat32_avg, width, label='TCGAT-TF32', color=colors[2], edgecolor='black', linewidth=1.5, zorder=2)
for i, rect in enumerate(bar2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 0.03, f'{gat32_avg[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12, zorder=3)

# 第三组柱状图
bar3 = ax.bar(ind + width, pyg_sp_64, width, label='PyG', color='lightseagreen', edgecolor='black', linewidth=1.5, zorder=2)
for i, rect in enumerate(bar3):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 0.03, f'{pyg_sp_64[i]:.2f}x', ha='center', va='bottom', rotation=90, fontsize=12, zorder=3)
ax2 = ax.twinx()
ax2.set_yscale('log')  # Use log scale
ax2.tick_params(axis='y', labelsize=16)
# Plot the time points as "coin-like" scatter markers
ax2.scatter(ind, gat16_sp_64_time, color='black', marker='*', s=150, label='GAT16 Time', zorder=2)

# 其他设置
ax.tick_params(axis='y', which='major', labelsize=18, width=1)  # 设置刻度
ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, zorder=0)
ax.xaxis.set_visible(False)  # 隐藏 x 轴刻度
# 设置横坐标刻度和标签
# ax.set_xticks(datasets)
# ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=14)
# 保存图像
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end_ori/gsage_final/h100-sage.png', dpi=800)

# 关闭图形
plt.close()