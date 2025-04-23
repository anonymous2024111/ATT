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
df_dgl = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_head/gat_final/result/dgl.csv')
df_att_fp16 = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_head/gat_final/result/mgat16.csv')
df_att_tf32 = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_head/gat_final/result/mgat32.csv')
# df_pyg = pd.read_csv('/home/shijinliang/module/tpds/ATT/end2end_ori/gat_final/result/pyg.csv')

df = pd.merge(df_dgl, df_att_fp16, on=['data', 'config'], how='inner')  
df = pd.merge(df, df_att_tf32, on=['data', 'config'], how='inner')  
# temp = pd.merge(temp, df_pyg, on=['data', 'config'], how='inner')  
# 遍历每个数据集
# 获取所有唯一的数据集名称

df["speedup_gat16"] = df["dgl"] / df["gat16"]
df["speedup_gat32"] = df["dgl"] / df["gat32"]
geo = round(stats.gmean(df["speedup_gat16"]),2)
print("geo-FP16: ", geo )
print("max-FP16: ", max(df["speedup_gat16"]) )
geo = round(stats.gmean(df["speedup_gat32"] ),2)
print("geo-TF32: ", geo )
print("max-TF32: ", max(df["speedup_gat32"] ) )


datasets = df["data"].unique()
colors = sns.color_palette("Greens", 6)  # 获取五个蓝色的渐变颜色
colors1 = sns.color_palette("Purples", 6)  # 获取五个蓝色的渐变颜色
patterns = ['#', '\\', '/', '\\', '|-']
# 遍历每个数据集绘图
for dataset in datasets:
    subset = df[df["data"] == dataset]

    # 创建新的图像
    plt.figure(figsize=(4, 4))
    
    # 创建主 y 轴（柱状图）
    ax1 = plt.gca()  # 获取当前轴
    bar_width = 0.2
    x_positions = np.arange(len(subset["config"]))

    ax1.bar(x_positions, subset["dgl"]/300*1000, width=bar_width, label="DGL", align='center', alpha=1, color='lightgrey', hatch=patterns[0], edgecolor='black')
    ax1.bar(x_positions + bar_width, subset["gat16"]/300*1000, width=bar_width, label="GAT16", align='center', alpha=1, color=colors1[3], edgecolor='black')
    ax1.bar(x_positions + 2*bar_width, subset["gat32"]/300*1000, width=bar_width, label="GAT32", align='center', alpha=1, color=colors1[2], edgecolor='black')
    
    # 设置 y 轴
    ax1.set_ylabel("Execution Time")
    ax1.set_xticks([])
    
    # 创建次 y 轴（折线图）
    ax2 = ax1.twinx()
    ax2.scatter(x_positions + bar_width, subset["speedup_gat16"], marker='^', linestyle='-', color='black', label="GAT16 Speedup", s=150)
    ax2.scatter(x_positions + 2*bar_width, subset["speedup_gat32"], marker='v', linestyle='-', color='black', label="GAT32 Speedup", s=150)
    ax2.set_ylabel("Speedup vs DGL")

    # # 图例
    # ax1.legend(loc="upper left")
    # ax2.legend(loc="upper right")
    
    # # 设置柱状图参数
    # plt.figure(figsize=(4, 5))
    # bar_width = 0.26
    # x_labels = subset["config"]
    # x_positions = range(len(x_labels))
    
    # df["speedup_gat16"] = df["dgl"] / df["gat16"]
    # df["speedup_gat32"] = df["dgl"] / df["gat32"]

    # plt.bar(x_positions, subset["dgl"], width=bar_width, label="DGL", align='center', alpha=1, color=colors[0], hatch=patterns[0], edgecolor='black')
    # plt.bar([x + bar_width for x in x_positions], subset["gat32"], width=bar_width, label="GAT32", align='center', alpha=1, color=colors[1], hatch=patterns[1], edgecolor='black')
    # plt.bar([x + 2 * bar_width for x in x_positions], subset["gat16"], width=bar_width, label="GAT16", align='center', alpha=1, color=colors[2], hatch=patterns[2], edgecolor='black')
    plt.xticks([])



    plt.savefig('/home/shijinliang/module/tpds/ATT/end2end_head/gat_final/plot/' + dataset + '.png', dpi=800)

