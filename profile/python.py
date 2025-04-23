import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.sparse import coo_matrix


dataset_I = ['soc-Epinions1', 'GitHub', 'blog']

dataset_II = [ 'artist','Reddit2',  'com-DBLP', 'amazon', 'amazon0505', 
                'dd', 'comamazon', 'yeast']

dataset_III = ['reddit', 'AmazonProducts']

dataset1 = ['ogbn-arxiv', 'ogbn-proteins', 'ogb']

dataset = dataset_I + dataset_II + dataset_III + dataset1
dataset = ['IGB_medium']

for data in dataset:
    
    graph = np.load('/public/home/shijinliang/gnns/' + data +'.npz')
    src_li=graph['src_li']
    dst_li=graph['dst_li']

    # 随机采样 50% 的边
    sample_size = len(src_li) // 100 # 取一半
    sample_indices = np.random.choice(len(src_li), sample_size, replace=False)  # 随机选择

    # 采样后的边
    src_sampled = np.array(src_li)[sample_indices]
    dst_sampled = np.array(dst_li)[sample_indices]
    # 画散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(src_sampled, dst_sampled, s=0.0001, marker='o', color='blue')
    plt.xticks([])  # 去掉 x 轴刻度
    plt.yticks([])  # 去掉 y 轴刻度
    # plt.xlabel("Column Index")
    # plt.ylabel("Row Index")
    # plt.title("Sparse Matrix Scatter Plot")
    # plt.gca().invert_yaxis()  # 使 (0,0) 位置在左上角
    plt.savefig('/home/shijinliang/module/tpds/ATT/profile/' + data + '.png', dpi=800)
    # 关闭图形
    plt.close()
    print(data + ' is success.')