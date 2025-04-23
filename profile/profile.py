import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.sparse import coo_matrix


dataset_I = ['soc-Epinions1', 'GitHub', 'artist', 'blog']

dataset_II = ['IGB_small', 'com-DBLP', 'amazon', 'amazon0505', 
                'dd', 'comamazon', 'yeast']

dataset_III = ['reddit', 'AmazonProducts']

dataset1 = ['ogbn-arxiv', 'ogbn-proteins', 'ogb']

dataset = dataset_I + dataset_II + dataset_III + dataset1

# dataset = ['IGB_medium']

for data in dataset:
    
    graph = np.load('/public/home/shijinliang/gnns/' + data +'.npz')
    src_li=graph['src_li']
    dst_li=graph['dst_li']
    spar1 = len(src_li) / (graph['num_nodes'] )
    print(data + ': ' + str(round(graph['num_nodes']/1000, 0)) + '; ' + str(round(len(src_li)/1000, 0)) + '; ' + str(round(spar1, 1)))