import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('./eva100/end2end_break/gat_no_pre')
import os

current_dir = os.path.dirname(__file__)
# project_dir = os.path.dirname(os.path.dirname(current_dir))
print(current_dir)           
sys.path.append(current_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mydgl import test_dgl
# from mypyg import test_pyg
from mgat_csr import test_mgat_csr
from mgat32_csr import test_mgat32_csr

#DGL
def dglGCN(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    with open("/home/shijinliang/module/tpds/ATT/end2end_break/gatv2_final/result/dgl.csv", "a") as f:
        f.write(data + ',')  # 追加换行符
    spmm = test_dgl.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    with open("/home/shijinliang/module/tpds/ATT/end2end_break/gatv2_final/result/dgl.csv", "a") as f:
        f.write("\n")  # 追加换行符



if __name__ == "__main__":


    dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                 'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                  'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                  'roadNet-TX', 'yeast']
    dataset_III = ['reddit', 'ogb', 'AmazonProducts']
    
    data_back = [ 'cora', 'ell-acc', 'pubmed', 'question', 'min']


    dataset1 = ['ogbn-arxiv', 'ogbn-proteins']
    dataset = dataset_I + dataset_II + dataset_III + dataset1
    dataset =  ['GitHub', 'blog', 'ogbn-arxiv']
    epoches = 1
    layer = [3]
    hidden = [64]
    head = [1]
    
    
    
    featuredim = 256
    classes = 10

    filename = current_dir + '/result/warmup.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in ['soc-Epinions1']:
                    dglGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('DGL-' + 'success')
    
    #DGL
    with open("/home/shijinliang/module/tpds/ATT/end2end_break/gatv2_final/result/dgl.csv", "w") as f:
        f.write('H100 : \n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset:
                    dglGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('DGL-' + 'success')
    print()    
   