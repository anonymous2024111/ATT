import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('./eva100/end2end_ori/gat_no_pre')
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
    spmm = test_dgl.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(head) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(head) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    


if __name__ == "__main__":


    dataset_I = ['soc-Epinions1', 'GitHub', 'artist', 'blog']

    dataset_II = ['Reddit2',  'com-DBLP', 'amazon', 'amazon0505', 
                  'dd', 'yelp', 'comamazon', 'yeast']
    
    dataset_III = ['reddit', 'AmazonProducts']
    
    dataset1 = ['ogbn-arxiv', 'ogbn-proteins', 'ogb']

    dataset = dataset_I + dataset_II + dataset_III + dataset1

    epoches = 300
    layer = [2]
    hidden = [64]
    head = [1, 4, 8]
    
    
    
    featuredim = 128
    classes = 10


    #DGL
    filename = current_dir + '/result/dgl.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset:
                    dglGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('DGL-' + 'success')
    print()    
            
   