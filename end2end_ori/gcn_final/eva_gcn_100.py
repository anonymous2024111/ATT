import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
import os

current_dir = os.path.dirname(__file__)
# project_dir = os.path.dirname(os.path.dirname(current_dir))
           
sys.path.append(current_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from mydgl import test_dgl
# from mypyg import test_pyg
from mgat_csr import test_mgat_csr
from mgat32_csr import test_mgat32_csr

#DGL
def dglGCN(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
    
#MGCN
def mGCN16(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_mgat_csr.test(data, epoches, head,num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgat-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm= test_mgat32_csr.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgat-tf32-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
# def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
#     spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
#     print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
#     res = []
#     res.append(data)
#     res.append(str(num_layers) + '-' + str(hidden))
#     res.append(spmm)
#     with open(csv_file, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(res)  



if __name__ == "__main__":


    dataset_I = ['soc-Epinions1', 'GitHub', 'artist', 'blog']

    dataset_II = ['IGB_small', 'com-DBLP', 'amazon', 'amazon0505', 
                  'dd', 'comamazon', 'yeast']
    
    dataset_III = ['reddit']
    
    dataset1 = ['ogbn-arxiv', 'ogbn-proteins']
    
    dataset_large = ['AmazonProducts', 'IGB_medium', 'ogb' ]

    dataset = dataset_I + dataset_II + dataset_III + dataset1
   
    epoches = 300
    # layer = [2, 4]
    # hidden = [64, 128, 256]
    layer = [3]
    hidden = [64, 128]
    head = [1]
    
    
    
    featuredim = 64
    classes = 10


    #DGL
    # filename = current_dir + '/result/dgl-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for head_num in head:
    #     for layer_num in layer:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 dglGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    # print('DGL-' + 'success')
    # print()    
    #warmup
    filename = current_dir + '/result/warmup.csv'
    for head_num in head:
        for layer_num in layer:
            for hidden_num in [64]:
                for data in ['soc-Epinions1']:
                    mGCN16(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')   
    print()     
      
    #MGCN-fp16
    filename = current_dir + '/result/mgat16.csv'
    # with open(filename, 'w', newline='') as file:
    #     file.write('data,config,gat16\n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset_large:
                    mGCN16(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')   
    print()         
 
    #MGCN-tf32   
    filename = current_dir + '/result/mgat32.csv'
    # with open(filename, 'w', newline='') as file:
    #     file.write('data,config,gat32\n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset_large:
                    mGCN32(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-tf32-' + 'success')
    print()
    