import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
# sys.path.append('./eva100/end2end_ori/gat_no_pre')
import os

current_dir = os.path.dirname(__file__)
# project_dir = os.path.dirname(os.path.dirname(current_dir))
           
sys.path.append(current_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from mydgl import test_dgl
from mypyg import test_pyg
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
    
    
#MGCN
def mGCN16(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_mgat_csr.test(data, epoches, head,num_layers, featuredim, hidden, classes)
    print(str(head) + '-' + str(hidden) + '-' + data + '-mgat-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(head) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm= test_mgat32_csr.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(head) + '-' + str(hidden) + '-' + data + '-mgat-tf32-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(head) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
    print(str(head) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(head) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  



if __name__ == "__main__":


    dataset_I = ['GitHub', 'blog', 'ogbn-arxiv']
    
    dataset_II = ['artist', 'amazon', 'reddit']


    dataset = dataset_I
    
    epoches = 300
    layer = [3]
    hidden = [128]
    head = [8]
    
    
    
    featuredim = 128
    classes = 10


    # #DGL
    # filename = current_dir + '/result/dgl-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,config,dgl : \n')
    # for head_num in head:
    #     for layer_num in layer:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 dglGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    # print('DGL-' + 'success')
    # print()    
            
    #MGCN-fp16
    filename = current_dir + '/result/mgat16.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,config,gat16 : \n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset:
                    mGCN16(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')   
    print()         
 
    # #MGCN-tf32   
    filename = current_dir + '/result/mgat32.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,config,gat32 : \n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset:
                    mGCN32(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-tf32-' + 'success')
    print()
    
    # # #PYG
    # filename = './eva100/end2end/gat_no_pre/result/pyg-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,config,pyg : \n')
    # for head_num in head:
    #     for layer_num in layer:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 pygGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)
    # print('Pyg-' + 'success')
