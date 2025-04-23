import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('./eva100/end2end/gat_acc_final')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mydgl import test_dgl
from mypyg import test_pyg
from mgat_csr import test_mgat_csr
from mgat32_csr import test_mgat32_csr
#DGL
def dglGCN(data, csv_file, epoches, head, hidden, layer):
    mean_value, std_dev  = test_dgl.test(data, epoches, head, hidden, layer)
    print(data + '-dgl-' + str(mean_value))
    res = []
    res.append(data)
    res.append(str(head))
    res.append(str(mean_value))
    res.append(str(std_dev))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
    
#MGCN
def mGCN16(data, csv_file, epoches, head, hidden, layer):
    mean_value, std_dev = test_mgat_csr.test(data, epoches, head, hidden, layer)
    print(data + '-mgat-fp16-'  + str(mean_value))
    res = []
    res.append(data)
    res.append(str(head))
    res.append(str(mean_value))
    res.append(str(std_dev))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, head, hidden, layer):
    mean_value, std_dev = test_mgat32_csr.test(data, epoches, head, hidden, layer)
    print(data + '-mgat-tf32-'  + str(mean_value))
    res = []
    res.append(data)
    res.append(str(head))
    res.append(str(mean_value))
    res.append(str(std_dev))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
def pygGCN(data, csv_file, epoches, head, hidden, layer):
    mean_value, std_dev = test_pyg.test(data, epoches, head, hidden, layer)
    print(data + '-pyg-' + '-' + str(mean_value))
    res = []
    res.append(data)
    res.append(str(head))
    res.append(str(mean_value))
    res.append(str(std_dev))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  



if __name__ == "__main__":

    dataset = [ 'cora', 'pubmed', 'question', 'min', 'texas', 'wiki', 'wis']
    dataset = ['ogbn-arxiv']
    epoches = 300
    layer = 3
    hidden = 128

    heads = [4, 8]
    
  
    # #DGL
    # filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_acc_final/result/dgl-acc.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,head,dgl,dgl- \n')
    # for head in heads:
    #     for data in dataset:
    #         dglGCN(data, filename, epoches, head, hidden, layer)
    # print('DGL-' + 'success')
                
    #MGCN-fp16
    # filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_acc_final/result/mgcn16-acc.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,head,gat16,gat16- \n')
    # for head in heads:
    #     for data in dataset:
    #         mGCN16(data, filename, epoches, head, hidden, layer)
    # print('MGCN-fp16-' + 'success')            
 
    # # #MGCN-tf32   
    filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_acc_final/result/mgcn32-acc.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,head,gat32,gat32- \n')
    for head in heads:
        for data in dataset:
            mGCN32(data, filename, epoches, head, hidden, layer)
    print('MGCN-tf32-' + 'success')

    # # #PYG
    # filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_acc_final/result/pyg-acc.csv'
    # with open(filename, 'w') as file:
    #     file.write('data,head,pyg,pyg- \n')
    # for head in heads:
    #     for data in dataset:
    #         pygGCN(data, filename, epoches, head, hidden, layer)
    # print('Pyg-' + 'success')

    # # print('MGAT_small_all success')