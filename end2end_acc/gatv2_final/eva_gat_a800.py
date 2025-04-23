import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from mydgl import test_dgl
from mypyg import test_pyg
from mgat_csr import test_mgat_csr
from mgat32_csr import test_mgat32_csr
#DGL
def dglGCN(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + str(head) + '-' + data + '-dgl-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers))
    res.append(str(hidden))
    res.append(str(head))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
    
#MGCN
def mGCN16(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_mgat_csr.test(data, epoches, head,num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + str(head) + '-' + data + '-mgat-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers))
    res.append(str(hidden))
    res.append(str(head))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm= test_mgat32_csr.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + str(head) + '-' + data + '-mgat-tf32-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers))
    res.append(str(hidden))
    res.append(str(head))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers))
    res.append(str(hidden))
    res.append(str(head))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  



if __name__ == "__main__":


    dataset_I = ['artist', 'Coauthor_Physics',  'FacebookPagePage', 'GitHub', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'amazon', 'amazon0505', 
                  'dd', 'comamazon', 'yeast']
    dataset_III = ['reddit', 'ogb', 'AmazonProducts', 'IGB-medium', 'IGB-small']

    dataset = dataset_I + dataset_II
    # dataset = [ 'email-Enron']
    # dataset_II_pyg =['ell',  'com-DBLP', 'amazon0505', 
    #               'dd',  'comamazon', 'roadNet-PA']
    # dataset_pyg = dataset_I + dataset_II_pyg
    
    dataset = ['Reddit2', 'amazon', 'yeast', 'ogb', 'IGB_small', 'IGB_medium']
    epoches = 300
    layer = [3]
    hidden = [64, 128]
    head = [1, 8]
    featuredim = 256
    classes = 16


    # layer = [3]
    # hidden = [128]
    # head = [8]
    dataset = ['ell']
    
    dataset = ['artist', 'Coauthor_Physics',  'FacebookPagePage', 'GitHub', 'blog', 'pubmed', 'email-Enron', 'loc-Brightkite','Reddit2', 'amazon', 'yeast', 'ogb', 'IGB_small', 'IGB_medium']
    header = ['data', 'layers', 'hidden', 'heads', 'time']
    dataset = ['yeast', 'ogb', 'IGB_small']
    for data in dataset:
        # #DGL
        # filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/dgl/dgl_'+ data + '.csv'
        # with open(filename, 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(header)
        # for layer_num in layer:
        #     for hidden_num in hidden:
        #         for head_num in head:
        #             dglGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
        # print('DGL-' + 'success')
                
        #MGCN-fp16
        filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/gat16/gat16_new_'+ data + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
        for layer_num in layer:
            for hidden_num in hidden:
                for head_num in head:
                    mGCN16(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
        print('GAT-fp16-' + 'success')            
 
        # # # #MGCN-tf32   
        # filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/gat32/gat32_'+ data + '.csv'
        # with open(filename, 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(header)
        # for layer_num in layer:
        #     for hidden_num in hidden:
        #         for head_num in head:
        #             mGCN32(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
        # print('GAT-tf32-' + 'success')
   
    # #PYG
    # filename = './eva100/end2end/gat_no_pre/resulta800/pyg-v2.csv'
    # # with open(filename, 'w') as file:
    # #     file.write('H100 : \n')
    # for layer_num in layer:
    #     for hidden_num in hidden:
    #         if layer_num==4 and hidden_num==256:
    #             for data in dataset_pyg:
    #                 pygGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)  
    #         else:
    #             for data in dataset:
    #                 if data == 'Reddit2' or data == 'yelp':
    #                     continue
    #                 pygGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)
    # print('Pyg-' + 'success')

    # # print('MGAT_small_all success')