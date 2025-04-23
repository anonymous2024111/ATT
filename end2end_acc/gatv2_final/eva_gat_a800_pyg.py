import torch
import numpy as np
# from scipy.sparse import *
import time
import csv
import sys
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from mypyg import test_pyg



#MGPYG
def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-'  + str(head) + '-' + data + '-pyg-' + '-' + str(spmm))
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
    
    dataset = ['IGB_small',  'yeast', 'GitHub', 'ell', 'Reddit2', 'amazon', 'ogb', 'FacebookPagePage']
    epoches = 300
    layer = [3]
    hidden = [64, 128]
    head = [1, 8]
    featuredim = 256
    classes = 16


    # layer = [3]
    # hidden = [128]
    # head = [8]
    dataset = ['artist', 'Coauthor_Physics',  'FacebookPagePage', 'GitHub', 'blog', 'pubmed', 'email-Enron', 'loc-Brightkite']
    
    header = ['data', 'layers', 'hidden', 'heads', 'time']

    for data in dataset:
        #PYG
        filename = '/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi/resulta800/pyg/pyg_'+ data + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
        for layer_num in layer:
            for hidden_num in hidden:
                for head_num in head:
                    pygGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
        print('Pyg-' + 'success')

    # # print('MGAT_small_all success')