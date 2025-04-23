import os.path as osp
import argparse
import time
import torch
import sys
import csv
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end_ori_overhead/gat_final')
from mgat_csr_ov.mdataset_fp16 import *
from mgat_csr_ov.mgat_conv import *
from mgat_csr_ov.gat_mgnn import *


def test(data, epoches, heads, layers, featuredim, hidden, classes):



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGAT_dataset_csr(data, featuredim, classes) 
    inputInfo.to(device)
    res=[]
    res.append(data)
    res.append(inputInfo.pre)
    csv_file = '/home/shijinliang/module/tpds/ATT/end2end_ori_overhead/gat_final/result/pre.csv'
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    return round(inputInfo.pre,4)

if __name__ == "__main__":


    
    dataset_I = ['soc-Epinions1', 'GitHub', 'artist', 'blog']

    dataset_II = ['IGB_small', 'com-DBLP', 'amazon', 'amazon0505', 
                  'dd', 'comamazon', 'yeast']
    
    dataset_III = ['reddit', 'AmazonProducts']
    
    dataset1 = ['ogbn-arxiv', 'ogbn-proteins', 'ogb']

    dataset =  dataset1

    layer = [2]
    hidden = [64]
    head = [4]
    
    featuredim = 64
    classes = 10
    epoches = 10
    csv_file = '/home/shijinliang/module/tpds/ATT/end2end_ori_overhead/gat_final/result/pre.csv'
    # with open(csv_file, 'w', newline='') as file:
    #     file.write('data,config,gat16\n')
    # #MGCN-fp16
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in ['GitHub', 'blog', 'ogbn-arxiv']:
                    test(data, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')   
    print()         
