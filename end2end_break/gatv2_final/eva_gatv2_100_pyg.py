import torch
import numpy as np
import time
import csv
import sys
sys.path.append('./eva100/end2end_ori/gat_no_pre')
import os

current_dir = os.path.dirname(__file__)
# project_dir = os.path.dirname(os.path.dirname(current_dir))
print(current_dir)           
sys.path.append(current_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mypyg import test_pyg
 
    
#MGPYG
def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    with open("/home/shijinliang/module/tpds/ATT/end2end_break/gatv2_final/result/pyg.csv", "a") as f:
        f.write(data + ',')  # 追加换行符    
    spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    with open("/home/shijinliang/module/tpds/ATT/end2end_break/gatv2_final/result/pyg.csv", "a") as f:
        f.write("\n")  # 追加换行符


if __name__ == "__main__":


    dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                 'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                  'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                  'roadNet-TX', 'yeast']
    dataset_III = ['reddit', 'ogb', 'AmazonProducts']

    dataset = dataset_I + dataset_II + dataset_III
    # dataset = ['amazon']
    dataset1 = ['ogbn-arxiv', 'ogbn-proteins']
    dataset_II_pyg =['ell',  'com-DBLP', 'amazon0505', 
                  'dd',  'comamazon', 'roadNet-PA']
    dataset_pyg = dataset_I + dataset_II
    epoches = 300
    dataset =  ['GitHub', 'blog', 'ogbn-arxiv']
    epoches = 1
    layer = [3]
    hidden = [64]
    head = [1]
    
    
    
    featuredim = 256
    classes = 10

  
    #warmup
    filename = current_dir + '/result/warmup.csv'
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in ['soc-Epinions1']:
                    if data == 'Reddit2' or data == 'yelp':
                        continue
                    pygGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('Pyg-' + 'success')
    
    #PYG
    filename = current_dir + '/result/pyg-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset:
                    if data == 'Reddit2' or data == 'yelp':
                        continue
                    pygGCN(data, filename, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('Pyg-' + 'success')

    # print('MGAT_small_all success')