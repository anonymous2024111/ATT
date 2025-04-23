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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from mypyg import test_pyg
 
    
#MGPYG
def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  



if __name__ == "__main__":

    dataset_I = ['soc-Epinions1', 'GitHub', 'artist', 'blog']

    dataset_II = ['IGB_small', 'amazon', 'amazon0505', 
                  'dd', 'comamazon', 'yeast']
    
    dataset_III = ['reddit']
    
    dataset1 = ['ogbn-arxiv']

    dataset = dataset_I + dataset_II + dataset_III + dataset1
    
    epoches = 300
    # layer = [2, 4]
    # hidden = [64, 128, 256]
    layer = [3]
    hidden = [128]
    head = [1]
    
    
    
    featuredim = 256
    classes = 10

  
    
    #PYG
    filename = current_dir + '/result/pyg.csv'
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