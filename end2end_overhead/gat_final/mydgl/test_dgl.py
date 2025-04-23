import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sys
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end_ori_overhead/gat_final')
from mydgl.mdataset import *
from mydgl.gat_dgl import GAT, train, evaluate
import time
    
def test(data, epoches, heads, layers, featuredim, hidden, classes):
    # start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data, featuredim, classes)
    
    torch.cuda.synchronize()
    start_time = time.time()
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)
    inputInfo.to(device)
    g = g.int().to(device)
    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time    
    print('preprocess: ' + str(execution_time))
    
    model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes, heads, layers).to(device)
    
    train(g, inputInfo.x, inputInfo.y, model, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(g, inputInfo.x, inputInfo.y, model, epoches)

    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算程序执行时间（按秒算）
    execution_time = end_time - start_time
    # print(execution_time)
    return round(execution_time,4)

if __name__ == "__main__":


    
    dataset1 = ['soc-Epinions1', 'ogbn-arxiv', 'ogbn-proteins', 'ogb']

    layer = [2]
    hidden = [64]
    head = [4]
    
    featuredim = 64
    classes = 10
    epoches = 10

    # #MGCN-fp16
    for head_num in head:
        for layer_num in layer:
            for hidden_num in hidden:
                for data in dataset1:
                    test(data, epoches, head_num, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')   
    print()    
