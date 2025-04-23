import numpy as np
import argparse
import torch
import sys
# sys.path.append('Eva/end2end/gat')
#sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi')
from mypyg.gat_pyg import GAT, train
from mypyg.mdataset import *
import time
def test(data, epoches, heads, layers, featuredim, hidden, classes):
        # start_time = time.time()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputInfo = MGCN_dataset(data,featuredim,classes)
        inputInfo.to(device)
        model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes, heads, layers).to(device)
        # train(inputInfo.edge_index, inputInfo.x, inputInfo.y, model, 10)
        torch.cuda.synchronize()
        start_time = time.time() 
        train(inputInfo.edge_index, inputInfo.x, inputInfo.y, model, epoches)
        # 记录程序结束时间
        torch.cuda.synchronize() 
        end_time = time.time()
        # 计算程序执行时间（按秒算）
        execution_time = end_time - start_time
        return round(execution_time,4)
# dataset = 'artist'
# test(dataset,10,2,3,512,128,6)