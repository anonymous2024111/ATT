import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sys
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre')
from mydgl.mdataset import *
from mydgl.gat_dgl import GAT, train, evaluate
import time
    
def test(data, epoches, heads, hidden, layers):
    # start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data)
    # print(inputInfo.num_nodes)
    # print(inputInfo.num_edges)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)
    inputInfo.to(device)
    g = g.int().to(device)
    model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes, heads, layers).to(device)
   
    mean_value, std_dev = train(g, inputInfo.x, inputInfo.y, model, epoches, inputInfo.test_mask)
    acc = evaluate(g, inputInfo.x, inputInfo.y, inputInfo.test_mask, model)
    acc = round(acc*100, 2)
    print(str(data) + ' DGL '": test_accuracy {:.2f}".format(acc))

    
    return mean_value, std_dev
