import numpy as np
import argparse
import torch
import sys
# sys.path.append('Eva/end2end/gat')
#sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gat_no_pre_multi')
from mypyg.gat_pyg import GAT, train, evaluate
from mypyg.mdataset import *
import time
def test(data, epoches, heads, hidden, layers):
        # start_time = time.time()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputInfo = MGCN_dataset(data)
        inputInfo.to(device)
        model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes, heads, layers).to(device)
        mean_value, std_dev =  train(inputInfo.edge_index, inputInfo.x, inputInfo.y, model, epoches,inputInfo.test_mask)
        acc = evaluate(inputInfo.edge_index, inputInfo.x, inputInfo.y, inputInfo.test_mask, model)
        acc = round(acc*100, 2)
        print(str(data) + ' PyG '": test_accuracy {:.2f}".format(acc))
    
        return mean_value, std_dev