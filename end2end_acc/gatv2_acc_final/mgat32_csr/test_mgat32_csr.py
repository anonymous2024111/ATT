import os.path as osp
import argparse
import time
import torch
import sys
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gatv2_acc_final')
from mgat32_csr.mdataset_tf32 import *
from mgat32_csr.mgat_conv import *
from mgat32_csr.gat_mgnn import *


def test(data, epoches, heads, hidden, layers):
    # start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGAT_dataset_csr(data) 
    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes,0.5, 0.2, heads, layers).to(device)

    mean_value, std_dev = train(model, inputInfo, epoches)
    acc = evaluate(model, inputInfo)
    acc = round(acc*100, 2)
    print(str(data) + ' MGAT '": test_accuracy {:.2f}".format(acc))
    
    
    # print(execution_time)
    return mean_value, std_dev