import os.path as osp
import argparse
import time
import torch
import sys
sys.path.append('/home/shijinliang/module/tpds/ATT/end2end/gat_acc_final')
from mgat32_csr.mdataset_tf32 import *
from mgat32_csr.mgat_conv import *
from mgat32_csr.gat_mgnn import *


def test(data, epoches, heads, layers, featuredim, hidden, classes):
    # start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGAT_dataset_csr(data, featuredim, classes) 
    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes,layers,0.5).to(device)

    train(model, inputInfo, 10)
    torch.cuda.synchronize()
    start_time = time.time()  
    train(model, inputInfo, epoches)
    
    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算程序执行时间（按秒算）
    execution_time = end_time - start_time
    
    # print(execution_time)
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'cora'
#     test(dataset, 100, 8, 3, 512, 128, 10)