import torch
from scipy.sparse import *
import sys
import csv
import pandas as pd
import time
import os

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(current_dir))
           
sys.path.append(project_dir + '/gat_final')

from mgat_csr_profile.mdataset_fp16 import *
from mgat_csr_profile.mgat_conv import *
from mgat_csr_profile.gat_mgnn import *

def test(data, epoches, heads, layers, featuredim, hidden, classes):
    # start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGAT_dataset_csr(data, featuredim, classes) 
    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes,0.5, 0.2, heads, layers).to(device)

    train(model, inputInfo, 1)
    print()
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


if __name__ == "__main__":
    dataset = 'blog'
    test(dataset, 1, 3, 2, 512, 128, 10)
    
   