import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from mgat32_csr.mgat_conv import *
from torch.optim import Adam


#########################################
## Build GAT-v2 Model
#########################################
class Net(torch.nn.Module):
    def __init__(self,in_feats, hidden_feats, out_feats, dropout, alpha, heads, num_layers):
        super(Net, self).__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv_multi(in_feats, hidden_feats, dropout, alpha, heads)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATv2Conv_multi(hidden_feats*heads, hidden_feats, dropout, alpha, heads))
        
        self.conv2 = GATv2Conv_multi(hidden_feats*heads, out_feats, dropout, alpha, 1)

    def forward(self, inputInfo):
        x = F.elu(self.conv1(inputInfo.x, inputInfo).flatten(1))
        for Gconv in self.hidden_layers:
            x = F.relu(Gconv(x, inputInfo).flatten(1))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, inputInfo).flatten(1)
        res = F.log_softmax(x, dim=1)
        return res



def evaluate(model, inputInfo):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits[inputInfo.test_mask]
        labels = inputInfo.y[inputInfo.test_mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
    
# Training 
def train(model, inputInfo, epoches):
    # loss_fcn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)  
    acc_list = []    
    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo)
        # if torch.isnan(logits).any().item() :
        loss =  F.nll_loss(logits, inputInfo.y)

        acc = evaluate(model, inputInfo)
        acc = round(acc*100, 2)
        if epoch>200:
            acc_list.append(acc)
        print(str(epoch) + ' MGAT '": test_accuracy {:.2f}".format(acc))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 计算均值
    mean_value = np.mean(acc_list)
    # 计算标准差
    std_dev = np.std(acc_list, ddof=1)  # 使用 n-1 自由度
    return mean_value, std_dev