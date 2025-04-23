import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from mgat_csr.mgat_conv import *
from torch.optim import Adam


#########################################
## Build GAT Model
#########################################

class Net(torch.nn.Module):
    def __init__(self,in_feats, hidden_feats, out_feats, dropout, alpha, heads, num_layers):
        super(Net, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv_multi(in_feats, hidden_feats, dropout, alpha)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATConv_multi(hidden_feats, hidden_feats, dropout, alpha))
        
        self.conv2 = GATConv_multi(hidden_feats, out_feats, dropout, alpha)

    def forward(self, inputInfo):
        x = F.elu(self.conv1(inputInfo.x, inputInfo).flatten(1))
        for Gconv in self.hidden_layers:
            x = F.relu(Gconv(x, inputInfo).flatten(1))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, inputInfo).flatten(1)
        res = F.log_softmax(x, dim=1)
        return res


def test1(model, inputInfo):
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
    
    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo)
        # if torch.isnan(logits).any().item() :
        loss =  F.nll_loss(logits, inputInfo.y)
    
        acc = test1(model, inputInfo)
        acc = round(acc*100, 2)
        print(str(epoch) + ' MGAT '": test_accuracy {:.2f}".format(acc))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()