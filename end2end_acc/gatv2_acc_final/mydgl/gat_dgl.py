import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import numpy as np 

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, num_layers):
        super().__init__()
        self.dropout = 0.5
        self.conv1 = GATConv(in_size, hid_size, heads)   

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATConv(hid_size*heads, hid_size, heads))
                
        self.conv2 = GATConv(hid_size*heads, out_size, 1)

    def forward(self, g, features):
        h=F.relu(self.conv1(g,features).flatten(1))
        for Gconv in self.hidden_layers:
            h = F.relu(Gconv(g,h).flatten(1))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(g,h).mean(1)
        h = F.log_softmax(h,dim=1)
        return h

#输入依次为图，结点特征，标签，验证集或测试集的mask，模型
#注意根据代码逻辑，图和结点特征和标签应该输入所有结点的数据，而不能只输入验证集的数据
def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        
        logits = logits[mask]
        labels = labels[mask]
        #probabilities = F.softmax(logits, dim=1) 
        #print(probabilities)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
    
def train(g, features, labels, model,epoches, test_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    acc_list = []
    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(g, features)
        loss = F.nll_loss(logits, labels)
        
        acc = evaluate(g, features, labels, test_mask, model)
        acc = round(acc*100, 2)
        if epoch>200:
            acc_list.append(acc)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    # 计算均值
    mean_value = np.mean(acc_list)
    # 计算标准差
    std_dev = np.std(acc_list, ddof=1)  # 使用 n-1 自由度
    return mean_value, std_dev
