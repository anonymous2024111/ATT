import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_size, hid_size, aggr='mean')
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(SAGEConv(hid_size, hid_size, aggr='mean'))
        
        self.conv2 = SAGEConv(hid_size, out_size, aggr='mean')
        self.dropout = dropout

    def forward(self, edge, features):
        h = features
        h = F.relu(self.conv1(h, edge))
        h = F.dropout(h, self.dropout, training=self.training)
        for layer in self.hidden_layers:
            h = F.relu(layer(h,edge))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(h,edge)
        h = F.log_softmax(h,dim=1)
        return h

def train(edge, features, labels, model,epoches):
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(edge, features)
        loss = F.nll_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
