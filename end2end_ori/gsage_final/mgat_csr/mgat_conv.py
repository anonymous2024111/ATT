#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
from torch.nn.parameter import Parameter
# from tqdm.std import tqdm
import SpMM_sr_bcrs
n_heads = 8
n_output = 8

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new



class MGCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_prime, inputInfo):
        ctx.inputInfo = inputInfo

        X_prime  = SpMM_sr_bcrs.forward_fp16_gnn(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            inputInfo.t_atomicTensor,
            X_prime, 
            inputInfo.num_nodes, 
            X_prime.size(1), 
            inputInfo.num_nodes_ori)[0]

        return X_prime.half()

    @staticmethod
    def backward(ctx, d_output):
        inputInfo = ctx.inputInfo
        # SPMM backward propaAGNNion.
        d_input_prime  = SpMM_sr_bcrs.forward_fp16_gnn(   
                    inputInfo.row_pointers, 
                    inputInfo.column_index, 
                    inputInfo.degrees, 
                    inputInfo.t_window_rowTensor,
                    inputInfo.t_atomicTensor,
                    d_output, 
                    inputInfo.num_nodes, 
                    d_output.size(1), 
                    inputInfo.num_nodes_ori)[0]

        return d_input_prime.half(), None


class dropout_gat:
    def __init__(self) :
        # 构造函数，用于初始化对象的属性
        self.ones = torch.ones(10, 2, dtype=torch.float16)
###################################
# Definition of each conv layers
###################################

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        #self.weights = Parameter(torch.ones(input_dim, output_dim))
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weights1 = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        # if self.weights is not None:
        #     nn.init.xavier_uniform_(self.weights)
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.weights1.data.uniform_(-stdv, stdv)


    def forward(self, X, inputInfo):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        #agg
        X_prime = torch.mm(X, self.weights.half())
        X_prime = MGCNFunction.apply(X_prime, inputInfo)
        X_prime = X_prime.div(inputInfo.dd1)
        
        #update
        X_prime_1 = torch.mm(X, self.weights1.half())
        
        #concat
        # X_prime_c = torch.cat([X_prime, X], dim=1)
        X_prime_out = X_prime + X_prime_1
        
        return X_prime_out
