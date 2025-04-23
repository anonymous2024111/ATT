#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.std import tqdm
import ATT_SpMM_v3
import ATT_SDDMM_v3
import numpy as np


class MGATFunction_multi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_prime0, X_prime1, a0, inputInfo):
        ctx.inputInfo = inputInfo    
        ctx.X_prime0 = X_prime0
        ctx.X_prime1 = X_prime1
        ctx.a0 = a0

        att = ATT_SDDMM_v3.attv2_fp16_h_nnz(
            inputInfo.row_pointers, 
            inputInfo.column_index,
            inputInfo.degrees, 
            inputInfo.window,
            X_prime0, 
            X_prime1,
            a0, 
            inputInfo.max,
            inputInfo.num_edges)[0]
        return att

    @staticmethod
    def backward(ctx, att_grad):
        inputInfo = ctx.inputInfo
        X_prime0 = ctx.X_prime0
        X_prime1 = ctx.X_prime1
        a0 = ctx.a0

        '''
        求a0梯度
        '''
        a0_right = ATT_SpMM_v3.spmm_fp16_h_m_rowsum(
            inputInfo.row_pointers,
            inputInfo.degrees_trans, 
            att_grad,
            inputInfo.window,
            inputInfo.automic,
            inputInfo.num_nodes_ori)[0].half()
        a0_grad_right = torch.bmm(a0_right.transpose(1,2), X_prime1)

        a0_left = ATT_SpMM_v3.spmm_fp16_h_m_rowsum(
            inputInfo.row_pointers,
            inputInfo.degrees, 
            att_grad,
            inputInfo.window,
            inputInfo.automic,
            inputInfo.num_nodes_ori)[0].half()
        a0_grad_left = torch.bmm(a0_left.transpose(1,2), X_prime0)
        
        # 求X_prime_grad
        X_prime_grad_right = torch.bmm(a0_right, a0) 
        X_prime_grad_left = torch.bmm(a0_left, a0)

        return X_prime_grad_left, X_prime_grad_right, a0_grad_left+a0_grad_right, None


class MGATSpmm_multi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        ctx.att = att
        ctx.X_prime = X_prime
        ctx.inputInfo = inputInfo

        X_prime= ATT_SpMM_v3.spmm_fp16_m_h(
            inputInfo.row_pointers,
            inputInfo.column_index, 
            inputInfo.degrees, 
            att,
            inputInfo.window,
            inputInfo.automic,
            X_prime)[0]

        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
        X_prime = ctx.X_prime
        inputInfo = ctx.inputInfo
        att = ctx.att
        X_prime_grad = X_prime_grad.half()
        
        #根据X_prime，通过SpMM反向传播求d_X_prime梯度
        X_prime_grad = X_prime_grad.permute(1, 0, 2).contiguous()
        d_X_prime= ATT_SpMM_v3.spmm_fp16_h_m(
            inputInfo.row_pointers,
            inputInfo.column_index, 
            inputInfo.degrees_trans, 
            att,
            inputInfo.window,
            inputInfo.automic,
            X_prime_grad)[0].half()
        
        #根据X_prime，通过SDDMM反向传播求att_grad梯度
        d_att = ATT_SDDMM_v3.sddmm_fp16_h_nnz(
            inputInfo.row_pointers, 
            inputInfo.column_index,
            inputInfo.degrees, 
            inputInfo.window,
            X_prime_grad,
            X_prime, 
            inputInfo.max,
            inputInfo.num_edges)[0]
        
        return d_att, d_X_prime, None

class MGATSpmm1_multi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, inputInfo):
        ctx.inputInfo = inputInfo

        X_prime= ATT_SpMM_v3.spmm_fp16_m_h_rowsum(
            inputInfo.row_pointers,
            inputInfo.degrees, 
            att,
            inputInfo.window,
            inputInfo.automic,
            inputInfo.num_nodes_ori)[0]
        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
        
        inputInfo = ctx.inputInfo
        X_prime_grad = X_prime_grad.half()
        #根据X_prime，通过SDDMM反向传播求att_grad梯度
 
        d_att = ATT_SDDMM_v3.sddmm_fp16_ones_h_nnz(
            inputInfo.row_pointers, 
            inputInfo.column_index,
            inputInfo.degrees, 
            inputInfo.window,
            X_prime_grad,
            inputInfo.max,
            inputInfo.num_edges)[0]
        
        return d_att, None

    
class GATv2Conv_multi(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, head):
        super(GATv2Conv_multi, self).__init__()
        self.alpha = alpha
        gain1 = nn.init.calculate_gain("relu")
        
        # 使用 nn.Linear 替代原始权重定义
        self.fc0 = nn.Linear(input_dim, output_dim * head, bias=False)
        nn.init.xavier_normal_(self.fc0.weight.data, gain=gain1)  # Xavier 初始化
        self.fc1 = nn.Linear(input_dim, output_dim * head, bias=False)
        nn.init.xavier_normal_(self.fc1.weight.data, gain=gain1)  # Xavier 初始化
        
        self.a0 = torch.nn.Parameter(torch.zeros(size=(head, 1, output_dim)))
        nn.init.xavier_normal_(self.a0.data, gain=gain1)

        self.output_dim = output_dim 
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.head = head
        
    def forward(self, X, inputInfo):
        #特征降维
        X_prime0 = self.fc0(X).half()
        X_prime1 = self.fc1(X).half()
        X_prime0 = X_prime0.view(-1, self.head, self.output_dim).permute(1, 0, 2).contiguous() 
        X_prime1 = X_prime1.view(-1, self.head, self.output_dim).permute(1, 0, 2).contiguous() 
        X_prime0 = self.leakyrelu(X_prime0)
        X_prime1 = self.leakyrelu(X_prime1)
        
        #求att和用于反向传播的att_trans
        att = MGATFunction_multi.apply(X_prime0, X_prime1, self.a0.half(), inputInfo)

        #exp
        max_value = torch.max(att, dim=1, keepdim=True)[0]  # 按行（每个头）计算最大值
        min_value = torch.min(att, dim=1, keepdim=True)[0]  # 按行（每个头）计算最小值
        att = (att - min_value) / (max_value - min_value)

        att = torch.exp(att)
        rows_sum = MGATSpmm1_multi.apply(att, inputInfo)
        #dropout
        att = self.dropout(att)
        
        #特征更新
        h_prime = MGATSpmm_multi.apply(att, X_prime1, inputInfo)

        h_prime = h_prime.div(rows_sum)

        return h_prime


