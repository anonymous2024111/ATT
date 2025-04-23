#!/usr/bin/env python3
import torch
import numpy as np
import time
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse import *
import Block_sr_bcrs


def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    
    # 计算原矩阵和转置矩阵的差
    difference = sparse_matrix - transposed_matrix
    
    # 如果差异矩阵的非零元素数量为 0，则说明矩阵对称
    return difference.nnz == 0

class MGAT_dataset_csr(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(MGAT_dataset_csr, self).__init__()
        self.graph = np.load('/public/home/shijinliang/gnns/' + data +'.npz')
        if data in ['IGB_small', 'IGB_medium']:
            self.num_features = 128
            self.num_classes = 16
        else:
            self.num_features = 128
            self.num_classes = 16

        self.init_edges()
        self.init_embedding()
        self.init_labels()

    def init_edges(self):
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+8-(self.graph['num_nodes']%8)
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        # 计算矩阵的转置
        #sysm = is_symmetric(adj)
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        dd = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        dd=torch.tensor(dd, dtype=torch.float32) 
        dd= torch.rsqrt(dd).to(torch.float16)  

        partize = 32
        
        self.row_pointers, \
        self.column_index, \
        self.degrees, \
        self.t_window_rowTensor, \
        self.t_atomicTensor = Block_sr_bcrs.blockProcess_fp16_balance(self.row_pointers, self.column_index, dd, 8, 8, 32)
    
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_ori, self.num_features).to(dtype=torch.float16)

    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes_ori,))


    def to(self, device):
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)
        self.degrees =  self.degrees.to(device)
        self.t_window_rowTensor =  self.t_window_rowTensor.to(device)
        self.t_atomicTensor = self.t_atomicTensor.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self
