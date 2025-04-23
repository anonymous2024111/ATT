#!/usr/bin/env python3
import torch
import numpy as np
import time
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse import *
import ATT_Block


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
        self.num_features = self.graph['in_size'].item()
        self.num_classes = self.graph['out_size'].item()

        self.init_edges()
        self.init_embedding()
        self.init_labels()
        self.init_others()

        # self.train_mask = torch.from_numpy(self.graph['train_mask'])
        # self.val_mask = torch.from_numpy(self.graph['val_mask'])
        # self.test_mask = torch.from_numpy(self.graph['test_mask'])
        


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
        col= torch.IntTensor(adj.indices)
        row = torch.IntTensor(adj.indptr)

        partize = 32
        self.row_pointers, \
        self.column_index, \
        self.degrees, \
        self.degrees_trans, \
        self.window, \
        self.automic = ATT_Block.blockProcess_sddmm_balance_gnn_trans(row, col, 8, 4, partize)
        self.max_vectors = torch.max(self.row_pointers[2::2]- self.row_pointers[:-2:2])
        self.max = self.max_vectors / 16
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_ori, self.num_features).to(dtype=torch.float32)

    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes_ori,))
        
    def init_others(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float32)

    def to(self, device):
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)
        self.degrees =  self.degrees.to(device)
        self.degrees_trans =  self.degrees_trans.to(device)
        self.window =  self.window.to(device)
        self.automic = self.automic.to(device)

        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
