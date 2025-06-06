import numpy as np
import torch
import ATT_Block
row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
value=col.float()
value1=col.half()
rowTensor, colTensor, valueTensor, window, atomic= ATT_Block.blockProcess_sddmm_balance_gnn(row,col,8,8,2)

print(rowTensor)
print(colTensor)
print(valueTensor)
print(window)
print(atomic)
