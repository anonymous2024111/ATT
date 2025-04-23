import numpy
import torch
import Libra5Block
import Libra5BenchmarkGCN
from scipy.sparse import *
import ATT_Block
import ATT_SpMM_v4
def check(row_pointers1, column_index1, dd, rhs, n) :
    row_pointers1 = row_pointers1[:n+1]
    dd = dd.numpy()
    value = []
    for i in range(len(row_pointers1) - 1):
        for j in range(row_pointers1[i], row_pointers1[i+1]):
            value.append(dd[i]*dd[column_index1[j]])
    # n = row_pointers1.size(0)-1
    sparse_matrix = csr_matrix((value, column_index1.numpy(), row_pointers1.numpy()), shape=(n, n))
    result = sparse_matrix.dot(rhs.numpy())
    return result

row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)

head = 3
row_pointers, column_index, degrees, window, automic=ATT_Block.blockProcess_sddmm_balance_gnn(row, col, 8, 8, 32)
csr_values = torch.ones(col.size(0), head, dtype=torch.float16)
dd = torch.ones_like(col, dtype=torch.float16)
print(row_pointers)
print(column_index)
dimN = 2
rhs = torch.ones(head, 30, dimN)
rhs = rhs.half()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rhs1=rhs.to(device)
row_pointers=row_pointers.to(device)
column_index=column_index.to(device)
degrees=degrees.to(device)
window=window.to(device)
automic=automic.to(device)
csr_values=csr_values.to(device)
print()

# rhs = rhs.float()
result =  ATT_SpMM_v4.spmm_fp16_m_h_rowsum(   
                        row_pointers, 
                        degrees, 
                        csr_values,
                        window,
                        automic, rhs1.size(1))[0]

res = check(row,col,dd,rhs[0],30)
print(result.shape)
print(res)
print(result)

for i in range(30):
    if (result[i][0][0] - res[i][0]) != 0 :
            print("No")
            exit(0)

#第二头  
if head > 1:          
    res = check(row,col,dd,rhs[1],30)
    for i in range(30):
        if (result[i][1][0] - res[i][0]) != 0 :
                print("No")
                exit(0)
        
print("PASS")
    