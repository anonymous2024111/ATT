import numpy
import torch
import ATT_Block
import ATT_SDDMM_v3
from scipy.sparse import *


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

row_pointers, column_index, degrees, window, automic =ATT_Block.blockProcess_sddmm_balance_gnn(row, col, 8, 16, 2)

# print(row_pointers)
# print(column_index)
# print(degrees)
# print(window)
# print(automic)
# print()


rows = 30
dimN = 125
head = 1
# rhs = torch.ones((head, rows, dimN), dtype=torch.float32)
rhs = torch.randint(low=1, high=3, size=(head, rows, dimN)).float()
rhs1 = torch.randint(low=1, high=3, size=(head, rows, dimN)).float()
weight = torch.ones((head, 1,dimN), dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rhs=rhs.to(device)
rhs1=rhs1.to(device)
weight=weight.to(device)
row_pointers=row_pointers.to(device)
column_index=column_index.to(device)
degrees=degrees.to(device)
window=window.to(device)

# wide = 16
# max_vectors = torch.max(row_pointers[1:]- row_pointers[:-1])
# if max_vectors%wide > 0 :
#     max_vectors += (wide - (max_vectors%wide))
# max = max_vectors / wide
wide = 16
max_vectors = torch.max(row_pointers[2::2]- row_pointers[:-2:2])
max = max_vectors / wide



print(max)
result = ATT_SDDMM_v3.attv2_tf32_h_nnz(
row_pointers, 
column_index,
degrees, 
window,
rhs, 
rhs1, 
weight, 
max,
row[-1].item())[0]

print(result)
        
if torch.dot(rhs[0][0],weight[0][0]) + torch.dot(rhs1[0][1],weight[0][0])!= result[0][0] :
    print("No")
    exit(0)

if head>1:
    if torch.dot(rhs[1][0],weight[1][0]) + torch.dot(rhs1[1][2],weight[1][0])!= result[1][1] :
        print("No")
        exit(0)
print("PASS")