#include <torch/extension.h>


std::vector<torch::Tensor> blockProcess_sddmm_gnn(torch::Tensor row1, torch::Tensor column1, int window1, int wide1, int partSize_t);

std::vector<torch::Tensor> blockProcess_sddmm_gnn_trans(torch::Tensor row1, torch::Tensor column1, int window1, int wide1, int partSize_t);