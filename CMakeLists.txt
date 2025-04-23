cmake_minimum_required(VERSION 3.13)


project(SPUTNIK CUDA CXX C)
add_compile_options(-std=c++17)
find_package(PythonInterp REQUIRED)

# 设置 C++ 标准
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA)
find_package(OpenMP REQUIRED)
set(Torch_DIR /home/shijinliang/.conda/envs/sjlpy39pro/lib/python3.9/site-packages/torch/share/cmake/Torch)  
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CUDA_GENCODE_FLAGS "-gencode arch=compute_90,code=sm_90")
message("CUDA_GENCODE_FLAGS:${CUDA_GENCODE_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_GENCODE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")


# add_subdirectory(benchmark/GCN-benchmark)
# add_subdirectory(benchmark/GAT-benchmark)
# add_subdirectory(Block)
add_subdirectory(benchmark/GATv2-benchmark)
add_subdirectory(benchmark/GCNv2-benchmark)
