from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='TMM_v4',
    # ext_modules=[module],
    ext_modules=[
       CUDAExtension(
            name='ATT_SpMM_v4', 
            sources=[
            './spmm-benchmark/src/benchmark.cpp',
            './spmm-benchmark/src/spmmKernel_fp16.cu',
            './spmm-benchmark/src/spmmKernel_tf32.cu',
            ],
         ),
       CUDAExtension(
            name='ATT_SDDMM_v4', 
            sources=[
            './sddmm-benchmark/src/benchmark.cpp',
            './sddmm-benchmark/src/sddmmKernel_tf32.cu',
            './sddmm-benchmark/src/sddmmKernel_fp16.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


