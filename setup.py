from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='ATT_kernel',
    # ext_modules=[module],
    ext_modules=[
        CUDAExtension(
            name='ATT_SpMM', 
            sources=[
            './benchmark/GCN-benchmark/src/benchmark.cpp'
            ],
            library_dirs=['./benchmark/GCN-benchmark/lib/libMGCN1.a'], 
            extra_objects=['./benchmark/GCN-benchmark/lib/libMGCN1.a'],
            extra_compile_args=['-O3']
         ),
        CUDAExtension(
        name='ATT_SpMMv2', 
        sources=[
        './benchmark/GCNv2-benchmark/src/benchmark.cpp'
        ],
        library_dirs=['./benchmark/GCNv2-benchmark/lib/libMGCN2.a'], 
        extra_objects=['./benchmark/GCNv2-benchmark/lib/libMGCN2.a'],
        extra_compile_args=['-O3']
        ),
        CUDAExtension(
            name='ATT_SDDMM', 
            sources=[
            './benchmark/GAT-benchmark/src/benchmark.cpp'
            ],
            library_dirs=['./benchmark/GAT-benchmark/lib/libMGAT1.a'], 
            extra_objects=['./benchmark/GAT-benchmark/lib/libMGAT1.a'],
            extra_compile_args=['-O3']
         ),
        CUDAExtension(
            name='ATT_SDDMMv2', 
            sources=[
            './benchmark/GATv2-benchmark/src/benchmark.cpp'
            ],
            library_dirs=['./benchmark/GATv2-benchmark/lib/libMGAT2.a'], 
            extra_objects=['./benchmark/GATv2-benchmark/lib/libMGAT2.a'],
            extra_compile_args=['-O3']
         ),
        CUDAExtension(
            name='ATT_Block', 
            sources=[
            './Block/example.cpp'
            ],
            library_dirs=['./Block/lib/libMBlock.a'], 
            extra_objects=['./Block/lib/libMBlock.a'],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
         ) ,
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


