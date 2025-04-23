from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='TMM_sr_bcrs',
    # ext_modules=[module],
    ext_modules=[
        CUDAExtension(
            name='Block_sr_bcrs', 
            sources=[
            './Block_gcn/example.cpp'
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
         ) ,
        CUDAExtension(
            name='SpMM_sr_bcrs', 
            sources=[
            './SpMM/src/benchmark.cpp',
            './SpMM/src/spmmKernel.cu',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


