from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='TMM_block_v3',
    # ext_modules=[module],
    ext_modules=[
        CUDAExtension(
            name='ATT_Block_v3', 
            sources=[
            './example1.cpp'
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
         ) ,
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


