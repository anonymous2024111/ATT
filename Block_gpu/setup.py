from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='ATT_Block_gpu',
    # ext_modules=[module],
    ext_modules=[
        CUDAExtension(
            name='ATT_Block_gpu', 
            sources=[
            './DTCSpMM.cpp',
            './DTCSpMM_kernel.cu'
            ],
         ) ,
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


