from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='focal_loss',
    ext_modules=[
        CUDAExtension('focal_loss', [
            'focal_loss_kernel.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
