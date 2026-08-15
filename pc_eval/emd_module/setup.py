from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name='emd',
    ext_modules=[
        CUDAExtension('emd', [
            str(ROOT / 'emd.cpp'),
            str(ROOT / 'emd_cuda.cu'),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
