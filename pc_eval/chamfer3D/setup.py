from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            str(ROOT / 'chamfer_cuda.cpp'),
            str(ROOT / 'chamfer3D.cu'),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
