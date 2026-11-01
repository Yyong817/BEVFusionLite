"""Build only the three CUDA extensions used by the selected detectors."""
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent


def extension(module, name, filenames):
    return CUDAExtension(f'pcdet.ops.{module}.{name}',
                         [str(ROOT / 'pcdet' / 'ops' / module / 'src' / name) for name in filenames])


setup(
    name='bevfusion-lite', version='0.1.0', python_requires='>=3.9',
    description='Focused nuScenes CenterPoint / BEVFusion research project extracted from OpenPCDet',
    license='Apache-2.0', packages=find_packages(include=['pcdet', 'pcdet.*', 'tools', 'tools.*']),
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        extension('iou3d_nms', 'iou3d_nms_cuda',
                  ['iou3d_cpu.cpp', 'iou3d_nms_api.cpp', 'iou3d_nms.cpp', 'iou3d_nms_kernel.cu']),
        extension('roiaware_pool3d', 'roiaware_pool3d_cuda',
                  ['roiaware_pool3d.cpp', 'roiaware_pool3d_kernel.cu']),
        extension('bev_pool', 'bev_pool_ext', ['bev_pool.cpp', 'bev_pool_cuda.cu']),
    ],
)
