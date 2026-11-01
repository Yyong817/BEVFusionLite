"""Report dependencies without requiring them just to invoke the command."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
required = ['torch', 'torchvision', 'spconv', 'numpy', 'scipy', 'numba', 'yaml',
            'easydict', 'tqdm', 'tensorboardX', 'skimage', 'PIL', 'cv2', 'pyquaternion', 'nuscenes']
missing = []
for name in required:
    present = importlib.util.find_spec(name) is not None
    print(f'{name}: {"found" if present else "MISSING"}')
    if not present:
        missing.append(name)
if 'torch' not in missing:
    import torch
    print(f'PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}; GPU: {torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        missing.append('CUDA GPU')
if not missing:
    import importlib
    for name in ['pcdet.ops.iou3d_nms.iou3d_nms_cuda',
                 'pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda', 'pcdet.ops.bev_pool.bev_pool_ext']:
        try:
            importlib.import_module(name)
            print(f'{name}: OK')
        except (ImportError, OSError) as error:
            print(f'{name}: {error}')
            missing.append(name)
sys.exit(bool(missing))
