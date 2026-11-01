"""Shared setup for the small command-line entrypoints."""
from pathlib import Path


def add_common_arguments(parser):
    parser.add_argument('--config', default='configs/bevfusion.yaml')
    parser.add_argument('--data-path', help='nuScenes parent directory, containing v1.0-*')
    parser.add_argument('--version', choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'])
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--output', help='default: output/<config-name>')


def load_config(args):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file, project_path
    cfg = cfg_from_yaml_file(args.config, EasyDict())
    if args.data_path:
        cfg.DATA_CONFIG.DATA_PATH = str(Path(args.data_path).expanduser().resolve())
    if args.version:
        cfg.DATA_CONFIG.VERSION = args.version
    if cfg.DATA_CONFIG.VERSION == 'v1.0-test':
        cfg.DATA_CONFIG.INFO_PATH.test = [f'nuscenes_infos_{cfg.DATA_CONFIG.MAX_SWEEPS}sweeps_test.pkl']
    for processor in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if processor.NAME == 'shuffle_points':
            processor.SHUFFLE_ENABLED.test = False
    output = Path(args.output).resolve() if args.output else project_path(f'output/{Path(args.config).stem}')
    output.mkdir(parents=True, exist_ok=True)
    return cfg, output


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This detector requires NVIDIA CUDA and compiled pcdet operators; see README.md.')


def build_loader(cfg, args, logger, training):
    from pcdet.datasets import build_dataloader
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size or (cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU if training else 1),
        dist=False, workers=args.workers, logger=logger, training=training, seed=args.seed,
    )
    if not len(dataset) or not len(loader):
        raise RuntimeError('No samples loaded. Check --data-path, --version and generated info files.')
    return dataset, loader


def snapshot_config(cfg, path):
    import yaml
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(val) for key, val in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(val) for val in value]
        return str(value) if isinstance(value, Path) else value
    Path(path).write_text(yaml.safe_dump(plain(cfg), sort_keys=False), encoding='utf-8')


def freeze_parameters(model, mode):
    if mode == 'all':
        return []
    if mode == 'fusion':
        if getattr(model, 'fuser', None) is None:
            raise ValueError('--trainable fusion requires BEVFusion')
        frozen = [name for name, module in model.named_children()
                  if name != 'fuser' and module is not None]
    else:
        frozen = [name for name in ['backbone_3d', 'image_backbone']
                  if getattr(model, name, None) is not None]
    for name in frozen:
        # 按名称获取模型中的模块
        # 原地关闭该模块所有参数的梯度计算
        getattr(model, name).requires_grad_(False)
    return frozen


def train_mode(model, frozen):
    import torch.nn as nn
    model.train()
    # Keep head.training=True for target assignment and loss, but fix the
    # stochastic layers and running statistics of frozen modules.
    types = (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
    for name in frozen:
        for module in getattr(model, name).modules():
            if isinstance(module, types) or module.__class__.__name__ == 'DropPath':
                module.eval()


def disable_image_pretrain(cfg):
    if 'IMAGE_BACKBONE' in cfg.MODEL:
        cfg.MODEL.IMAGE_BACKBONE.INIT_CFG = None
