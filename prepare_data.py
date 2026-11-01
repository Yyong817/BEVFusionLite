"""Create nuScenes infos with camera calibration and an optional GT database."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True, help='parent directory containing v1.0-*')
    parser.add_argument('--version', choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'], default='v1.0-trainval')
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--gt-database', action='store_true', help='required by the LiDAR training configs')
    args = parser.parse_args()
    if args.sweeps < 1 or (args.gt_database and args.version == 'v1.0-test'):
        parser.error('sweeps must be positive; test split cannot generate a GT database')

    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset, create_nuscenes_info
    from pcdet.utils import common_utils

    root = Path(args.data_path).expanduser().resolve()
    if not (root / args.version).is_dir():
        parser.error(f'Missing {root / args.version}; see README.md for dataset layout')
    create_nuscenes_info(args.version, root, root, max_sweeps=args.sweeps, with_cam=True)
    if args.gt_database:
        cfg = cfg_from_yaml_file('configs/centerpoint.yaml', EasyDict())
        cfg.DATA_CONFIG.VERSION = args.version
        cfg.DATA_CONFIG.MAX_SWEEPS = args.sweeps
        cfg.DATA_CONFIG.INFO_PATH.train = [f'nuscenes_infos_{args.sweeps}sweeps_train.pkl']
        cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
        # Database creation must not try to load the database being created.
        cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST = []
        dataset = NuScenesDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, root_path=root,
                                  logger=common_utils.create_logger(), training=True)
        dataset.create_groundtruth_database(max_sweeps=args.sweeps)


if __name__ == '__main__':
    main()
