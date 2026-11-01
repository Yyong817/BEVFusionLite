"""Behavior checks for config inheritance, neutral gates, and frozen heads."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_CONFIG = importlib.util.find_spec('yaml') is not None
HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ['torch', 'numpy', 'easydict'])


@unittest.skipUnless(HAS_CONFIG, 'requires PyYAML')
class ConfigTest(unittest.TestCase):
    def test_gated_keeps_baseline_settings(self):
        from pcdet.config import load_yaml
        baseline = load_yaml('configs/bevfusion.yaml')
        gated = load_yaml('configs/bevfusion_gated.yaml')
        self.assertEqual(gated['MODEL']['FUSER']['NAME'], 'GatedFuser')
        self.assertEqual(gated['MODEL']['FUSER']['IN_CHANNEL'], 336)
        self.assertEqual(gated['MODEL']['FUSER']['OUT_CHANNEL'], 256)
        self.assertEqual(gated['MODEL']['DENSE_HEAD'], baseline['MODEL']['DENSE_HEAD'])
        self.assertEqual(gated['DATA_CONFIG'], baseline['DATA_CONFIG'])
        self.assertEqual(gated['OPTIMIZATION'], baseline['OPTIMIZATION'])

    def test_lidar_configs(self):
        from pcdet.config import load_yaml, project_path
        for name, head in [('centerpoint', 'CenterHead'), ('transfusion_lidar', 'TransFusionHead')]:
            cfg = load_yaml(f'configs/{name}.yaml')
            self.assertEqual(cfg['MODEL']['DENSE_HEAD']['NAME'], head)
            self.assertTrue(project_path(cfg['DATA_CONFIG']['DATA_PATH']).is_absolute())

    def test_cycle_is_rejected(self):
        from pcdet.config import ROOT, _expand
        path = ROOT / 'configs/bevfusion.yaml'
        with self.assertRaises(ValueError):
            _expand({'_BASE_CONFIG_': str(path)}, (path,))

    def test_configured_modules_and_transforms_are_retained(self):
        import ast
        from pcdet.config import load_yaml
        locations = {
            'VFE': 'backbones_3d/vfe', 'BACKBONE_3D': 'backbones_3d',
            'MAP_TO_BEV': 'backbones_2d/map_to_bev', 'IMAGE_BACKBONE': 'backbones_image',
            'NECK': 'backbones_image/img_neck', 'VTRANSFORM': 'view_transforms',
            'FUSER': 'backbones_2d/fuser', 'BACKBONE_2D': 'backbones_2d', 'DENSE_HEAD': 'dense_heads',
        }
        for config in ['centerpoint', 'transfusion_lidar', 'bevfusion', 'bevfusion_gated']:
            cfg = load_yaml(f'configs/{config}.yaml')
            for key, folder in locations.items():
                if key not in cfg['MODEL']:
                    continue
                tree = ast.parse((ROOT / 'pcdet/models' / folder / '__init__.py').read_text(encoding='utf-8'))
                names = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names}
                self.assertIn(cfg['MODEL'][key]['NAME'], names, f'{config}: {key}')
            for folder, source, transforms in [
                ('processor', 'data_processor.py', cfg['DATA_CONFIG']['DATA_PROCESSOR']),
                ('augmentor', 'data_augmentor.py', cfg['DATA_CONFIG']['DATA_AUGMENTOR']['AUG_CONFIG_LIST']),
            ]:
                tree = ast.parse((ROOT / 'pcdet/datasets' / folder / source).read_text(encoding='utf-8'))
                methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
                for transform in transforms:
                    self.assertIn(transform['NAME'], methods)


@unittest.skipUnless(HAS_TORCH, 'requires torch, numpy and easydict; no CUDA needed')
class FusionTest(unittest.TestCase):
    def test_neutral_initialization_and_gate_gradient(self):
        import torch
        from easydict import EasyDict
        from pcdet.models.backbones_2d.fuser import ConvFuser, GatedFuser
        cfg = EasyDict(IN_CHANNEL=7, OUT_CHANNEL=5, GATE_HIDDEN=4, SAVE_WEIGHTS=True)
        baseline, gated = ConvFuser(cfg).eval(), GatedFuser(cfg).eval()
        gated.conv.load_state_dict(baseline.conv.state_dict())
        image, lidar = torch.randn(2, 3, 8, 9), torch.randn(2, 4, 8, 9)
        inputs = dict(spatial_features_img=image, spatial_features=lidar)
        expected = baseline(dict(inputs))['spatial_features']
        actual = gated(dict(inputs))
        torch.testing.assert_close(actual['spatial_features'], expected)
        torch.testing.assert_close(actual['fusion_weights'].sum(1), torch.ones(2, 8, 9))
        actual['spatial_features'].square().mean().backward()
        self.assertGreater(gated.gate[-1].weight.grad.abs().sum().item(), 0)

    def test_frozen_head_still_computes_loss_and_backpropagates(self):
        import torch
        from torch import nn
        from tools.runtime import freeze_parameters, train_mode

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fuser = nn.Linear(3, 3)
                self.dense_head = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 1))

        model = Toy()
        frozen = freeze_parameters(model, 'fusion')
        train_mode(model, frozen)
        self.assertTrue(model.dense_head.training)
        self.assertFalse(model.dense_head[0].training)
        model.dense_head(model.fuser(torch.randn(4, 3))).square().mean().backward()
        self.assertIsNotNone(model.fuser.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.dense_head.parameters()))


if __name__ == '__main__':
    unittest.main()
