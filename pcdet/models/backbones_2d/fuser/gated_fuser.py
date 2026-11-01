"""Spatial modality gates with a checkpoint-compatible ConvFuser projection."""
import torch
from torch import nn
from .convfuser import ConvFuser


class GatedFuser(ConvFuser):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.gate = nn.Sequential(
            nn.Conv2d(model_cfg.IN_CHANNEL, model_cfg.get('GATE_HIDDEN', 32), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_cfg.get('GATE_HIDDEN', 32), 2, 1),
        )
        # softmax(0, 0) * 2 = (1, 1): initially reproduce ConvFuser.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, batch_dict):
        image = batch_dict['spatial_features_img']
        lidar = batch_dict['spatial_features']
        if image.shape[0] != lidar.shape[0] or image.shape[2:] != lidar.shape[2:]:
            raise ValueError(f'BEV shapes do not align: {image.shape}, {lidar.shape}')
        weights = self.gate(torch.cat([image, lidar], dim=1)).softmax(dim=1)
        features = torch.cat([image * (2 * weights[:, :1]), lidar * (2 * weights[:, 1:])], dim=1)
        batch_dict['spatial_features'] = self.conv(features)
        if self.model_cfg.get('SAVE_WEIGHTS', False):
            batch_dict['fusion_weights'] = weights.detach()
        return batch_dict
