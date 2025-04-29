import torch
import torch.nn as nn
import numpy as np
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=False)
from midas.dpt_depth import DPT
from midas.blocks import Interpolate


class MiDaSUQ(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 2, kernel_size=1, stride=1, padding=0),
            nn.Identity(),
        )
        super().__init__(head, **kwargs)
        self.relu = nn.ReLU(True) if non_negative else nn.Identity()
        self.softplus = nn.Softplus()
        
        if path is not None:
            self.load(path)

    def forward(self, x):
        output = super().forward(x)
        depth = self.relu(output[:, 0, :, :])
        logvar_depth = self.softplus(output[:, 1, :, :])
        return depth, logvar_depth

