"""Ref_MambaTM restoration network used by EHETM."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.Hilbert3d import Hilbert3d
from Utils.mambablock import (
    MambaLayerglobal_TM,
    MambaLayerlocal_TM,
    MambaLayerglobalRef,
    MambaLayerlocalRef,
)


def act():
    return nn.GELU()


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


class CAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            act(),
            nn.Conv2d(channels, channels, 3, padding=1),
            CALayer(channels),
        )

    def forward(self, x):
        return x + self.body(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, growth, 3, padding=1), act())

    def forward(self, x):
        return torch.cat([x, self.body(x)], 1)


class RDB(nn.Module):
    def __init__(self, channels, growth, layers=3, dilated=False):
        super().__init__()
        modules, current = [], channels
        for _ in range(layers):
            if dilated:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(current, growth, 3, padding=2, dilation=2),
                        act(),
                    )
                )
            else:
                modules.append(DenseLayer(current, growth))
            current += growth
        self.layers = nn.ModuleList(modules)
        self.compress = nn.Conv2d(current, channels, 1)

    def forward(self, x):
        feature = x
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                feature = layer(feature)
            else:
                feature = torch.cat([feature, layer(feature)], 1)
        return x + self.compress(feature)


class Encoder(nn.Module):
    def __init__(self, features=12, dilated=False):
        super().__init__()
        self.stem = nn.Conv2d(1, features, 5, padding=2)
        self.enc1 = RDB(features, features, dilated=dilated)
        self.down1 = nn.Conv2d(features, features * 2, 5, stride=2, padding=2)
        self.enc2 = RDB(features * 2, features, dilated=dilated)
        self.down2 = nn.Conv2d(features * 2, features * 4, 5, stride=2, padding=2)
        self.enc3 = RDB(features * 4, features * 2, dilated=dilated)
        self.down3 = nn.Conv2d(features * 4, features * 8, 5, stride=2, padding=2)
        self.high = RDB(features * 8, features * 2, dilated=dilated)

    def forward(self, x):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        return x1, x2, x3, self.high(self.down3(x3))


class Decoder(nn.Module):
    def __init__(self, features=12):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 3, 2, 1, output_padding=1)
        self.fuse3 = CAB(features * 8)
        self.up2 = nn.ConvTranspose2d(features * 8, features * 2, 3, 2, 1, output_padding=1)
        self.fuse2 = CAB(features * 4)
        self.up1 = nn.ConvTranspose2d(features * 4, features, 3, 2, 1, output_padding=1)
        self.fuse1 = CAB(features * 2)
        self.output = nn.Sequential(
            nn.Conv2d(features * 2, features, 3, padding=1),
            act(),
            nn.Conv2d(features, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, high, x3, x2, x1):
        x3 = self.fuse3(torch.cat([self.up3(high), x3], 1))
        x2 = self.fuse2(torch.cat([self.up2(x3), x2], 1))
        x1 = self.fuse1(torch.cat([self.up1(x2), x1], 1))
        return self.output(x1)


class RefMambaTM(nn.Module):
    """Restore a grayscale video using a repeated EPAW stability-guide sequence."""

    def __init__(self, features=12, blocks=3, reference_blocks=1):
        super().__init__()
        if reference_blocks >= blocks:
            raise ValueError("reference_blocks must be smaller than blocks")
        self.features = features
        self.blocks = blocks
        self.reference_blocks = reference_blocks
        dim = features * 8
        self.image_encoder = Encoder(features)
        self.guide_encoder = Encoder(features, dilated=True)
        self.spatial_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        self.local_blocks = nn.ModuleList()
        for _ in range(blocks - reference_blocks):
            self.spatial_blocks.append(MambaLayerglobal_TM(dim=dim))
            self.temporal_blocks.append(MambaLayerglobal_TM(dim=dim, spatial_first=False))
            self.local_blocks.append(MambaLayerlocal_TM(dim=dim))
        for _ in range(reference_blocks):
            self.spatial_blocks.append(MambaLayerglobalRef(dim=dim))
            self.temporal_blocks.append(MambaLayerglobalRef(dim=dim, spatial_first=False))
            self.local_blocks.append(MambaLayerlocalRef(dim=dim))
        self.decoder = Decoder(features)
        self.register_buffer("h_curve", torch.empty(0, dtype=torch.long), persistent=False)
        self.curve_shape = None

    def _set_h_curve(self, height, width, frames, device):
        h, w = math.ceil(height / 8), math.ceil(width / 8)
        curve = torch.tensor(
            list(Hilbert3d(width=w, height=h, depth=frames)), dtype=torch.long
        )
        curve = curve[:, 0] * w * frames + curve[:, 1] * frames + curve[:, 2]
        self.h_curve = curve.to(device)
        self.curve_shape = (height, width, frames)

    def _mamba_encode(self, feature):
        for index in range(self.blocks - self.reference_blocks):
            feature = self.spatial_blocks[index](feature)
            feature = self.temporal_blocks[index](feature)
            feature = self.local_blocks[index](feature, self.h_curve)
        return feature

    def forward(self, images, guide_sequence):
        b, t, c, h, w = images.shape
        if self.curve_shape != (h, w, t) or self.h_curve.device != images.device:
            self._set_h_curve(h, w, t, images.device)
        x1, x2, x3, high = self.image_encoder(images.reshape(b * t, c, h, w))
        _, _, _, guide_high = self.guide_encoder(guide_sequence.reshape(b * t, 1, h, w))
        high = high.reshape(b, t, high.shape[1], high.shape[2], high.shape[3])
        guide_high = guide_high.reshape(
            b, t, guide_high.shape[1], guide_high.shape[2], guide_high.shape[3]
        )
        high = self._mamba_encode(high)
        guide_high = self._mamba_encode(guide_high)
        for index in range(self.blocks - self.reference_blocks, self.blocks):
            high = self.spatial_blocks[index](high, guide_high)
            high = self.temporal_blocks[index](high, guide_high)
            high = self.local_blocks[index](high, guide_high, self.h_curve)
        high = high.reshape(b * t, high.shape[2], high.shape[3], high.shape[4])
        restored = self.decoder(high, x3, x2, x1)
        return restored.reshape(b, t, 1, h, w)


def build_restoration_model():
    return RefMambaTM(features=16, blocks=3, reference_blocks=1)


if __name__ == "__main__":
    model = build_restoration_model()
    print(model(torch.randn(1, 5, 1, 64, 64), torch.randn(1, 5, 1, 64, 64)).shape)
