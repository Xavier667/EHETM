"""CBAM-enhanced ETStableNet for scalar rigid-motion guidance."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, 1)


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel=(1, 5, 5)):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel, padding=tuple(k // 2 for k in kernel), bias=False)

    def forward(self, x):
        attention = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(attention))


class CBAM3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention3D(channels)
        self.spatial = SpatialAttention3D()

    def forward(self, x):
        return self.spatial(self.channel(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(in_channels, growth, 3, padding=1), nn.GELU())

    def forward(self, x):
        return torch.cat([x, self.body(x)], 1)


class RDB(nn.Module):
    def __init__(self, channels, growth, layers=3):
        super().__init__()
        modules, current = [], channels
        for _ in range(layers):
            modules.append(DenseLayer(current, growth))
            current += growth
        self.dense = nn.Sequential(*modules)
        self.compress = conv1x1(current, channels)
        self.cbam = CBAM3D(channels)

    def forward(self, x):
        return x + self.cbam(self.compress(self.dense(x)))


class CAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 16, 1)
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = self.body(x)
        return x + residual * self.ca(residual)


class EncoderDecoder3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.stem = nn.Conv3d(2, base, 3, padding=1)
        self.enc1 = RDB(base, base)
        self.down1 = nn.Conv3d(base, base * 2, 3, stride=(1, 2, 2), padding=1)
        self.enc2 = RDB(base * 2, base * 2)
        self.down2 = nn.Conv3d(base * 2, base * 3, 3, stride=(1, 2, 2), padding=1)
        self.enc3 = RDB(base * 3, base * 2)
        self.down3 = nn.Conv3d(base * 3, base * 4, 3, stride=(1, 2, 2), padding=1)
        self.bottleneck = RDB(base * 4, base * 2, layers=2)
        self.up3 = nn.ConvTranspose3d(base * 4, base * 3, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.fuse3 = CAB(base * 6)
        self.up2 = nn.ConvTranspose3d(base * 6, base * 2, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.fuse2 = CAB(base * 4)
        self.up1 = nn.ConvTranspose3d(base * 4, base, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1))
        self.fuse1 = CAB(base * 2)
        self.output = nn.Sequential(
            nn.Conv3d(base * 2, base, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(base, 1, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        bottleneck = self.bottleneck(self.down3(x3))
        x3 = self.fuse3(torch.cat([self.up3(bottleneck), x3], 1))
        x2 = self.fuse2(torch.cat([self.up2(x3), x2], 1))
        x1 = self.fuse1(torch.cat([self.up1(x2), x1], 1))
        return self.output(x1)


class TemporalAggregation(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.score = nn.Sequential(nn.Conv1d(1, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, 1, 1))

    def forward(self, sequence):
        context = sequence.abs().mean(dim=(1, 3, 4)).unsqueeze(1)
        weight = torch.softmax(self.score(context), 2).unsqueeze(-1).unsqueeze(-1)
        return (sequence * weight).sum(2)


class ETStableNet(nn.Module):
    """Predict normalized scalar motion fields.

    Supports single-frame event voxels [B, 2, 10, H, W] and multi-frame windows
    [B, T, 2, 10, H, W]. Multi-frame inputs are processed frame-wise with shared
    weights and return [B, T, 1, H, W].
    """

    def __init__(self, base=16):
        super().__init__()
        self.encoder_decoder = EncoderDecoder3D(base)
        self.temporal_aggregation = TemporalAggregation()

    def forward_single(self, event_voxel):
        scalar_sequence = self.encoder_decoder(event_voxel)
        return torch.sigmoid(self.temporal_aggregation(scalar_sequence))

    def forward(self, event_voxel):
        if event_voxel.ndim == 6:
            b, t = event_voxel.shape[:2]
            prediction = self.forward_single(event_voxel.reshape(b * t, *event_voxel.shape[2:]))
            return prediction.reshape(b, t, *prediction.shape[1:])
        return self.forward_single(event_voxel)


if __name__ == "__main__":
    model = ETStableNet()
    print(model(torch.randn(2, 2, 10, 64, 64)).shape)
    print(model(torch.randn(2, 5, 2, 10, 64, 64)).shape)
