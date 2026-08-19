"""Medium EPAWStableNet with hand-crafted event statistics and fast residual encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_map(x, eps=1e-6):
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ])

        self.register_buffer(
            "kernel",
            torch.stack([kernel_x, kernel_x.t()]).unsqueeze(1),
        )

    def forward(self, images):
        b, t, _, h, w = images.shape

        images_2d = images.reshape(b * t, 1, h, w)
        images_2d = F.pad(
            images_2d,
            (1, 1, 1, 1),
            mode="reflect",
        )

        gradient = F.conv2d(
            images_2d,
            self.kernel,
            padding=0,
        )

        return gradient.reshape(b, t, 2, h, w)


class WarpedEventStatistics(nn.Module):
    """Hand-crafted per-pixel event density, adaptive TCTS and polarity alternation."""

    def __init__(self, patch_size=5, eps=1e-6):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def forward(self, streams, batch, frames, height, width, device, dtype):
        density_maps, tcts_maps, alternation_maps = [], [], []
        pixels = height * width
        for b in range(batch):
            for frame in range(frames):
                events = streams[b][frame].to(
                    device=device, dtype=dtype, non_blocking=True
                )
                density = torch.zeros(pixels, device=device, dtype=dtype)
                tcts = torch.zeros(pixels, device=device, dtype=dtype)
                alternation = torch.zeros(pixels, device=device, dtype=dtype)
                if events.numel() == 0:
                    density_maps.append(density.view(1, height, width))
                    tcts_maps.append(tcts.view(1, height, width))
                    alternation_maps.append(tcts.view(1, height, width))
                    continue
                x, y, polarity, timestamp = events.unbind(1)
                timestamp = timestamp - timestamp.min()
                timestamp = timestamp / (timestamp.max() + self.eps)
                xi = x.round().long().clamp(0, width - 1)
                yi = y.round().long().clamp(0, height - 1)
                pixel_id = yi * width + xi

                count = torch.zeros(pixels, device=device, dtype=dtype)
                tsum, t2sum = torch.zeros_like(count), torch.zeros_like(count)
                count.index_add_(0, pixel_id, torch.ones_like(timestamp))
                tsum.index_add_(0, pixel_id, timestamp)
                t2sum.index_add_(0, pixel_id, timestamp.square())
                mean = tsum / count.clamp_min(1)
                variance = (t2sum / count.clamp_min(1) - mean.square()).clamp_min(0)
                occupied = count > 0
                density_log = torch.zeros_like(count)
                temporal_consistency = torch.zeros_like(count)
                if occupied.any():
                    valid_variance = variance[occupied]
                    mean_variance = valid_variance.mean().clamp_min(self.eps)
                    temporal_consistency = torch.exp(-variance / (mean_variance + self.eps)) * occupied.to(dtype)
                    density_log = torch.log1p(count)
                    density_log = density_log / density_log[occupied].max().clamp_min(self.eps)
                    threshold = torch.quantile(temporal_consistency[occupied].float(), 0.70).to(dtype)
                    tcts = temporal_consistency * density_log * (temporal_consistency <= threshold).to(dtype)

                # Per-pixel polarity alternation: sort events by pixel, then timestamp.
                order = torch.argsort(pixel_id.float() * 2.0 + timestamp.float())
                sorted_pixel, sorted_polarity = pixel_id[order], polarity[order]
                if sorted_pixel.numel() > 1:
                    same_pixel = sorted_pixel[1:] == sorted_pixel[:-1]
                    flip = same_pixel & (sorted_polarity[1:] * sorted_polarity[:-1] < 0)
                    transition_count = torch.zeros(pixels, device=device, dtype=dtype)
                    transition_count.index_add_(
                        0,
                        sorted_pixel[1:][same_pixel],
                        torch.ones_like(sorted_pixel[1:][same_pixel], dtype=dtype),
                    )
                    if flip.any():
                        ids = sorted_pixel[1:][flip]
                        alternation.index_add_(0, ids, torch.ones_like(ids, dtype=dtype))
                    alternation = alternation * density_log

                density_maps.append(normalize_map(density_log.view(1, height, width), self.eps))
                tcts_maps.append(normalize_map(tcts.view(1, height, width), self.eps))
                alternation_maps.append(normalize_map(alternation.view(1, height, width), self.eps))
        return (
            torch.stack(density_maps).reshape(batch, frames, 1, height, width),
            torch.stack(tcts_maps).reshape(batch, frames, 1, height, width),
            torch.stack(alternation_maps).reshape(batch, frames, 1, height, width),
        )


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


class SpatialAttention3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv3d(2, 1, (1, 5, 5), padding=(0, 2, 2), bias=False)

    def forward(self, x):
        attention = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.body(attention))


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            ChannelAttention3D(channels),
            SpatialAttention3D(),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 4)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = self.body(x)
        return x + residual * self.ca(residual)


class CrossAttention(nn.Module):
    def __init__(self, channels=32, heads=4, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.scene_embed = nn.Conv2d(channels, channels, 1)
        self.et_embed = nn.Conv2d(1, channels, 1)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock2D(channels),
        )

    def forward(self, scene_feature, et_feature):
        h, w = scene_feature.shape[-2:]
        query = F.adaptive_avg_pool2d(self.scene_embed(scene_feature), self.pool_size)
        key_value = F.adaptive_avg_pool2d(self.et_embed(et_feature), self.pool_size)
        query = query.flatten(2).transpose(1, 2)
        key_value = key_value.flatten(2).transpose(1, 2)
        context, _ = self.attention(query, key_value, key_value, need_weights=False)
        context = context.transpose(1, 2).reshape(
            scene_feature.shape[0], -1, self.pool_size, self.pool_size
        )
        context = F.interpolate(context, (h, w), mode="bilinear", align_corners=False)
        return self.output(scene_feature + context)


class EPAWStableNet(nn.Module):
    """Produce per-frame stability guides [B, T, 1, H, W]."""

    def __init__(self, channels=32, blocks=4, event_patch=4):
    # def __init__(self, channels=128, blocks=8, event_patch=4):
        super().__init__()
        self.sobel = SobelGradient()
        self.statistics = WarpedEventStatistics(event_patch)
        branch_channels = channels // 2
        self.gradient_encoder = nn.Sequential(
            nn.Conv3d(2, branch_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock3D(branch_channels),
        )
        self.event_encoder = nn.Sequential(
            nn.Conv3d(3, branch_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock3D(branch_channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GELU(),
            ResidualBlock3D(channels),
        )
        self.body = nn.Sequential(*[ResidualBlock3D(channels) for _ in range(blocks)])
        self.cross_attention = CrossAttention(channels)
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, padding=1),
            nn.GELU(),
            ResidualBlock2D(channels // 2),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, padding=1),
            nn.GELU(),
            ResidualBlock2D(channels // 2),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _expand_et_feature(et_feature, batch, frames, height, width):
        if et_feature.ndim == 5:
            return et_feature.reshape(batch * frames, 1, height, width)
        center = frames // 2
        et_sequence = et_feature.new_zeros(batch, frames, 1, height, width)
        et_sequence[:, center] = et_feature
        return et_sequence.reshape(batch * frames, 1, height, width)

    def forward(self, images, warped_event_stream, et_feature):
        b, t, _, h, w = images.shape
        gradient = self.sobel(images)
        density, tcts, alternation = self.statistics(
            warped_event_stream, b, t, h, w, images.device, images.dtype
        )
        gradient_feature = self.gradient_encoder(gradient.transpose(1, 2))
        event_maps = torch.cat([density, tcts, alternation], dim=2).transpose(1, 2)
        event_feature = self.event_encoder(event_maps)
        feature = self.body(self.fusion(torch.cat([gradient_feature, event_feature], 1)))

        scene_feature = feature.permute(0, 2, 1, 3, 4).reshape(b * t, -1, h, w)
        et_prior = self._expand_et_feature(et_feature, b, t, h, w)
        attended_feature = self.cross_attention(scene_feature, et_prior)

        degraded_gradient_prior = normalize_map(
            gradient.square().sum(dim=2, keepdim=True).clamp_min(1e-8).sqrt()
        )
        degraded_gradient_prior_2d = degraded_gradient_prior.reshape(b * t, 1, h, w)
        refine_input = torch.cat([attended_feature, degraded_gradient_prior_2d], dim=1)
        correction = 0.5 * self.residual_head(refine_input)
        residual_guide = (degraded_gradient_prior_2d + correction).clamp(0, 1)
        reliability = self.reliability_gate(refine_input)
        guide = (degraded_gradient_prior_2d + reliability * correction).clamp(0, 1)

        guide = guide.reshape(b, t, 1, h, w)
        correction = correction.reshape(b, t, 1, h, w)
        residual_guide = residual_guide.reshape(b, t, 1, h, w)
        reliability = reliability.reshape(b, t, 1, h, w)
        aux = {
            "image_gradient": gradient,
            "degraded_gradient_prior": degraded_gradient_prior,
            "correction": correction,
            "residual_guide": residual_guide,
            "reliability": reliability,
            "et_prior": et_prior.reshape(b, t, 1, h, w),
            "event_density": density,
            "tcts": tcts,
            "polarity_alternation": alternation,
        }
        return guide, aux
