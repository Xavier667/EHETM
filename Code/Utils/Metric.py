import torch
import torch.nn.functional as F
from typing import Tuple

def _gaussian(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g

def create_window(window_size: int, channel: int, sigma: float, device: torch.device, dtype: torch.dtype):
    """Create 2D gaussian window for conv2d filtering. Returns shape [C,1,ks,ks] for grouped conv."""
    _1d = _gaussian(window_size, sigma, device=device, dtype=dtype)
    _2d = _1d[:, None] * _1d[None, :]           # [ks, ks]
    window = _2d.unsqueeze(0).unsqueeze(0)      # [1,1,ks,ks]
    window = window.repeat(channel, 1, 1, 1)    # [C,1,ks,ks]
    return window

def ssim_pytorch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K: Tuple[float, float] = (0.01, 0.03),
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute SSIM between img1 and img2 (PyTorch implementation).
    Args:
        img1, img2: tensors with shape [N, C, H, W], values in same scale (e.g. [0,1])
        window_size: gaussian window size (odd)
        sigma: gaussian sigma
        data_range: dynamic range of pixel values (max-min). Use 1.0 for [0,1]
        K: stability constants (K1, K2)
        reduction: 'mean' (scalar), 'none' (per-batch)
    Returns:
        scalar SSIM (if reduction='mean') or tensor shape [N] (if 'none')
    """
    assert img1.shape == img2.shape, "img shapes must match"
    assert img1.dim() == 4, "expected [N,C,H,W]"

    device = img1.device
    dtype = img1.dtype
    N, C, H, W = img1.shape
    # ensure proper window_size
    if window_size % 2 == 0:
        window_size += 1

    # create gaussian window
    window = create_window(window_size, C, sigma, device, dtype)  # [C,1,ks,ks]

    # padding so output size matches
    padding = window_size // 2

    # conv using groups=C to filter per-channel
    mu1 = F.conv2d(img1, window, groups=C, padding=padding)
    mu2 = F.conv2d(img2, window, groups=C, padding=padding)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=C, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=C, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=C, padding=padding) - mu1_mu2

    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # ssim map per channel
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)

    # average over spatial dims then channels
    # shape: [N, C, H, W] -> mean over H,W -> [N,C] -> mean over C -> [N]
    ssim_per_channel = ssim_map.view(N, C, -1).mean(dim=2)  # [N, C]
    ssim_per_image = ssim_per_channel.mean(dim=1)          # [N]

    if reduction == "mean":
        return ssim_per_image.mean()
    elif reduction == "none":
        return ssim_per_image
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

def psnr_torch(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))
