"""Shared helpers for staged single-GPU and torchrun DDP training."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Data.ctth_dataset import CTTHDataset, collate_ctth
from Utils.Metric import psnr_torch, ssim_pytorch


# TRAIN_PHASES = (
#     {"start": 1, "end": 100, "crop_size": 128, "batch_size": 8, "frames": 20},
#     {"start": 101, "end": 200, "crop_size": 256, "batch_size": 4, "frames": 10},
#     {"start": 201, "end": 300, "crop_size": 512, "batch_size": 2, "frames": 5},
# )

TRAIN_PHASES = (
    {"start": 1, "end": 30, "crop_size": 256, "batch_size": 4, "frames": 5},
)


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def foreground_weighted_charbonnier(pred, target, foreground_weight=5.0, eps=1e-3):
    """Keep background supervision while emphasizing sparse motion magnitudes."""
    error = torch.sqrt((pred - target).square() + eps * eps)
    weight = 1.0 + foreground_weight * target.detach().clamp(0, 1).sqrt()
    return (error * weight).sum() / weight.sum()


def sobel_edges(image):
    kernel_x = image.new_tensor((
        (-1.0, 0.0, 1.0),
        (-2.0, 0.0, 2.0),
        (-1.0, 0.0, 1.0),
    )).view(1, 1, 3, 3) / 8.0

    kernel_y = kernel_x.transpose(-1, -2)
    channels = image.shape[1]

    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)

    image_pad = F.pad(image, (1, 1, 1, 1), mode="reflect")

    return (
        F.conv2d(image_pad, kernel_x, padding=0, groups=channels),
        F.conv2d(image_pad, kernel_y, padding=0, groups=channels),
    )


def stage1_sharpening_loss(
    pred,
    target,
    foreground_weight=5.0,
    sobel_weight=0.1,
    ssim_weight=0.05,
    eps=1e-3,
):
    """Supervise sparse motion magnitudes while preserving sharp target edges."""
    # Disable nested autocast: flat regions make edge and SSIM gradients fragile
    # in float16 even when the model forward pass itself is safe to run with AMP.
    with torch.cuda.amp.autocast(enabled=False):
        pred_float, target_float = pred.float(), target.float()
        pixel_loss = foreground_weighted_charbonnier(
            pred_float, target_float, foreground_weight, eps
        )
        pred_dx, pred_dy = sobel_edges(pred_float)
        target_dx, target_dy = sobel_edges(target_float)
        sobel_loss = charbonnier(pred_dx, target_dx, eps) + charbonnier(pred_dy, target_dy, eps)
        ssim_loss = 1.0 - ssim_pytorch(pred_float, target_float).clamp(-1.0, 1.0)
        return pixel_loss + sobel_weight * sobel_loss + ssim_weight * ssim_loss


def stage1_structure_fidelity_loss(
    pred,
    target,
    pixel_weight=1.0,
    multiscale_weight=0.5,
    sobel_weight=0.05,
    ssim_weight=0.02,
    scales=(2, 4, 8),
    eps=1e-3,
    return_components=False,
):
    """Dense Stage 1 supervision focused on pixel and global-structure fidelity.

    Full-resolution MSE directly optimizes the error underlying PSNR. MSE on
    progressively average-pooled maps emphasizes low-frequency/global layout,
    while small Sobel and SSIM terms preserve local boundaries and structure.
    Unlike the legacy loss, this formulation makes no sparse-foreground
    assumption and therefore does not suppress the relatively rare low-motion
    pixels in EFTurb's dynamic scenes.
    """
    if min(pixel_weight, multiscale_weight, sobel_weight, ssim_weight) < 0:
        raise ValueError("Stage 1 loss weights must be non-negative")

    # Compute all loss terms in float32. Small gradients in flat regions are
    # unnecessarily fragile in float16 even when the ET forward uses AMP.
    with torch.cuda.amp.autocast(enabled=False):
        pred_float = pred.float().clamp(0.0, 1.0)
        target_float = target.float().clamp(0.0, 1.0)

        pixel_mse = F.mse_loss(pred_float, target_float)

        multiscale_terms = []
        height, width = pred_float.shape[-2:]
        for scale in scales:
            if scale > 1 and min(height, width) >= scale:
                pred_scale = F.avg_pool2d(pred_float, kernel_size=scale, stride=scale)
                target_scale = F.avg_pool2d(target_float, kernel_size=scale, stride=scale)
                multiscale_terms.append(F.mse_loss(pred_scale, target_scale))
        multiscale_mse = (
            torch.stack(multiscale_terms).mean()
            if multiscale_terms
            else pixel_mse.new_zeros(())
        )

        pred_dx, pred_dy = sobel_edges(pred_float)
        target_dx, target_dy = sobel_edges(target_float)
        sobel_loss = (
            0.5 * (
                charbonnier(pred_dx, target_dx, eps)
                + charbonnier(pred_dy, target_dy, eps)
            )
            - eps
        ).clamp_min(0.0)
        ssim_loss = 1.0 - ssim_pytorch(pred_float, target_float).clamp(-1.0, 1.0)

        total = (
            pixel_weight * pixel_mse
            + multiscale_weight * multiscale_mse
            + sobel_weight * sobel_loss
            + ssim_weight * ssim_loss
        )

    if not return_components:
        return total
    return total, {
        "PIXEL_MSE": pixel_mse.detach(),
        "MS_MSE": multiscale_mse.detach(),
        "SOBEL": sobel_loss.detach(),
        "SSIM_LOSS": ssim_loss.detach(),
    }


def motion_weighted_charbonnier(pred, target, motion_weight=2.0, gradient_weight=0.1, eps=1e-3):
    """Emphasize motion structure so sparse targets cannot collapse to zero."""
    error = torch.sqrt((pred - target).square() + eps * eps)
    target_detached = target.detach().clamp(0, 1)
    mean_motion = target_detached.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    weight = (1.0 + motion_weight * target_detached / mean_motion).clamp_max(20.0)
    pixel_loss = (error * weight).sum() / weight.sum()

    pred_dx, pred_dy = pred[..., 1:] - pred[..., :-1], pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., 1:] - target[..., :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    gradient_loss = charbonnier(pred_dx, target_dx, eps) + charbonnier(pred_dy, target_dy, eps)
    return pixel_loss + gradient_weight * gradient_loss


def device_from_arg(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def configure_visible_gpus(gpus):
    if not gpus:
        return
    selected = [item.strip() for item in gpus.split(",") if item.strip()]
    if not selected or any(not item.isdigit() for item in selected):
        raise ValueError("--gpus must be a comma-separated list such as `0`, `2,3` or `0,2,5,7`")
    if len(set(selected)) != len(selected):
        raise ValueError("--gpus must not contain duplicate GPU indices")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)


def setup_training(args):
    configure_visible_gpus(args.gpus)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if distributed:
        if args.device != "cuda":
            raise RuntimeError("DDP training requires CUDA. Launch with torchrun and --device cuda.")
        if args.gpus and torch.cuda.device_count() != int(os.environ["WORLD_SIZE"]):
            raise RuntimeError(
                f"torchrun started {os.environ['WORLD_SIZE']} processes but "
                f"{torch.cuda.device_count()} GPUs are visible. Set --nproc_per_node "
                "to the number of GPUs selected by --gpus."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    seed = args.seed + (dist.get_rank() if distributed else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    return device


def cleanup_training():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def wrap_model(model, device):
    model = model.to(device)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[device.index])
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def move_tensors(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def phase_for_epoch(epoch):
    for phase in TRAIN_PHASES:
        if phase["start"] <= epoch <= phase["end"]:
            return phase
    return TRAIN_PHASES[-1]


def make_loader(
    args,
    split,
    load_events=True,
    load_gt=True,
    shuffle=None,
    crop_size=None,
    frames=None,
    batch_size=None,
):
    dataset = CTTHDataset(
        root=args.data_root,
        split=split,
        crop_size=crop_size or args.crop_size,
        frames=frames or args.frames,
        random_crop=split == "train",
        flow_scale=args.flow_scale,
        voxel_scale=args.voxel_scale,
        load_images=load_events or load_gt,
        load_events=load_events,
        load_gt=load_gt,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        noise_std=args.noise_std if split == "train" else 0.0,
    )
    use_shuffle = split == "train" if shuffle is None else shuffle
    sampler = DistributedSampler(dataset, shuffle=use_shuffle) if dist.is_initialized() and split == "train" else None
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=use_shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_ctth,
        pin_memory=True,
        drop_last=split == "train",
        persistent_workers=args.workers > 0,
    )


def train_loader_for_epoch(args, epoch, **kwargs):
    phase = phase_for_epoch(epoch)
    batch_size = args.train_batch_size_override or phase["batch_size"]
    return make_loader(args, "train", crop_size=phase["crop_size"], frames=phase["frames"], batch_size=batch_size, **kwargs)


def set_loader_epoch(loader, epoch):
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def save_checkpoint(path, **items):
    if not is_main_process():
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, path)


def checkpoint_path(path, epoch):
    path = Path(path)
    return path.with_name(f"{path.stem}_epoch_{epoch:03d}{path.suffix}")


def load_state(model, checkpoint, key, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(state[key])
    return state


def maybe_resume(args, models, optimizer, scheduler, scaler, device):
    if not args.resume:
        return 1
    state = torch.load(args.resume, map_location=device, weights_only=False)
    for key, model in models.items():
        if key in state:
            unwrap_model(model).load_state_dict(state[key])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0) + 1


class ImageMetrics:
    def __init__(self, device, use_lpips=True):
        self.device = device
        self.lpips = None
        if use_lpips:
            try:
                import lpips

                self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
                self.lpips.requires_grad_(False)
            except Exception as exc:
                if is_main_process():
                    print(f"LPIPS unavailable ({exc}); install `lpips` to enable this metric.")
        self.reset()

    def reset(self):
        self.total = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        self.count = 0
        self.lpips_count = 0

    @torch.no_grad()
    def update(self, pred, target, include_lpips=True):
        pred = pred.detach().float().clamp(0, 1)
        target = target.detach().float().clamp(0, 1)
        self.total["psnr"] += psnr_torch(pred, target).item()
        self.total["ssim"] += ssim_pytorch(pred, target).item()
        if self.lpips is not None and include_lpips:
            pred_rgb = pred.repeat(1, 3, 1, 1) * 2 - 1
            target_rgb = target.repeat(1, 3, 1, 1) * 2 - 1
            self.total["lpips"] += self.lpips(pred_rgb, target_rgb).mean().item()
            self.lpips_count += 1
        self.count += 1

    def summary(self, sync=False):
        values = torch.tensor(
            [self.total["psnr"], self.total["ssim"], self.total["lpips"], self.count, self.lpips_count],
            device=self.device,
            dtype=torch.float64,
        )
        if sync and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        psnr, ssim, lpips_total, count, lpips_count = values.tolist()
        count = max(count, 1)
        lpips = f"{lpips_total / lpips_count:.4f}" if lpips_count else "N/A"
        return f"PSNR={psnr / count:.3f} SSIM={ssim / count:.4f} LPIPS={lpips}"


class FlowMetrics(ImageMetrics):
    """Backward-compatible name for the public PSNR/SSIM/LPIPS metric set."""


def save_preview(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().clamp(0, 1)[0, 0].cpu().numpy()
    Image.fromarray((image * 255).round().astype(np.uint8)).save(path)


def save_stage1_preview(prediction, target, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save(image, suffix, contrast=False):
        image = image.detach().float().clamp(0, 1)[0, 0].cpu().numpy()
        if contrast:
            scale = max(float(np.percentile(image, 99.5)), 1e-6)
            image = np.clip(image / scale, 0, 1)
        output = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        Image.fromarray((image * 255).round().astype(np.uint8)).save(output)

    save(prediction, "prediction_raw")
    save(prediction, "prediction_contrast", contrast=True)
    save(target, "target_raw")
    save(target, "target_contrast", contrast=True)


def make_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and torch.cuda.is_available(),
        init_scale=1024.0,
        growth_interval=2000,
    )
    return optimizer, scheduler, scaler


def autocast(args):
    return torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available())


def backward_step(loss, optimizer, scaler, model, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    finite = torch.tensor(
        all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()),
        device=loss.device,
        dtype=torch.int,
    )
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        return False
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return True


def tensors_are_finite(*values):
    device = values[0].device
    finite = torch.tensor(
        all(torch.isfinite(value).all() for value in values),
        device=device,
        dtype=torch.int,
    )
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def report_interval(stage, epoch, step, loss, metrics, optimizer, sync=False, extra=""):
    loss_tensor = torch.tensor(loss, device=metrics.device, dtype=torch.float64)
    if sync and dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
    lr = optimizer.param_groups[0]["lr"]
    if is_main_process():
        suffix = f" {extra}" if extra else ""
        print(f"{stage} epoch={epoch:03d} iter={step:05d} lr={lr:.3e} loss={loss_tensor.item():.6f} {metrics.summary(sync=sync)}{suffix}")
    elif sync:
        metrics.summary(sync=True)


def add_common_args(parser):
    default_data_root = Path(__file__).resolve().parents[1] / "datasets" / "EFTurb"
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--crop-size", type=int, default=256, help="Legacy crop size option")
    parser.add_argument("--validation-crop-size", type=int, default=256, help="Validation crop size; keep modest when GPUs are shared")
    parser.add_argument("--frames", type=int, default=5, help="Used for validation; training uses the fixed three-phase schedule")
    parser.add_argument("--batch-size", type=int, default=1, help="Used for validation; training uses the fixed three-phase schedule")
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size-override", type=int, default=0, help="Override the training batch size; 0 uses the configured curriculum")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flow-scale", type=float, default=4000.0, help="Fixed log1p normalization scale for flow targets")
    parser.add_argument("--voxel-scale", type=float, default=4.0, help="Fixed signed-log1p normalization scale for event voxels")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpus", default="", help="Physical GPU indices, for example `2`, `2,3` or `0,2,5,7`")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs the entire epoch")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--validation-samples", type=int, default=8, help="0 validates the entire held-out set")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.005, help="Gaussian noise augmentation for training images")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=8.0, help="Compatibility option for older motion-weighted losses")
    parser.add_argument("--gradient-weight", type=float, default=0.05, help="Compatibility option for older motion-weighted losses")
    parser.add_argument("--foreground-weight", type=float, default=2.0, help="Legacy Stage 1 sparse-foreground loss weight")
    parser.add_argument("--sobel-weight", type=float, default=0.1, help="Stage 1 Sobel edge consistency loss weight")
    parser.add_argument("--ssim-weight", type=float, default=0.02, help="Stage 1 SSIM structure loss weight")
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--resume", default="")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser
