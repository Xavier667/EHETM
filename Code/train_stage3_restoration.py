"""Stage 3 joint restoration pretraining on CTTH+ and EFTurb.

This script intentionally trains Ref_MambaTM from scratch with Sobel gradients
of the degraded frames as guides. It does not load ET/EPAW or an old
restoration checkpoint. ``--resume`` is only for continuing an interrupted
joint-pretraining run, including its optimizer and scheduler state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from Data.ctth_dataset import (
    DEFAULT_CTTH_ROOT,
    split_sequences as split_ctth_sequences,
)
from Data.efturb_dataset import (
    DEFAULT_EFTURB_ROOT,
    split_sequences as split_efturb_sequences,
)
from Model.Ref_MambaTM import build_restoration_model
from Utils.Metric import ssim_pytorch
from Utils.train_utils import (
    ImageMetrics,
    add_common_args,
    autocast,
    backward_step,
    barrier,
    charbonnier,
    checkpoint_path,
    cleanup_training,
    is_main_process,
    move_tensors,
    save_checkpoint,
    save_preview,
    setup_training,
    tensors_are_finite,
    unwrap_model,
    wrap_model,
)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def numeric_images(folder):
    folder = Path(folder)
    return {
        int(path.stem): path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem.isdigit()
    }


class MultiFrameRestorationDataset(Dataset):
    """Stage-3-only loader: degraded frame windows and matching GT windows."""

    def __init__(
        self,
        root,
        domain,
        split,
        crop_size,
        frames,
        split_seed,
        train_ratio,
        noise_std=0.0,
        horizontal_flip_probability=0.0,
    ):
        if domain not in {"EFTurb", "CTTH"}:
            raise ValueError(f"unsupported domain: {domain}")
        if crop_size <= 0 or crop_size % 4:
            raise ValueError("crop size must be positive and divisible by 4")
        if frames <= 0:
            raise ValueError("frames must be positive")
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{domain} root does not exist: {self.root}")
        self.domain = domain
        self.split = split
        self.crop_size = crop_size
        self.frames = frames
        self.radius = frames // 2
        self.noise_std = noise_std if split == "train" else 0.0
        self.horizontal_flip_probability = (
            horizontal_flip_probability if split == "train" else 0.0
        )
        self.random_crop = split == "train"
        self.samples = []

        if domain == "EFTurb":
            sequences = split_efturb_sequences(
                self.root, split, split_seed, train_ratio
            )
            input_relative = Path("Turb/frames")
            gt_relative = Path("GT/frames")
        else:
            sequences = split_ctth_sequences(
                self.root, split, split_seed, train_ratio
            )
            input_relative = Path("Turb/frames_gray")
            gt_relative = Path("GT/frames_gray")

        for sequence in sequences:
            images = numeric_images(sequence / input_relative)
            gt_images = numeric_images(sequence / gt_relative)
            relative = sequence.relative_to(self.root)
            sequence_id = "__".join(relative.parts)
            for center in sorted(set(images) & set(gt_images)):
                window = range(
                    center - self.radius,
                    center + (self.frames - self.radius),
                )
                if all(index in images and index in gt_images for index in window):
                    self.samples.append(
                        {
                            "sequence": sequence_id,
                            "center": center,
                            "images": [images[index] for index in window],
                            "gt": [gt_images[index] for index in window],
                        }
                    )
        if not self.samples:
            raise RuntimeError(
                f"no {frames}-frame restoration samples found in {domain} split={split}"
            )

    def __len__(self):
        return len(self.samples)

    def crop_box(self, height, width):
        crop = min(self.crop_size, height, width)
        if crop % 4:
            raise ValueError(
                f"effective crop {crop} must be divisible by 4 for image {width}x{height}"
            )
        if self.random_crop:
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        else:
            top = (height - crop) // 2
            left = (width - crop) // 2
        return top, left, crop

    @staticmethod
    def load_image(path, top, left, crop):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        image = image[top : top + crop, left : left + crop]
        return torch.from_numpy(image.copy()).unsqueeze(0)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["images"][0]) as image:
            width, height = image.size
        top, left, crop = self.crop_box(height, width)
        images = torch.stack(
            [self.load_image(path, top, left, crop) for path in sample["images"]]
        )
        gt = torch.stack(
            [self.load_image(path, top, left, crop) for path in sample["gt"]]
        )
        if self.horizontal_flip_probability > 0 and random.random() < self.horizontal_flip_probability:
            images = torch.flip(images, dims=(-1,))
            gt = torch.flip(gt, dims=(-1,))
        if self.noise_std > 0:
            images = (images + torch.randn_like(images) * self.noise_std).clamp_(0, 1)
        return {
            "images": images,
            "gt": gt,
            "sequence": sample["sequence"],
            "center": sample["center"],
            "domain": self.domain,
        }


def collate_restoration(samples):
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "gt": torch.stack([sample["gt"] for sample in samples]),
        "sequence": [sample["sequence"] for sample in samples],
        "center": [sample["center"] for sample in samples],
        "domain": [sample["domain"] for sample in samples],
    }


def repeated_shuffled_indices(length, count, seed):
    if length <= 0:
        raise ValueError("cannot sample an empty domain")
    result = []
    cycle = 0
    while len(result) < count:
        indices = list(range(length))
        random.Random(seed + cycle * 1000003).shuffle(indices)
        result.extend(indices)
        cycle += 1
    return result[:count]


class BalancedJointDataset(Dataset):
    """Deterministic joint mapping; by default covers both domains in full."""

    def __init__(
        self,
        efturb_dataset,
        ctth_dataset,
        efturb_weight,
        ctth_weight,
        samples_per_epoch,
        seed,
        epoch,
    ):
        if efturb_weight <= 0 or ctth_weight <= 0:
            raise ValueError("domain sampling weights must be positive")
        self.datasets = {"EFTurb": efturb_dataset, "CTTH": ctth_dataset}
        pattern = ["EFTurb"] * efturb_weight + ["CTTH"] * ctth_weight
        if samples_per_epoch <= 0:
            # Default joint pretraining uses every training window from both
            # datasets exactly once before the DDP sampler performs its own
            # global shuffle. A DDP run may repeat at most world_size-1 items
            # solely to give every rank the same number of optimization steps.
            efturb_indices = repeated_shuffled_indices(
                len(efturb_dataset), len(efturb_dataset), seed + epoch * 1009 + 11
            )
            ctth_indices = repeated_shuffled_indices(
                len(ctth_dataset), len(ctth_dataset), seed + epoch * 1009 + 29
            )
            self.entries = (
                [("EFTurb", index) for index in efturb_indices]
                + [("CTTH", index) for index in ctth_indices]
            )
            random.Random(seed + epoch * 1009 + 47).shuffle(self.entries)
            self.domain_counts = {
                "EFTurb": len(efturb_dataset),
                "CTTH": len(ctth_dataset),
            }
            return
        else:
            cycles = max(1, math.ceil(samples_per_epoch / len(pattern)))
            domains = (pattern * cycles)[:samples_per_epoch]
        self.domain_counts = {
            "EFTurb": domains.count("EFTurb"),
            "CTTH": domains.count("CTTH"),
        }
        domain_indices = {
            "EFTurb": iter(
                repeated_shuffled_indices(
                    len(efturb_dataset),
                    self.domain_counts["EFTurb"],
                    seed + epoch * 1009 + 11,
                )
            ),
            "CTTH": iter(
                repeated_shuffled_indices(
                    len(ctth_dataset),
                    self.domain_counts["CTTH"],
                    seed + epoch * 1009 + 29,
                )
            ),
        }
        self.entries = [(domain, next(domain_indices[domain])) for domain in domains]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        domain, sample_index = self.entries[index]
        return self.datasets[domain][sample_index]


def make_domain_dataset(args, domain, split, crop_size, frames):
    return MultiFrameRestorationDataset(
        root=args.efturb_root if domain == "EFTurb" else args.ctth_root,
        domain=domain,
        split=split,
        crop_size=crop_size,
        frames=frames,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        noise_std=args.noise_std,
        horizontal_flip_probability=args.horizontal_flip_probability,
    )


def make_train_loader(args, efturb_dataset, ctth_dataset, epoch):
    dataset = BalancedJointDataset(
        efturb_dataset,
        ctth_dataset,
        args.efturb_sampling_weight,
        args.ctth_sampling_weight,
        args.samples_per_epoch,
        args.seed,
        epoch,
    )
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        sampler.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_restoration,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    return loader, dataset.domain_counts


def evenly_spaced_subset(dataset, sample_count):
    if sample_count <= 0 or sample_count >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
    return Subset(dataset, sorted(set(int(index) for index in indices)))


def make_validation_loader(args, domain):
    dataset = make_domain_dataset(
        args, domain, "test", args.eval_crop_size, args.eval_frames
    )
    selected = evenly_spaced_subset(dataset, args.validation_samples)
    loader = DataLoader(
        selected,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_restoration,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    return loader, len(dataset), len(selected)


def normalize_map(tensor, eps=1e-6):
    minimum = tensor.amin(dim=(-2, -1), keepdim=True)
    maximum = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - minimum) / (maximum - minimum + eps)


def degraded_gradient_guide(images):
    batch, frames, _, height, width = images.shape
    kernel_x = images.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    flat = images.reshape(batch * frames, 1, height, width)
    padded = F.pad(flat, (1, 1, 1, 1), mode="reflect")
    gradient_x = F.conv2d(padded, kernel_x)
    gradient_y = F.conv2d(padded, kernel_y)
    magnitude = (gradient_x.square() + gradient_y.square()).clamp_min(1e-8).sqrt()
    return normalize_map(magnitude).reshape(batch, frames, 1, height, width)


def flatten_time(tensor):
    if tensor.ndim == 5:
        batch, frames = tensor.shape[:2]
        return tensor.reshape(batch * frames, *tensor.shape[2:])
    return tensor


def center_frame(tensor):
    return tensor[:, tensor.shape[1] // 2] if tensor.ndim == 5 else tensor


def restoration_loss(prediction, target, edge_weight, ssim_weight):
    with torch.cuda.amp.autocast(enabled=False):
        prediction = flatten_time(prediction.float().clamp(0, 1))
        target = flatten_time(target.float().clamp(0, 1))
        pixel = charbonnier(prediction, target)
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        edge = 0.5 * (
            charbonnier(pred_dx, target_dx) + charbonnier(pred_dy, target_dy)
        )
        structural = 1.0 - ssim_pytorch(prediction, target).clamp(-1, 1)
        total = pixel + edge_weight * edge + ssim_weight * structural
    return total, {
        "pixel": pixel.detach(),
        "edge": edge.detach(),
        "ssim_loss": structural.detach(),
    }


def restore(images, restoration):
    guide = degraded_gradient_guide(images)
    return restoration(images, guide), guide


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def make_optimizer_and_scheduler(restoration, args):
    optimizer = torch.optim.AdamW(
        restoration.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    minimum_ratio = args.min_lr / args.lr

    def multiplier(epoch_index):
        if args.warmup_epochs > 0 and epoch_index < args.warmup_epochs:
            progress = epoch_index / args.warmup_epochs
            return args.warmup_start_factor + (
                1.0 - args.warmup_start_factor
            ) * progress
        cosine_epochs = max(args.epochs - args.warmup_epochs, 1)
        progress = min(
            max((epoch_index - args.warmup_epochs) / cosine_epochs, 0.0), 1.0
        )
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and torch.cuda.is_available(),
        init_scale=1024.0,
        growth_interval=2000,
    )
    return optimizer, scheduler, scaler


def reduce_domain_statistics(statistics, device):
    values = torch.tensor(
        [
            statistics["EFTurb"][0],
            statistics["EFTurb"][1],
            statistics["CTTH"][0],
            statistics["CTTH"][1],
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    efturb_sum, efturb_count, ctth_sum, ctth_count = values.tolist()
    return {
        "EFTurb": efturb_sum / max(efturb_count, 1),
        "CTTH": ctth_sum / max(ctth_count, 1),
    }


class JointLogger:
    def __init__(self, log_dir, args):
        self.root = Path(log_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.text_path = self.root / "train_log.txt"
        self.train_csv = self.root / "epoch_metrics.csv"
        self.validation_csv = self.root / "validation_metrics.csv"
        self.interval_csv = self.root / "interval_metrics.csv"
        self._header(
            self.interval_csv,
            ["time", "epoch", "iter", "global_step", "loss", "pixel", "edge", "ssim_loss", "lr", "metrics"],
        )
        self._header(
            self.train_csv,
            ["time", "epoch", "loss", "efturb_loss", "ctth_loss", "lr", "metrics"],
        )
        self._header(
            self.validation_csv,
            ["time", "epoch", "domain", "loss", "psnr", "ssim", "lpips", "samples"],
        )
        self.write("======== joint Stage 3 pretraining started ========")
        self.write("args: " + json.dumps(vars(args), ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _header(path, fields):
        if path.exists() and path.stat().st_size:
            return
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(fields)

    def write(self, message):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.text_path.open("a", encoding="utf-8") as file:
            file.write(f"[{now}] {message}\n")

    @staticmethod
    def append(path, row):
        with path.open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(row)

    def interval(self, epoch, step, global_step, loss, components, lr, metrics):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.append(
            self.interval_csv,
            [now, epoch, step, global_step, loss, components["pixel"], components["edge"], components["ssim_loss"], lr, metrics],
        )
        self.write(
            f"interval epoch={epoch:03d} iter={step:05d} global_step={global_step} "
            f"lr={lr:.8e} loss={loss:.8f} pixel={components['pixel']:.8f} "
            f"edge={components['edge']:.8f} ssim_loss={components['ssim_loss']:.8f} "
            f"{metrics}"
        )

    def train_epoch(self, epoch, loss, domain_losses, lr, metrics):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.append(
            self.train_csv,
            [now, epoch, loss, domain_losses["EFTurb"], domain_losses["CTTH"], lr, metrics],
        )
        self.write(
            f"epoch={epoch:03d} loss={loss:.8f} EFTurb_loss={domain_losses['EFTurb']:.8f} "
            f"CTTH_loss={domain_losses['CTTH']:.8f} lr={lr:.8e} {metrics}"
        )

    def validation(self, epoch, result):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.append(
            self.validation_csv,
            [now, epoch, result["domain"], result["loss"], result["psnr"], result["ssim"], result["lpips"], result["samples"]],
        )
        self.write(
            f"validation epoch={epoch:03d} domain={result['domain']} "
            f"loss={result['loss']:.8f} PSNR={result['psnr']:.3f} "
            f"SSIM={result['ssim']:.4f} LPIPS={result['lpips']} samples={result['samples']}"
        )


@torch.no_grad()
def validate_domain(args, epoch, domain, loader, restoration, device):
    restoration = unwrap_model(restoration).eval()
    metrics = ImageMetrics(device, use_lpips=True)
    loss_total = 0.0
    steps = 0
    for step, batch in enumerate(loader, 1):
        batch = move_tensors(batch, device)
        with autocast(args):
            restored, guide = restore(batch["images"], restoration)
        loss, _ = restoration_loss(
            restored, batch["gt"], args.edge_loss_weight, args.restoration_ssim_weight
        )
        loss_total += loss.item()
        steps += 1
        metrics.update(
            flatten_time(restored),
            flatten_time(batch["gt"]),
        )
        if step <= 4:
            preview = (
                Path(args.results_dir)
                / domain.lower()
                / f"epoch_{epoch:03d}"
                / f"{batch['sequence'][0]}_{batch['center'][0]:05d}.png"
            )
            save_preview(center_frame(restored), preview)
            save_preview(
                center_frame(guide),
                preview.with_name(f"{preview.stem}_guide{preview.suffix}"),
            )
    count = max(metrics.count, 1)
    result = {
        "domain": domain,
        "loss": loss_total / max(steps, 1),
        "psnr": metrics.total["psnr"] / count,
        "ssim": metrics.total["ssim"] / count,
        "lpips": (
            metrics.total["lpips"] / metrics.lpips_count
            if metrics.lpips_count
            else None
        ),
        "samples": len(loader.dataset),
    }
    print(
        f"stage3 validation epoch={epoch:03d} domain={domain} "
        f"loss={result['loss']:.6f} PSNR={result['psnr']:.3f} "
        f"SSIM={result['ssim']:.4f} LPIPS={result['lpips']} "
        f"samples={result['samples']}",
        flush=True,
    )
    return result


def best_checkpoint_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}_best{path.suffix}")


def load_joint_resume(args, restoration, optimizer, scheduler, scaler, device):
    if not args.resume:
        return 1, 0, float("inf"), 0
    state = torch.load(args.resume, map_location=device, weights_only=False)
    required = {"restoration_model", "optimizer", "scheduler", "epoch"}
    missing = sorted(required - set(state))
    if missing:
        raise KeyError(f"resume checkpoint is incomplete; missing {missing}")
    unwrap_model(restoration).load_state_dict(state["restoration_model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return (
        int(state["epoch"]) + 1,
        int(state.get("global_step", 0)),
        float(state.get("best_joint_val_loss", float("inf"))),
        int(state.get("epochs_without_improvement", 0)),
    )


def parse_args():
    parser = add_common_args(argparse.ArgumentParser())
    parser.set_defaults(
        epochs=30,
        lr=1e-4,
        min_lr=1e-6,
        print_every=50,
        validation_samples=32,
        noise_std=0.005,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ctth-root", default=str(DEFAULT_CTTH_ROOT))
    parser.add_argument("--efturb-root", default=str(DEFAULT_EFTURB_ROOT))
    parser.add_argument("--efturb-sampling-weight", type=int, default=1)
    parser.add_argument("--ctth-sampling-weight", type=int, default=1)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="0 uses every EFTurb and CTTH training window once; positive values enable weighted sampling",
    )
    parser.add_argument("--train-crop-size", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-frames", type=int, default=5)
    parser.add_argument("--eval-crop-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-frames", type=int, default=5)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--edge-loss-weight", type=float, default=0.05)
    parser.add_argument("--restoration-ssim-weight", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "checkpoints" / "stage3_restoration.pt"),
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "outputs" / "validation" / "stage3_pretrain"),
    )
    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_ROOT / "outputs" / "logs" / "stage3_pretrain"),
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.lr <= 0 or args.min_lr < 0:
        parser.error("epochs/lr must be positive and min-lr cannot be negative")
    if args.min_lr >= args.lr:
        parser.error("--min-lr must be smaller than --lr")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs must satisfy 0 <= warmup-epochs < epochs")
    if not 0 < args.warmup_start_factor <= 1:
        parser.error("--warmup-start-factor must be in (0, 1]")
    if not 0 <= args.horizontal_flip_probability <= 1:
        parser.error("--horizontal-flip-probability must be in [0, 1]")
    if min(args.edge_loss_weight, args.restoration_ssim_weight, args.weight_decay) < 0:
        parser.error("loss weights and weight decay cannot be negative")
    if args.samples_per_epoch < 0:
        parser.error("--samples-per-epoch cannot be negative")
    return args


def main():
    args = parse_args()
    device = setup_training(args)
    logger = JointLogger(args.log_dir, args) if is_main_process() else None

    train_efturb = make_domain_dataset(
        args, "EFTurb", "train", args.train_crop_size, args.train_frames
    )
    train_ctth = make_domain_dataset(
        args, "CTTH", "train", args.train_crop_size, args.train_frames
    )
    if is_main_process():
        val_efturb_loader, val_efturb_total, val_efturb_used = make_validation_loader(
            args, "EFTurb"
        )
        val_ctth_loader, val_ctth_total, val_ctth_used = make_validation_loader(
            args, "CTTH"
        )
    else:
        val_efturb_loader = val_ctth_loader = None
        val_efturb_total = val_efturb_used = val_ctth_total = val_ctth_used = 0

    restoration = wrap_model(build_restoration_model(), device)
    optimizer, scheduler, scaler = make_optimizer_and_scheduler(restoration, args)
    start_epoch, global_step, best_joint_val_loss, epochs_without_improvement = load_joint_resume(
        args, restoration, optimizer, scheduler, scaler, device
    )
    if is_main_process():
        sampling_description = (
            "full EFTurb + full CTTH every epoch"
            if args.samples_per_epoch == 0
            else (
                f"weighted {args.efturb_sampling_weight}:"
                f"{args.ctth_sampling_weight}, samples_per_epoch={args.samples_per_epoch}"
            )
        )
        message = (
            "stage3 joint pretraining: random initialization, degraded-gradient guide, "
            f"EFTurb_train={len(train_efturb)} CTTH_train={len(train_ctth)} "
            f"EFTurb_val={val_efturb_used}/{val_efturb_total} "
            f"CTTH_val={val_ctth_used}/{val_ctth_total} "
            f"sampling={sampling_description} "
            f"lr={args.lr:.3e} min_lr={args.min_lr:.3e} warmup={args.warmup_epochs}"
        )
        if args.resume:
            message = (
                f"resuming joint pretraining at epoch {start_epoch} from {args.resume}; "
                + message.split(": ", 1)[1]
            )
        print(message, flush=True)
        logger.write(message)

    stop_training = False
    for epoch in range(start_epoch, args.epochs + 1):
        loader, planned_counts = make_train_loader(
            args, train_efturb, train_ctth, epoch
        )
        restoration.train()
        metrics = ImageMetrics(device, use_lpips=is_main_process())
        epoch_metrics = ImageMetrics(device, use_lpips=False)
        epoch_loss_sum = 0.0
        epoch_steps = 0
        domain_statistics = {"EFTurb": [0.0, 0], "CTTH": [0.0, 0]}
        interval = {
            "loss": 0.0,
            "pixel": 0.0,
            "edge": 0.0,
            "ssim_loss": 0.0,
            "steps": 0,
        }
        lr_used = current_lr(optimizer)
        for step, batch in enumerate(loader, 1):
            batch = move_tensors(batch, device)
            with autocast(args):
                restored, guide = restore(batch["images"], restoration)
            loss, components = restoration_loss(
                restored,
                batch["gt"],
                args.edge_loss_weight,
                args.restoration_ssim_weight,
            )
            if not tensors_are_finite(restored, batch["gt"], guide, loss):
                if is_main_process():
                    print(
                        f"stage3 epoch={epoch:03d} iter={step:05d} skipped non-finite forward",
                        flush=True,
                    )
                continue
            if not backward_step(loss, optimizer, scaler, restoration, args.grad_clip):
                if is_main_process():
                    print(
                        f"stage3 epoch={epoch:03d} iter={step:05d} skipped non-finite gradients",
                        flush=True,
                    )
                continue
            global_step += 1
            loss_value = loss.item()
            epoch_loss_sum += loss_value
            epoch_steps += 1
            interval["loss"] += loss_value
            interval["pixel"] += components["pixel"].item()
            interval["edge"] += components["edge"].item()
            interval["ssim_loss"] += components["ssim_loss"].item()
            interval["steps"] += 1
            batch_domains = set(batch["domain"])
            for domain in batch_domains:
                indices = [i for i, value in enumerate(batch["domain"]) if value == domain]
                if len(batch_domains) == 1:
                    domain_loss_value = loss_value
                else:
                    with torch.no_grad():
                        domain_loss, _ = restoration_loss(
                            restored[indices],
                            batch["gt"][indices],
                            args.edge_loss_weight,
                            args.restoration_ssim_weight,
                        )
                    domain_loss_value = domain_loss.item()
                domain_statistics[domain][0] += domain_loss_value * len(indices)
                domain_statistics[domain][1] += len(indices)
            metrics.update(
                flatten_time(restored),
                flatten_time(batch["gt"]),
                include_lpips=step == 1 or step % args.print_every == 0,
            )
            epoch_metrics.update(
                flatten_time(restored.detach()),
                flatten_time(batch["gt"]),
                include_lpips=False,
            )

            if step % args.print_every == 0:
                count = max(interval["steps"], 1)
                summary = metrics.summary(sync=True)
                values = {
                    key: interval[key] / count
                    for key in ("loss", "pixel", "edge", "ssim_loss")
                }
                if is_main_process():
                    print(
                        f"stage3-joint epoch={epoch:03d} iter={step:05d}/{len(loader):05d} "
                        f"lr={current_lr(optimizer):.3e} loss={values['loss']:.6f} "
                        f"pixel={values['pixel']:.6f} edge={values['edge']:.6f} "
                        f"ssim_loss={values['ssim_loss']:.6f} {summary}",
                        flush=True,
                    )
                    logger.interval(
                        epoch, step, global_step, values["loss"], values,
                        current_lr(optimizer), summary,
                    )
                for key in interval:
                    interval[key] = 0 if key == "steps" else 0.0
                metrics.reset()
            if args.max_steps and step >= args.max_steps:
                break

        # All ranks participate; this is the complete epoch rather than only
        # the unfinished logging interval.
        summary = epoch_metrics.summary(sync=True)
        domain_losses = reduce_domain_statistics(domain_statistics, device)
        global_totals = torch.tensor(
            [epoch_loss_sum, epoch_steps], dtype=torch.float64, device=device
        )
        if dist.is_initialized():
            dist.all_reduce(global_totals, op=dist.ReduceOp.SUM)
        train_loss = global_totals[0].item() / max(global_totals[1].item(), 1)
        if is_main_process():
            message = (
                f"stage3-joint epoch={epoch:03d} completed loss={train_loss:.6f} "
                f"EFTurb_loss={domain_losses['EFTurb']:.6f} "
                f"CTTH_loss={domain_losses['CTTH']:.6f} lr={lr_used:.3e} "
                f"planned_samples={planned_counts}"
            )
            print(message, flush=True)
            logger.train_epoch(epoch, train_loss, domain_losses, lr_used, summary)
        scheduler.step()

        validation = None
        barrier()
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            if is_main_process():
                torch.cuda.empty_cache()
                efturb_result = validate_domain(
                    args, epoch, "EFTurb", val_efturb_loader, restoration, device
                )
                ctth_result = validate_domain(
                    args, epoch, "CTTH", val_ctth_loader, restoration, device
                )
                logger.validation(epoch, efturb_result)
                logger.validation(epoch, ctth_result)
                joint_loss = 0.5 * (
                    efturb_result["loss"] + ctth_result["loss"]
                )
                validation = {
                    "EFTurb": efturb_result,
                    "CTTH": ctth_result,
                    "joint_loss": joint_loss,
                }
                improved = joint_loss < best_joint_val_loss
                if improved:
                    best_joint_val_loss = joint_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += args.validate_every
                print(
                    f"stage3-joint validation epoch={epoch:03d} "
                    f"joint_loss={joint_loss:.6f} best={best_joint_val_loss:.6f} "
                    f"patience={epochs_without_improvement}/{args.early_stopping_patience}",
                    flush=True,
                )
            else:
                improved = False
        else:
            improved = False
        barrier()

        if is_main_process():
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "restoration_model": unwrap_model(restoration).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_joint_val_loss": best_joint_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "validation": validation,
                "args": vars(args),
            }
            save_checkpoint(args.output, **state)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                save_checkpoint(checkpoint_path(args.output, epoch), **state)
            if improved:
                save_checkpoint(best_checkpoint_path(args.output), **state)
                logger.write(
                    f"new best joint checkpoint at epoch={epoch}: "
                    f"joint_loss={best_joint_val_loss:.8f}"
                )
            stop_training = (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            )
        stop_tensor = torch.tensor(
            int(stop_training), dtype=torch.int, device=device
        )
        if dist.is_initialized():
            dist.broadcast(stop_tensor, src=0)
        stop_training = bool(stop_tensor.item())
        if stop_training:
            if is_main_process():
                message = (
                    f"early stopping after epoch {epoch}; best joint validation "
                    f"loss={best_joint_val_loss:.8f}"
                )
                print(message, flush=True)
                logger.write(message)
            break
        del loader, metrics, epoch_metrics
        torch.cuda.empty_cache()

    if is_main_process():
        logger.write("======== joint Stage 3 pretraining finished ========")
    cleanup_training()


if __name__ == "__main__":
    main()
