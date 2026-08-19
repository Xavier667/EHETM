"""Stage 2: train EPAWStableNet with gradient-map supervision and logs."""

import argparse
import csv
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
logging.getLogger("mmcv").setLevel(logging.ERROR)

from Model.EPAWStableNet_new import EPAWStableNet
from Model.ETStableNet import ETStableNet
from Data.efturb_dataset import (
    DEFAULT_EFTURB_ROOT, EFTurbDataset, collate_efturb, numeric_files,
)
from Utils.train_utils import (
    ImageMetrics, add_common_args, autocast, backward_step, barrier, charbonnier,
    checkpoint_path, cleanup_training, is_main_process, load_state, make_loader,
    make_optimizer_and_scheduler, maybe_resume, move_tensors, phase_for_epoch,
    report_interval, save_checkpoint, save_preview, set_loader_epoch,
    setup_training, tensors_are_finite, train_loader_for_epoch, unwrap_model, wrap_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent


class MultiFrameGradientEFTurbDataset(EFTurbDataset):
    """EFTurb stage2 dataset with clean gradient targets for every input frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gradient_cache = {}
        filtered = []
        for sample in self.samples:
            sequence_path = sample["sequence_path"]
            if sequence_path not in gradient_cache:
                gradient_cache[sequence_path] = numeric_files(
                    sequence_path / "Optical_Flow" / "raw_gradient_png_8bit", ".png"
                )
            gradients = gradient_cache[sequence_path]
            frame_ids = [int(Path(path).stem) for path in sample["images"]]
            if all(frame_id in gradients for frame_id in frame_ids):
                sample = dict(sample)
                sample["gradient_window"] = [gradients[frame_id] for frame_id in frame_ids]
                filtered.append(sample)
        if not filtered:
            raise RuntimeError(
                f"No multi-frame gradient samples found under {self.root} for split={self.split}"
            )
        dropped = len(self.samples) - len(filtered)
        self.samples = filtered
        if dropped:
            print(
                f"[INFO] dropped {dropped} samples without full {self.frames}-frame gradient supervision "
                f"for split={self.split}",
                flush=True,
            )

    def __getitem__(self, index):
        output = super().__getitem__(index)
        top, left, crop = [int(value) for value in output["crop"].tolist()]
        output["gradient_target"] = torch.stack(
            [self._image(path, top, left, crop) for path in self.samples[index]["gradient_window"]]
        )
        return output


def make_multiframe_loader(
    args,
    split,
    load_events=True,
    load_gt=True,
    shuffle=None,
    crop_size=None,
    frames=None,
    batch_size=None,
):
    dataset = MultiFrameGradientEFTurbDataset(
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
        collate_fn=collate_efturb,
        pin_memory=True,
        drop_last=split == "train",
        persistent_workers=args.workers > 0,
    )


def train_multiframe_loader_for_epoch(args, epoch, **kwargs):
    phase = phase_for_epoch(epoch)
    batch_size = args.train_batch_size_override or phase["batch_size"]
    return make_multiframe_loader(
        args,
        "train",
        crop_size=phase["crop_size"],
        frames=phase["frames"],
        batch_size=batch_size,
        **kwargs,
    )


def flatten_time(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        b, t = tensor.shape[:2]
        return tensor.reshape(b * t, *tensor.shape[2:])
    return tensor


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


def epaw_guidance_loss(guide, target, edge_loss_weight=0.0):
    pixel_loss = charbonnier(guide, target)
    if edge_loss_weight <= 0:
        return pixel_loss
    with torch.cuda.amp.autocast(enabled=False):
        guide_float = flatten_time(guide).float()
        target_float = flatten_time(target).float()
        guide_dx, guide_dy = sobel_edges(guide_float)
        target_dx, target_dy = sobel_edges(target_float)
        edge_loss = charbonnier(guide_dx, target_dx) + charbonnier(guide_dy, target_dy)
    return pixel_loss + edge_loss_weight * edge_loss


def save_tensor_preview(tensor, path, normalize=False, signed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 5:
        center = tensor.shape[1] // 2
        tensor = tensor[0, center]
    elif tensor.ndim == 4:
        tensor = tensor[0]
    elif tensor.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape for preview: {tuple(tensor.shape)}")

    if tensor.ndim == 3 and tensor.shape[0] == 2:
        image = tensor.square().sum(dim=0).sqrt()
    elif tensor.ndim == 3:
        image = tensor[0]
    else:
        image = tensor

    if signed:
        max_abs = image.abs().max().clamp_min(1e-6)
        image = image / (2.0 * max_abs) + 0.5
    elif normalize:
        image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)
    else:
        image = image.clamp(0, 1)
    image = image.numpy()
    Image.fromarray((image * 255).round().astype(np.uint8)).save(path)


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0


def parse_metric_summary(summary):
    metrics = {}
    for key, value in re.findall(r"([A-Za-z_][\w./-]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", summary):
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    return metrics


def tensor_finite_stats(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        return f"{name}: not_tensor"
    with torch.no_grad():
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        nan_count = torch.isnan(detached).sum().item()
        inf_count = torch.isinf(detached).sum().item()
        finite_count = finite.sum().item()
        total = detached.numel()
        if finite_count:
            values = detached[finite]
            return (
                f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
                f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} "
                f"min={values.min().item():.6g} max={values.max().item():.6g} "
                f"mean={values.float().mean().item():.6g}"
            )
        return (
            f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
            f"finite=0/{total} nan={nan_count} inf={inf_count}"
        )


def batch_meta(batch):
    parts = []
    for key in ("sequence", "center", "index"):
        if key in batch:
            value = batch[key]
            try:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().tolist()
                parts.append(f"{key}={value}")
            except Exception:
                parts.append(f"{key}=<unprintable>")
    return " ".join(parts)


def save_bad_batch(args, epoch, step, batch, guide, loss, reason):
    if not is_main_process():
        return None
    bad_dir = Path(args.log_dir) / "bad_batches"
    bad_dir.mkdir(parents=True, exist_ok=True)
    path = bad_dir / f"bad_epoch_{epoch:03d}_iter_{step:05d}.pt"
    payload = {
        "epoch": epoch,
        "step": step,
        "reason": reason,
        "loss": loss.detach().cpu() if isinstance(loss, torch.Tensor) else loss,
        "guide": guide.detach().cpu() if isinstance(guide, torch.Tensor) else None,
    }
    for key in ("event_voxel", "images", "gradient_target"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            payload[key] = batch[key].detach().cpu()
    for key in ("sequence", "center", "index"):
        if key in batch:
            payload[key] = batch[key]
    torch.save(payload, path)
    return path


def log_nonfinite(logger, args, epoch, step, batch, guide, loss, reason):
    if not is_main_process():
        return
    lines = [
        f"NON_FINITE reason={reason} epoch={epoch:03d} iter={step:05d} {batch_meta(batch)}",
        tensor_finite_stats("event_voxel", batch.get("event_voxel")),
        tensor_finite_stats("images", batch.get("images")),
        tensor_finite_stats("gradient_target", batch.get("gradient_target")),
        tensor_finite_stats("guide", guide),
        tensor_finite_stats("loss", loss),
    ]
    bad_path = save_bad_batch(args, epoch, step, batch, guide, loss, reason)
    if bad_path is not None:
        lines.append(f"saved_bad_batch={bad_path}")
    message = "\n".join(lines)
    print(message)
    if logger is not None:
        logger.write_text(message)


class TrainingLogger:
    def __init__(self, log_dir, args):
        self.log_dir = Path(log_dir)
        self.curve_dir = self.log_dir / "curves"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.curve_dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = self.log_dir / "train_log.txt"
        self.interval_csv = self.log_dir / "interval_metrics.csv"
        self.epoch_csv = self.log_dir / "epoch_metrics.csv"
        self.val_csv = self.log_dir / "validation_metrics.csv"
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self._write_header(self.interval_csv, ["time", "epoch", "iter", "global_step", "loss", "lr", "metrics_json", "metrics_text"])
        self._write_header(self.epoch_csv, ["time", "epoch", "avg_loss", "best_interval_loss", "best_train_loss", "lr", "metrics_json", "metrics_text"])
        self._write_header(self.val_csv, ["time", "epoch", "val_loss", "best_val_loss", "metrics_json", "metrics_text"])
        self.write_text("======== training started ========")
        self.write_text("args: " + json.dumps(vars(args), ensure_ascii=False, sort_keys=True))

    def _write_header(self, path, header):
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

    def write_text(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def append_csv(self, path, row):
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)

    def log_interval(self, epoch, step, global_step, loss, lr, metrics_text):
        metrics = parse_metric_summary(metrics_text)
        self.best_train_loss = min(self.best_train_loss, loss)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.append_csv(
            self.interval_csv,
            [now, epoch, step, global_step, f"{loss:.8f}", f"{lr:.8e}", json.dumps(metrics), metrics_text],
        )
        self.write_text(
            f"interval epoch={epoch:03d} iter={step:05d} global_step={global_step} "
            f"loss={loss:.8f} best_train_loss={self.best_train_loss:.8f} lr={lr:.8e} {metrics_text}"
        )

    def log_epoch(self, epoch, avg_loss, best_interval_loss, lr, metrics_text):
        metrics = parse_metric_summary(metrics_text)
        self.best_train_loss = min(self.best_train_loss, avg_loss, best_interval_loss)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.append_csv(
            self.epoch_csv,
            [
                now, epoch, f"{avg_loss:.8f}", f"{best_interval_loss:.8f}",
                f"{self.best_train_loss:.8f}", f"{lr:.8e}", json.dumps(metrics), metrics_text,
            ],
        )
        self.write_text(
            f"epoch epoch={epoch:03d} avg_loss={avg_loss:.8f} "
            f"best_interval_loss={best_interval_loss:.8f} best_train_loss={self.best_train_loss:.8f} "
            f"lr={lr:.8e} {metrics_text}"
        )
        self.plot_curves()

    def log_validation(self, epoch, val_loss, metrics_text):
        metrics = parse_metric_summary(metrics_text)
        self.best_val_loss = min(self.best_val_loss, val_loss)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.append_csv(
            self.val_csv,
            [now, epoch, f"{val_loss:.8f}", f"{self.best_val_loss:.8f}", json.dumps(metrics), metrics_text],
        )
        self.write_text(
            f"validation epoch={epoch:03d} val_loss={val_loss:.8f} "
            f"best_val_loss={self.best_val_loss:.8f} {metrics_text}"
        )
        self.plot_curves()

    def _read_float_series(self, path, x_key, y_key):
        if not path.exists():
            return [], []
        xs, ys = [], []
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    xs.append(float(row[x_key]))
                    ys.append(float(row[y_key]))
                except (KeyError, TypeError, ValueError):
                    continue
        return xs, ys

    def plot_curves(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.write_text(f"curve plotting skipped: {repr(exc)}")
            return
        plots = [
            (self.interval_csv, "global_step", "loss", "train_interval_loss.png", "Train Interval Loss"),
            (self.epoch_csv, "epoch", "avg_loss", "train_epoch_avg_loss.png", "Train Epoch Avg Loss"),
            (self.epoch_csv, "epoch", "best_interval_loss", "train_epoch_best_interval_loss.png", "Train Epoch Best Interval Loss"),
            (self.val_csv, "epoch", "val_loss", "validation_loss.png", "Validation Loss"),
        ]
        for csv_path, x_key, y_key, name, title in plots:
            xs, ys = self._read_float_series(csv_path, x_key, y_key)
            if not xs:
                continue
            plt.figure(figsize=(7, 4), dpi=160)
            plt.plot(xs, ys, linewidth=1.8)
            plt.xlabel(x_key)
            plt.ylabel(y_key)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.curve_dir / name)
            plt.close()


@torch.no_grad()
def predict(batch, et_model, epaw_model):
    et_feature = et_model(batch["event_voxel"])
    return epaw_model(batch["images"], batch["warped_events"], et_feature)


@torch.no_grad()
def validate(args, epoch, et_model, epaw_model, device, logger=None):
    if not is_main_process():
        return None
    torch.cuda.empty_cache()
    loader = make_multiframe_loader(
        args,
        "test",
        load_events=True,
        load_gt=False,
        shuffle=False,
        crop_size=args.validation_crop_size,
        batch_size=args.validation_batch_size,
    )
    metrics = ImageMetrics(device, use_lpips=True)
    et_model = unwrap_model(et_model).eval()
    epaw_model = unwrap_model(epaw_model).eval()
    total_loss, steps = 0.0, 0
    for step, batch in enumerate(loader, 1):
        batch = move_tensors(batch, device)
        with autocast(args):
            guide, _ = predict(batch, et_model, epaw_model)
            loss = epaw_guidance_loss(guide, batch["gradient_target"], args.edge_loss_weight)
        total_loss += loss.item()
        steps += 1
        metrics.update(flatten_time(guide), flatten_time(batch["gradient_target"]))
        if step <= 4:
            base_path = Path(args.results_dir) / f"epoch_{epoch:03d}" / f"{batch['sequence'][0]}_{batch['center'][0]:05d}.png"
            save_tensor_preview(guide, base_path.with_name(f"{base_path.stem}_guide.png"))
            save_tensor_preview(
                batch["gradient_target"],
                base_path.with_name(f"{base_path.stem}_target.png"),
            )
        if args.validation_samples and step >= args.validation_samples:
            break
    val_loss = total_loss / max(steps, 1)
    summary = metrics.summary()
    print(f"stage2 validation epoch={epoch:03d} val_loss={val_loss:.6f} {summary}")
    if logger is not None:
        logger.log_validation(epoch, val_loss, summary)
    del metrics, loader
    torch.cuda.empty_cache()
    return val_loss


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.set_defaults(data_root=str(DEFAULT_EFTURB_ROOT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--et-checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "stage1_et.pt"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "checkpoints" / "stage2_epaw.pt"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "outputs" / "validation" / "stage2"))
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "outputs" / "logs" / "stage2"))
    parser.add_argument("--log-every", type=int, default=0, help="0 uses --print-every")
    parser.add_argument("--edge-loss-weight", type=float, default=0.05, help="Optional Sobel edge consistency weight for EPAW guide supervision")
    parser.add_argument(
        "--pretrained",
        default="",
        help=(
            "Load only epaw_model weights from a Stage 2 checkpoint. Training "
            "starts at epoch 1 with a new optimizer, scheduler, and AMP scaler. "
            "The frozen ET model is still loaded from --et-checkpoint."
        ),
    )
    args = parser.parse_args()
    if args.pretrained and args.resume:
        parser.error("--pretrained and --resume are mutually exclusive")
    device = setup_training(args)
    logger = TrainingLogger(args.log_dir, args) if is_main_process() else None
    et_model = ETStableNet().to(device).eval()
    load_state(et_model, args.et_checkpoint, "et_model", device)
    et_model.requires_grad_(False)
    epaw_model = wrap_model(EPAWStableNet(), device)
    optimizer, scheduler, scaler = make_optimizer_and_scheduler(epaw_model, args)
    if args.pretrained:
        load_state(epaw_model, args.pretrained, "epaw_model", device)
        start_epoch = 1
        if is_main_process():
            message = (
                f"loaded pretrained EPAW weights only from {args.pretrained}; "
                "starting at epoch 1 with fresh optimizer/scheduler/scaler; "
                f"frozen ET checkpoint={args.et_checkpoint}"
            )
            print(message)
            logger.write_text(message)
    else:
        start_epoch = maybe_resume(
            args, {"epaw_model": epaw_model}, optimizer, scheduler, scaler, device
        )
    loader = None
    previous_phase = None
    log_every = args.log_every if args.log_every > 0 else args.print_every
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        phase = phase_for_epoch(epoch)
        if phase != previous_phase:
            loader = train_multiframe_loader_for_epoch(args, epoch, load_events=True, load_gt=False)
            previous_phase = phase
            if is_main_process():
                message = (
                    f"stage2 phase: epochs={phase['start']}-{phase['end']} crop={phase['crop_size']} "
                    f"batch={args.train_batch_size_override or phase['batch_size']} frames={phase['frames']} "
                    f"log_every={log_every}"
                )
                print(message)
                logger.write_text(message)
        set_loader_epoch(loader, epoch)
        epaw_model.train()
        metrics = ImageMetrics(device, use_lpips=is_main_process())
        epoch_metrics = ImageMetrics(device, use_lpips=False)
        interval_total, interval_steps = 0.0, 0
        epoch_total, epoch_steps = 0.0, 0
        best_interval_loss = float("inf")
        for step, batch in enumerate(loader, 1):
            batch = move_tensors(batch, device)
            with autocast(args):
                et_feature = et_model(batch["event_voxel"])
                guide = epaw_model(batch["images"], batch["warped_events"], et_feature)[0]
                loss = epaw_guidance_loss(guide, batch["gradient_target"], args.edge_loss_weight)
            if not tensors_are_finite(guide, batch["gradient_target"], loss):
                if is_main_process():
                    log_nonfinite(logger, args, epoch, step, batch, guide, loss, "forward_values")
                continue
            if not backward_step(loss, optimizer, scaler, epaw_model, args.grad_clip):
                if is_main_process():
                    message = f"stage2 epoch={epoch:03d} iter={step:05d} skipped: non-finite gradients"
                    print(message)
                    logger.write_text(message)
                continue
            global_step += 1
            loss_value = loss.item()
            interval_total += loss_value
            interval_steps += 1
            epoch_total += loss_value
            epoch_steps += 1
            metrics.update(
                flatten_time(guide),
                flatten_time(batch["gradient_target"]),
                include_lpips=step == 1 or step % args.print_every == 0,
            )
            epoch_metrics.update(
                flatten_time(guide.detach()),
                flatten_time(batch["gradient_target"]),
            )
            if step % log_every == 0:
                interval_loss = interval_total / max(interval_steps, 1)
                best_interval_loss = min(best_interval_loss, interval_loss)
                report_interval("stage2", epoch, step, interval_loss, metrics, optimizer, sync=True)
                if is_main_process():
                    logger.log_interval(epoch, step, global_step, interval_loss, current_lr(optimizer), metrics.summary())
                interval_total, interval_steps = 0.0, 0
                metrics.reset()
            if args.max_steps and step >= args.max_steps:
                break
        if interval_steps:
            interval_loss = interval_total / interval_steps
            best_interval_loss = min(best_interval_loss, interval_loss)
            report_interval("stage2", epoch, step, interval_loss, metrics, optimizer, sync=True)
            if is_main_process():
                logger.log_interval(epoch, step, global_step, interval_loss, current_lr(optimizer), metrics.summary())
        epoch_avg_loss = epoch_total / max(epoch_steps, 1)
        if is_main_process():
            logger.log_epoch(epoch, epoch_avg_loss, best_interval_loss, current_lr(optimizer), epoch_metrics.summary())
        scheduler.step()
        if is_main_process():
            state = dict(
                epoch=epoch,
                et_model=et_model.state_dict(),
                epaw_model=unwrap_model(epaw_model).state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                args=vars(args),
            )
            save_checkpoint(args.output, **state)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                save_checkpoint(checkpoint_path(args.output, epoch), **state)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            del metrics, epoch_metrics
            torch.cuda.empty_cache()
            barrier()
            validate(args, epoch, et_model, epaw_model, device, logger)
            barrier()
    if is_main_process():
        logger.write_text("======== training finished ========")
        logger.plot_curves()
    cleanup_training()


if __name__ == "__main__":
    main()
