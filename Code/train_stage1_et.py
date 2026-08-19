"""Stage 1: train ETStableNet with scalar motion-field supervision and logs."""

import argparse
import csv
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
logging.getLogger("mmcv").setLevel(logging.ERROR)

from Model.ETStableNet import ETStableNet
from Data.efturb_stage1_dataset import (
    DEFAULT_EFTURB_ROOT, make_multiframe_flow_loader,
)
from Utils.train_utils import (
    FlowMetrics, add_common_args, autocast, backward_step, barrier,
    checkpoint_path, cleanup_training, is_main_process, load_state, make_loader,
    make_optimizer_and_scheduler, maybe_resume, move_tensors,
    report_interval, save_checkpoint, save_stage1_preview, set_loader_epoch,
    stage1_sharpening_loss, stage1_structure_fidelity_loss,
    setup_training, tensors_are_finite, unwrap_model, wrap_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def flatten_time(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        b, t = tensor.shape[:2]
        return tensor.reshape(b * t, *tensor.shape[2:])
    return tensor


def center_frame(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        return tensor[:, tensor.shape[1] // 2]
    return tensor


def predict_multiframe(model, event_voxel):
    return model(event_voxel)


def stage1_multiframe_loss(prediction, target, args):
    prediction = flatten_time(prediction)
    target = flatten_time(target)
    if args.stage1_loss == "legacy":
        loss = stage1_sharpening_loss(
            prediction,
            target,
            foreground_weight=args.foreground_weight,
            sobel_weight=args.sobel_weight,
            ssim_weight=args.ssim_weight,
        )
        return loss, {"LEGACY_LOSS": loss.detach()}
    return stage1_structure_fidelity_loss(
        prediction,
        target,
        pixel_weight=args.pixel_weight,
        multiscale_weight=args.multiscale_weight,
        sobel_weight=args.sobel_weight,
        ssim_weight=args.ssim_weight,
        return_components=True,
    )


def add_loss_components(total, components):
    for key, value in components.items():
        total[key] = total.get(key, 0.0) + float(value.item())


def loss_components_text(total, steps):
    if not total or steps <= 0:
        return ""
    return " ".join(f"{key}={value / steps:.6f}" for key, value in sorted(total.items()))


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0


def parse_metric_summary(summary):
    """Parse strings such as 'epe=0.123 mae=0.045' into a numeric dict."""
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
            min_value = values.min().item()
            max_value = values.max().item()
            mean_value = values.float().mean().item()
            return (
                f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
                f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} "
                f"min={min_value:.6g} max={max_value:.6g} mean={mean_value:.6g}"
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


def save_bad_batch(args, epoch, step, batch, prediction, loss, reason):
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
        "prediction": prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else None,
    }
    for key in ("event_voxel", "flow_target"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            payload[key] = batch[key].detach().cpu()
    for key in ("sequence", "center", "index"):
        if key in batch:
            payload[key] = batch[key]
    torch.save(payload, path)
    return path


def log_nonfinite(logger, args, epoch, step, batch, prediction, loss, reason):
    if not is_main_process():
        return
    lines = [
        f"NON_FINITE reason={reason} epoch={epoch:03d} iter={step:05d} {batch_meta(batch)}",
        tensor_finite_stats("event_voxel", batch.get("event_voxel")),
        tensor_finite_stats("flow_target", batch.get("flow_target")),
        tensor_finite_stats("prediction", prediction),
        tensor_finite_stats("loss", loss),
    ]
    bad_path = save_bad_batch(args, epoch, step, batch, prediction, loss, reason)
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
def validate(args, epoch, model, device, logger=None):
    if not is_main_process():
        return None
    torch.cuda.empty_cache()
    loader = make_multiframe_flow_loader(
        args,
        "test",
        load_events=False,
        load_gt=False,
        shuffle=False,
        crop_size=256,
        frames=5,
        batch_size=1,
    )
    metrics = FlowMetrics(device)
    model = unwrap_model(model).eval()
    total_loss, steps = 0.0, 0
    component_total = {}
    for step, batch in enumerate(loader, 1):
        batch = move_tensors(batch, device)
        with autocast(args):
            prediction = predict_multiframe(model, batch["event_voxel"])
            loss, loss_parts = stage1_multiframe_loss(prediction, batch["flow_target"], args)
        total_loss += loss.item()
        steps += 1
        add_loss_components(component_total, loss_parts)
        metrics.update(flatten_time(prediction), flatten_time(batch["flow_target"]))
        if step <= 4:
            save_stage1_preview(
                center_frame(prediction),
                center_frame(batch["flow_target"]),
                Path(args.results_dir) / f"epoch_{epoch:03d}" / f"{batch['sequence'][0]}_{batch['center'][0]:05d}.png",
            )
        if args.validation_samples and step >= args.validation_samples:
            break
    val_loss = total_loss / max(steps, 1)
    summary = metrics.summary()
    component_summary = loss_components_text(component_total, steps)
    if component_summary:
        summary = f"{summary} {component_summary}"
    print(f"stage1 validation epoch={epoch:03d} val_loss={val_loss:.6f} {summary}")
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
    parser.add_argument("--output", default=str(PROJECT_ROOT / "checkpoints" / "stage1_et.pt"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "outputs" / "validation" / "stage1"))
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "outputs" / "logs" / "stage1"))
    parser.add_argument("--log-every", type=int, default=0, help="0 uses --print-every")
    parser.add_argument(
        "--stage1-loss",
        choices=["structure-fidelity", "legacy"],
        default="structure-fidelity",
        help="Dense EFTurb structure-fidelity loss or the old sparse CTTH-oriented loss.",
    )
    parser.add_argument("--pixel-weight", type=float, default=1.0, help="Full-resolution MSE weight")
    parser.add_argument(
        "--multiscale-weight",
        type=float,
        default=0.5,
        help="Weight for average-pooled 2x/4x/8x global-structure MSE",
    )
    parser.set_defaults(sobel_weight=0.05, ssim_weight=0.02)
    parser.add_argument(
        "--pretrained",
        default="",
        help=(
            "Load only et_model weights from a Stage 1 checkpoint. Training starts "
            "at epoch 1 with a new optimizer, scheduler, and AMP scaler."
        ),
    )
    args = parser.parse_args()
    if args.pretrained and args.resume:
        parser.error("--pretrained and --resume are mutually exclusive")
    device = setup_training(args)
    logger = TrainingLogger(args.log_dir, args) if is_main_process() else None
    if min(args.pixel_weight, args.multiscale_weight, args.sobel_weight, args.ssim_weight) < 0:
        parser.error("Stage 1 loss weights must be non-negative")
    model = wrap_model(ETStableNet(), device)
    optimizer, scheduler, scaler = make_optimizer_and_scheduler(model, args)
    if args.pretrained:
        load_state(model, args.pretrained, "et_model", device)
        start_epoch = 1
        if is_main_process():
            message = (
                f"loaded pretrained ET weights only from {args.pretrained}; "
                "starting at epoch 1 with fresh optimizer/scheduler/scaler"
            )
            print(message)
            logger.write_text(message)
    else:
        start_epoch = maybe_resume(
            args, {"et_model": model}, optimizer, scheduler, scaler, device
        )
    loader = make_multiframe_flow_loader(
        args,
        "train",
        load_events=False,
        load_gt=False,
        crop_size=256,
        frames=5,
        batch_size=1,
    )
    log_every = args.log_every if args.log_every > 0 else args.print_every
    if is_main_process():
        message = (
            f"stage1 multi-frame training: epochs={args.epochs} crop=256 batch=1 frames=5 "
            f"loss={args.stage1_loss} pixel={args.pixel_weight} multiscale={args.multiscale_weight} "
            f"sobel={args.sobel_weight} ssim={args.ssim_weight} log_every={log_every}"
        )
        print(message)
        logger.write_text(message)
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        set_loader_epoch(loader, epoch)
        model.train()
        metrics = FlowMetrics(device, use_lpips=is_main_process())
        epoch_metrics = FlowMetrics(device, use_lpips=False)
        interval_total, interval_steps = 0.0, 0
        epoch_total, epoch_steps = 0.0, 0
        interval_components, epoch_components = {}, {}
        best_interval_loss = float("inf")
        for step, batch in enumerate(loader, 1):
            batch = move_tensors(batch, device)
            with autocast(args):
                prediction = predict_multiframe(model, batch["event_voxel"])
                loss, loss_parts = stage1_multiframe_loss(prediction, batch["flow_target"], args)
            if not tensors_are_finite(prediction, batch["flow_target"], loss):
                if is_main_process():
                    log_nonfinite(logger, args, epoch, step, batch, prediction, loss, "forward_values")
                continue
            if not backward_step(loss, optimizer, scaler, model, args.grad_clip):
                if is_main_process():
                    message = f"stage1 epoch={epoch:03d} iter={step:05d} skipped: non-finite gradients"
                    print(message)
                    logger.write_text(message)
                continue

            global_step += 1
            loss_value = loss.item()
            interval_total += loss_value
            interval_steps += 1
            epoch_total += loss_value
            epoch_steps += 1
            add_loss_components(interval_components, loss_parts)
            add_loss_components(epoch_components, loss_parts)
            metrics.update(
                flatten_time(prediction),
                flatten_time(batch["flow_target"]),
                include_lpips=step == 1 or step % args.print_every == 0,
            )
            epoch_metrics.update(flatten_time(prediction.detach()), flatten_time(batch["flow_target"]))
            if step % log_every == 0:
                interval_loss = interval_total / max(interval_steps, 1)
                best_interval_loss = min(best_interval_loss, interval_loss)
                component_summary = loss_components_text(interval_components, interval_steps)
                report_interval(
                    "stage1", epoch, step, interval_loss, metrics, optimizer,
                    sync=True, extra=component_summary,
                )
                if is_main_process():
                    summary = metrics.summary()
                    if component_summary:
                        summary = f"{summary} {component_summary}"
                    logger.log_interval(epoch, step, global_step, interval_loss, current_lr(optimizer), summary)
                interval_total, interval_steps = 0.0, 0
                interval_components = {}
                metrics.reset()
            if args.max_steps and step >= args.max_steps:
                break
        if interval_steps:
            interval_loss = interval_total / interval_steps
            best_interval_loss = min(best_interval_loss, interval_loss)
            component_summary = loss_components_text(interval_components, interval_steps)
            report_interval(
                "stage1", epoch, step, interval_loss, metrics, optimizer,
                sync=True, extra=component_summary,
            )
            if is_main_process():
                summary = metrics.summary()
                if component_summary:
                    summary = f"{summary} {component_summary}"
                logger.log_interval(epoch, step, global_step, interval_loss, current_lr(optimizer), summary)
        epoch_avg_loss = epoch_total / max(epoch_steps, 1)
        if is_main_process():
            summary = epoch_metrics.summary()
            component_summary = loss_components_text(epoch_components, epoch_steps)
            if component_summary:
                summary = f"{summary} {component_summary}"
            logger.log_epoch(epoch, epoch_avg_loss, best_interval_loss, current_lr(optimizer), summary)
        scheduler.step()
        if is_main_process():
            state = dict(epoch=epoch, et_model=unwrap_model(model).state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), args=vars(args))
            save_checkpoint(args.output, **state)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                save_checkpoint(checkpoint_path(args.output, epoch), **state)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            del metrics, epoch_metrics
            torch.cuda.empty_cache()
            barrier()
            validate(args, epoch, model, device, logger)
            barrier()
    if is_main_process():
        logger.write_text("======== training finished ========")
        logger.plot_curves()
    cleanup_training()


if __name__ == "__main__":
    main()
