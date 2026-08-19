"""Stage 3: ET/EPAW event-guided fine-tuning of Ref_MambaTM.

This script is intended for the transition:

  gradient-pretrained restoration -> frozen ET/EPAW guided restoration fine-tune

Use --pretrained (or --restoration-pretrained) for the first fine-tuning run. Use
--finetune-resume to continue an interrupted event-guided fine-tuning run
with model, optimizer, scheduler, scaler, and epoch restored.
"""
import warnings
import argparse
import csv
import json
import re
import time
from pathlib import Path
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from Data.efturb_dataset import DEFAULT_EFTURB_ROOT
from Data.stage3_finetune_dataset import (
    DEFAULT_CTTH_ROOT,
    DEFAULT_INDEX_DIR,
    JointStage3FineTuneDataset,
    SequenceChunkSampler,
    Stage3FineTuneDataset,
    collate_stage3_finetune,
    stage3_worker_init,
)
from Model.EPAWStableNet_new import EPAWStableNet
from Model.ETStableNet import ETStableNet
from Model.Ref_MambaTM import build_restoration_model
from Utils.train_utils import (
    ImageMetrics, add_common_args, autocast, backward_step, barrier, charbonnier,
    checkpoint_path, cleanup_training, is_main_process,
    make_optimizer_and_scheduler, move_tensors,
    report_interval, save_checkpoint, save_preview, set_loader_epoch,
    setup_training, tensors_are_finite, unwrap_model, wrap_model,
)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


PROJECT_ROOT = Path(__file__).resolve().parent


def make_domain_dataset(args, domain, split, crop_size, frames, load_events):
    root = args.efturb_root if domain == "efturb" else args.ctth_root
    domain_count = 2 if args.finetune_dataset == "joint" else 1
    return Stage3FineTuneDataset(
        root=root,
        domain=domain,
        split=split,
        crop_size=crop_size,
        frames=frames,
        voxel_scale=args.voxel_scale,
        load_events=load_events,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        noise_std=args.noise_std if split == "train" else 0.0,
        image_cache_mb=args.image_cache_mb / domain_count,
        event_cache_mb=args.event_cache_mb / domain_count,
        index_dir=args.dataset_index_dir,
        rebuild_index=args.rebuild_data_index,
    )


def make_multiframe_restoration_loader(
    args,
    split,
    load_events=True,
    shuffle=None,
    crop_size=None,
    frames=None,
    batch_size=None,
    dataset_mode=None,
):
    mode = dataset_mode or args.finetune_dataset
    domains = ("efturb", "ctth") if mode == "joint" else (mode,)
    datasets = [
        make_domain_dataset(
            args,
            domain,
            split,
            crop_size or args.crop_size,
            frames or args.frames,
            load_events,
        )
        for domain in domains
    ]
    dataset = (
        datasets[0]
        if len(datasets) == 1
        else JointStage3FineTuneDataset(datasets)
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        sizes = ", ".join(
            f"{domain}={len(domain_dataset)}"
            for domain, domain_dataset in zip(domains, datasets)
        )
        print(f"[INFO] fine-tune dataset split={split} mode={mode}: {sizes}", flush=True)
    use_shuffle = split == "train" if shuffle is None else shuffle
    sampler = None
    if split == "train":
        sampler = SequenceChunkSampler(
            dataset,
            chunk_size=args.sequence_chunk_size,
            shuffle=use_shuffle,
            seed=args.seed,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
        )
    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=use_shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_stage3_finetune,
        pin_memory=True,
        drop_last=split == "train",
        persistent_workers=args.workers > 0,
        timeout=args.data_loader_timeout if args.workers > 0 else 0,
        worker_init_fn=stage3_worker_init,
        **loader_kwargs,
    )


def train_multiframe_restoration_loader_for_epoch(args, epoch, **kwargs):
    return make_multiframe_restoration_loader(
        args,
        "train",
        crop_size=args.train_crop_size,
        frames=args.train_frames,
        batch_size=args.train_batch_size,
        **kwargs,
    )


def flatten_time(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        b, t = tensor.shape[:2]
        return tensor.reshape(b * t, *tensor.shape[2:])
    return tensor


def center_frame(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        return tensor[:, tensor.shape[1] // 2]
    return tensor


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
        self._write_header(
            self.interval_csv,
            ["time", "epoch", "iter", "global_step", "loss", "lr", "metrics_json", "metrics_text"],
        )
        self._write_header(
            self.epoch_csv,
            ["time", "epoch", "avg_loss", "best_interval_loss", "best_train_loss", "lr", "metrics_json", "metrics_text"],
        )
        self._write_header(
            self.val_csv,
            ["time", "epoch", "val_loss", "best_val_loss", "metrics_json", "metrics_text"],
        )
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


def guide_from_epaw(batch, et_model, epaw_model):
    with torch.no_grad():
        event_voxel = batch["event_voxel"]
        if event_voxel.ndim == 6:
            event_voxel = center_frame(event_voxel)
        et_feature = et_model(event_voxel)
        if et_feature.ndim == 5:
            et_feature = center_frame(et_feature)
        guide = epaw_model(batch["images"], batch["warped_events"], et_feature)[0]
        if guide.ndim == 4:
            guide = guide.unsqueeze(1).expand(-1, batch["images"].shape[1], -1, -1, -1)
    return guide


def restore(batch, restoration, et_model, epaw_model):
    guide_sequence = guide_from_epaw(batch, et_model, epaw_model)
    return restoration(batch["images"], guide_sequence), guide_sequence


@torch.no_grad()
def validate(
    args,
    epoch,
    et_model,
    epaw_model,
    restoration,
    device,
    logger=None,
    dataset_mode=None,
):
    if not is_main_process():
        return
    torch.cuda.empty_cache()
    loader = make_multiframe_restoration_loader(
        args,
        "test",
        load_events=True,
        shuffle=False,
        crop_size=args.eval_crop_size,
        frames=args.eval_frames,
        batch_size=args.eval_batch_size,
        dataset_mode=dataset_mode,
    )
    validation_domain = dataset_mode or args.finetune_dataset
    metrics = ImageMetrics(device)
    if et_model is not None:
        et_model = unwrap_model(et_model).eval()
    if epaw_model is not None:
        epaw_model = unwrap_model(epaw_model).eval()
    restoration = unwrap_model(restoration).eval()
    total_loss, steps = 0.0, 0
    for step, batch in enumerate(loader, 1):
        batch = move_tensors(batch, device)
        restored, guide = restore(batch, restoration, et_model, epaw_model)
        loss = charbonnier(restored, batch["gt"])
        total_loss += loss.item()
        steps += 1
        metrics.update(flatten_time(restored), flatten_time(batch["gt"]))
        if step <= 4:
            base_path = (
                Path(args.results_dir)
                / validation_domain
                / f"epoch_{epoch:03d}"
                / f"{batch['sequence'][0]}_{batch['center'][0]:05d}.png"
            )
            save_preview(center_frame(restored), base_path)
            save_preview(center_frame(guide), base_path.with_name(f"{base_path.stem}_guide{base_path.suffix}"))
        if args.validation_samples and step >= args.validation_samples:
            break
    val_loss = total_loss / max(steps, 1)
    summary = metrics.summary()
    print(
        f"stage3_Finetuning validation dataset={validation_domain} "
        f"epoch={epoch:03d} val_loss={val_loss:.6f} {summary}"
    )
    if logger is not None:
        logger.log_validation(
            epoch,
            val_loss,
            f"DATASET={validation_domain} {summary}",
        )
    del metrics, loader
    torch.cuda.empty_cache()
    return val_loss


def load_guidance_stack(checkpoint, device, source):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{source} is not a checkpoint dictionary")
    missing = [key for key in ("et_model", "epaw_model") if key not in checkpoint]
    if missing:
        raise KeyError(f"{source} is missing guidance weights: {missing}")
    et_model = ETStableNet().to(device).eval()
    epaw_model = EPAWStableNet().to(device).eval()
    et_model.load_state_dict(checkpoint["et_model"])
    epaw_model.load_state_dict(checkpoint["epaw_model"])
    et_model.requires_grad_(False)
    epaw_model.requires_grad_(False)
    return et_model, epaw_model


def load_guidance_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return load_guidance_stack(checkpoint, device, checkpoint_path)


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def extract_restoration_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("restoration_model", "model", "state_dict", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(
        "Could not find restoration weights in checkpoint. Expected one of: "
        "restoration_model, model, state_dict, net, network."
    )


def load_restoration_pretrained(restoration, checkpoint_path, device, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(extract_restoration_state(checkpoint))
    missing, unexpected = unwrap_model(restoration).load_state_dict(state_dict, strict=strict)
    return missing, unexpected


def load_finetune_resume(restoration, optimizer, scheduler, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing = [
        key for key in ("et_model", "epaw_model", "restoration_model", "optimizer")
        if key not in checkpoint
    ]
    if missing:
        raise KeyError(f"{checkpoint_path} is not a complete fine-tune checkpoint; missing {missing}")
    state_dict = strip_module_prefix(extract_restoration_state(checkpoint))
    unwrap_model(restoration).load_state_dict(state_dict, strict=True)

    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = int(checkpoint.get("epoch", 0))
    return epoch + 1, checkpoint


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.set_defaults(data_root=str(DEFAULT_EFTURB_ROOT), workers=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--epaw-checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "stage2_epaw.pt"),
        help="Stage 2 checkpoint containing et_model and epaw_model; used only for a fresh fine-tune.",
    )
    parser.add_argument(
        "--restoration-pretrained",
        "--pretrained",
        dest="restoration_pretrained",
        default=None,
        help="Restoration pretraining checkpoint. Only restoration model weights are loaded.",
    )
    parser.add_argument(
        "--finetune-resume",
        default=None,
        help="Event-guided fine-tuning checkpoint to resume, e.g. restoration_event_guided_finetune_epoch_003.pt.",
    )
    parser.add_argument(
        "--strict-pretrained",
        action="store_true",
        help="Require exact restoration checkpoint key matching.",
    )
    parser.add_argument(
        "--finetune-dataset",
        "--dataset-mode",
        dest="finetune_dataset",
        choices=["ctth", "efturb", "joint"],
        default="efturb",
        help="Fine-tune on CTTH+ only, EFTurb only, or the complete 90%% train split of both.",
    )
    parser.add_argument(
        "--efturb-root",
        default=None,
        help="EFTurb root. By default this reuses --data-root for backward compatibility.",
    )
    parser.add_argument("--ctth-root", default=str(DEFAULT_CTTH_ROOT))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "checkpoints" / "stage3_finetune.pt"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "outputs" / "validation" / "stage3_finetune"))
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "outputs" / "logs" / "stage3_finetune"))
    parser.add_argument("--log-every", type=int, default=0, help="0 uses --print-every")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches queued per worker; 2 matches the stable Stage 1/2 loader default.",
    )
    parser.add_argument(
        "--data-loader-timeout",
        type=float,
        default=0,
        help=(
            "Seconds to wait for a DataLoader batch before raising an explicit "
            "timeout instead of appearing frozen; 0 waits indefinitely."
        ),
    )
    parser.add_argument(
        "--sequence-chunk-size",
        type=int,
        default=8,
        help=(
            "Shuffle short runs of neighbouring windows instead of every frame "
            "independently to improve decoded-data locality."
        ),
    )
    parser.add_argument(
        "--dataset-index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help=(
            "Small persistent path manifest directory. This stores metadata only "
            "and avoids rescanning every dataset folder after a restart."
        ),
    )
    parser.add_argument(
        "--rebuild-data-index",
        action="store_true",
        help="Ignore saved path manifests and rescan the selected dataset.",
    )
    parser.add_argument(
        "--image-cache-mb",
        type=float,
        default=64.0,
        help="Maximum decoded grayscale-image RAM cache per worker (0 disables).",
    )
    parser.add_argument(
        "--event-cache-mb",
        type=float,
        default=128.0,
        help="Maximum decoded raw-event RAM cache per worker (0 disables).",
    )
    parser.add_argument("--train-crop-size", type=int, default=512, help="Stage3 training crop size")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Stage3 training batch size per process")
    parser.add_argument("--train-frames", type=int, default=5, help="Number of frames used for Stage3 training")
    parser.add_argument("--eval-crop-size", type=int, default=512, help="Stage3 validation/inference crop size")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Stage3 validation/inference batch size")
    parser.add_argument("--eval-frames", type=int, default=5, help="Number of frames used for Stage3 validation/inference")
    args = parser.parse_args()
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive")
    if args.data_loader_timeout < 0:
        parser.error("--data-loader-timeout must be non-negative")
    if args.sequence_chunk_size < 1:
        parser.error("--sequence-chunk-size must be positive")
    if min(args.image_cache_mb, args.event_cache_mb) < 0:
        parser.error("cache sizes must be non-negative")
    if args.efturb_root is None:
        args.efturb_root = args.data_root
    if args.restoration_pretrained and args.finetune_resume:
        parser.error(
            "--pretrained/--restoration-pretrained and --finetune-resume are mutually exclusive"
        )
    if args.restoration_pretrained and args.resume:
        parser.error("--pretrained/--restoration-pretrained and --resume are mutually exclusive")
    if args.resume:
        parser.error("This script uses --finetune-resume, not the common --resume option")
    device = setup_training(args)
    logger = TrainingLogger(args.log_dir, args) if is_main_process() else None

    restoration = wrap_model(build_restoration_model(), device)
    optimizer, scheduler, scaler = make_optimizer_and_scheduler(restoration, args)
    if args.finetune_resume:
        start_epoch, resume_state = load_finetune_resume(
            restoration,
            optimizer,
            scheduler,
            scaler,
            args.finetune_resume,
            device,
        )
        et_model, epaw_model = load_guidance_stack(
            resume_state,
            device,
            args.finetune_resume,
        )
        global_step = int(resume_state.get("global_step", 0))
        if is_main_process():
            resume_lr = current_lr(optimizer)
            message = (
                f"resumed event-guided fine-tuning from {args.finetune_resume}; "
                f"next_epoch={start_epoch} lr={resume_lr:.8e}"
            )
            print(message)
            logger.write_text(message)
    else:
        if not args.restoration_pretrained:
            raise ValueError(
                "Provide --pretrained (or --restoration-pretrained) for a fresh fine-tune, "
                "or --finetune-resume to continue."
            )
        et_model, epaw_model = load_guidance_checkpoint(args.epaw_checkpoint, device)
        missing, unexpected = load_restoration_pretrained(
            restoration,
            args.restoration_pretrained,
            device,
            strict=args.strict_pretrained,
        )
        start_epoch = 1
        global_step = 0
        if is_main_process():
            message = (
                "loaded restoration pretrained weights only; optimizer/scheduler/scaler are restarted "
                f"from lr={args.lr:.8e}; checkpoint={args.restoration_pretrained}"
            )
            print(message)
            logger.write_text(message)
            if missing:
                logger.write_text(f"pretrained missing keys: {missing}")
                print(f"[WARN] pretrained missing keys: {len(missing)}")
            if unexpected:
                logger.write_text(f"pretrained unexpected keys: {unexpected}")
                print(f"[WARN] pretrained unexpected keys: {len(unexpected)}")
    loader = None
    previous_loader_key = None
    log_every = args.log_every if args.log_every > 0 else args.print_every
    metrics = ImageMetrics(device, use_lpips=is_main_process())
    epoch_metrics = ImageMetrics(device, use_lpips=False)
    for epoch in range(start_epoch, args.epochs + 1):
        loader_key = (
            args.finetune_dataset,
            args.efturb_root,
            args.ctth_root,
            args.train_crop_size,
            args.train_batch_size,
            args.train_frames,
        )
        if loader_key != previous_loader_key:
            loader = train_multiframe_restoration_loader_for_epoch(
                args,
                epoch,
                load_events=True,
            )
            previous_loader_key = loader_key
            if is_main_process():
                message = (
                    f"stage3_Finetuning settings: guide=ET/EPAW "
                    f"dataset={args.finetune_dataset} epochs={start_epoch}-{args.epochs} "
                    f"train_crop={args.train_crop_size} train_batch={args.train_batch_size} "
                    f"train_frames={args.train_frames} eval_crop={args.eval_crop_size} "
                    f"eval_batch={args.eval_batch_size} eval_frames={args.eval_frames} "
                    f"log_every={log_every}"
                )
                print(message)
                logger.write_text(message)
        set_loader_epoch(loader, epoch)
        restoration.train()
        metrics.reset()
        epoch_metrics.reset()
        interval_total, interval_steps = 0.0, 0
        epoch_total, epoch_steps = 0.0, 0
        best_interval_loss = float("inf")
        for step, batch in enumerate(loader, 1):
            batch = move_tensors(batch, device)
            with autocast(args):
                restored, guide = restore(batch, restoration, et_model, epaw_model)
                loss = charbonnier(restored, batch["gt"])
            if not tensors_are_finite(restored, batch["gt"], guide, loss):
                if is_main_process():
                    message = f"stage3_Finetuning epoch={epoch:03d} iter={step:05d} skipped: non-finite forward values"
                    print(message)
                    logger.write_text(message)
                continue
            if not backward_step(loss, optimizer, scaler, restoration, args.grad_clip):
                if is_main_process():
                    message = f"stage3_Finetuning epoch={epoch:03d} iter={step:05d} skipped: non-finite gradients"
                    print(message)
                    logger.write_text(message)
                continue
            global_step += 1
            loss_value = loss.item()
            interval_total, interval_steps = interval_total + loss_value, interval_steps + 1
            epoch_total, epoch_steps = epoch_total + loss_value, epoch_steps + 1
            previous_psnr = metrics.total["psnr"]
            previous_ssim = metrics.total["ssim"]
            metrics.update(
                flatten_time(restored),
                flatten_time(batch["gt"]),
                include_lpips=step == 1 or step % args.print_every == 0,
            )
            # Reuse the already-computed PSNR/SSIM instead of running both
            # metrics a second time for the epoch accumulator.
            epoch_metrics.total["psnr"] += metrics.total["psnr"] - previous_psnr
            epoch_metrics.total["ssim"] += metrics.total["ssim"] - previous_ssim
            epoch_metrics.count += 1
            if step % log_every == 0:
                interval_loss = interval_total / max(interval_steps, 1)
                best_interval_loss = min(best_interval_loss, interval_loss)
                report_interval("stage3_Finetuning", epoch, step, interval_loss, metrics, optimizer, sync=True)
                if is_main_process():
                    logger.log_interval(
                        epoch,
                        step,
                        global_step,
                        interval_loss,
                        current_lr(optimizer),
                        metrics.summary(),
                    )
                interval_total, interval_steps = 0.0, 0
                metrics.reset()
            if args.max_steps and step >= args.max_steps:
                break
        if interval_steps:
            interval_loss = interval_total / interval_steps
            best_interval_loss = min(best_interval_loss, interval_loss)
            report_interval("stage3_Finetuning", epoch, step, interval_loss, metrics, optimizer, sync=True)
            if is_main_process():
                logger.log_interval(epoch, step, global_step, interval_loss, current_lr(optimizer), metrics.summary())
        epoch_avg_loss = epoch_total / max(epoch_steps, 1)
        if is_main_process():
            logger.log_epoch(epoch, epoch_avg_loss, best_interval_loss, current_lr(optimizer), epoch_metrics.summary())
        scheduler.step()
        if is_main_process():
            state = dict(
                epoch=epoch,
                restoration_model=unwrap_model(restoration).state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                args=vars(args),
                restoration_pretrained=args.restoration_pretrained,
                finetune_resume=args.finetune_resume,
                global_step=global_step,
            )
            state["et_model"] = et_model.state_dict()
            state["epaw_model"] = epaw_model.state_dict()
            save_checkpoint(args.output, **state)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                save_checkpoint(checkpoint_path(args.output, epoch), **state)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            torch.cuda.empty_cache()
            barrier()
            validation_domains = (
                ("efturb", "ctth")
                if args.finetune_dataset == "joint"
                else (args.finetune_dataset,)
            )
            for validation_domain in validation_domains:
                validate(
                    args,
                    epoch,
                    et_model,
                    epaw_model,
                    restoration,
                    device,
                    logger,
                    dataset_mode=validation_domain,
                )
            barrier()
    if is_main_process():
        logger.write_text("======== training finished ========")
        logger.plot_curves()
    cleanup_training()


if __name__ == "__main__":
    main()
