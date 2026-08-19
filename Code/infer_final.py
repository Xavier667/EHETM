"""Unified final inference for EFTurb, CTTH+ and real event-camera data.

Only one complete ``restoration_event_guided_finetune_epoch_***.pt`` checkpoint
is required.  The checkpoint must contain ET, EPAW and restoration weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

from Model.EPAWStableNet_new import EPAWStableNet
from Model.ETStableNet import ETStableNet
from Model.Ref_MambaTM import build_restoration_model
from Utils.Metric import ssim_pytorch
from Utils.train_utils import device_from_arg


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def strip_module_prefix(state_dict):
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def checkpoint_args(checkpoint):
    value = checkpoint.get("args", {})
    if isinstance(value, dict):
        return value
    try:
        return vars(value)
    except TypeError:
        return {}


def load_complete_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{checkpoint_path} is not a checkpoint dictionary")
    required = ("et_model", "epaw_model", "restoration_model")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            f"{checkpoint_path} is not a complete fine-tune checkpoint; "
            f"missing weights: {missing}"
        )

    et_model = ETStableNet().to(device)
    epaw_model = EPAWStableNet().to(device)
    restoration = build_restoration_model().to(device)
    et_model.load_state_dict(strip_module_prefix(checkpoint["et_model"]), strict=True)
    epaw_model.load_state_dict(strip_module_prefix(checkpoint["epaw_model"]), strict=True)
    restoration.load_state_dict(
        strip_module_prefix(checkpoint["restoration_model"]), strict=True
    )
    for model in (et_model, epaw_model, restoration):
        model.requires_grad_(False)
        model.eval()
    return checkpoint, et_model, epaw_model, restoration


def numeric_images(folder):
    folder = Path(folder)
    return {
        int(path.stem): path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem.isdigit()
    } if folder.is_dir() else {}


def numeric_npz(folder):
    folder = Path(folder)
    return {
        int(path.stem): path
        for path in folder.glob("*.npz")
        if path.stem.isdigit()
    } if folder.is_dir() else {}


def first_dir(sequence, candidates):
    for relative in candidates:
        path = sequence / relative
        if path.is_dir():
            return path
    return None


def sequence_layout(sequence, data_format):
    if data_format == "efturb":
        frame_candidates = ("Turb/frames",)
        gt_candidates = ("GT/frames",)
    elif data_format == "ctth":
        frame_candidates = ("Turb/frames_gray",)
        gt_candidates = ("GT/frames_gray",)
    else:
        # A real sequence may use the compact layout or either public layout.
        frame_candidates = (
            "frames", "images", "degraded", "Turb/frames", "Turb/frames_gray"
        )
        gt_candidates = (
            "GT/frames", "GT/frames_gray", "gt", "ground_truth"
        )

    return {
        "frames": first_dir(sequence, frame_candidates),
        "events": first_dir(sequence, ("events", "Turb/events")),
        "voxels": first_dir(sequence, ("event_voxel", "Turb/event_voxel")),
        "gt": first_dir(sequence, gt_candidates),
    }


def is_sequence(path, data_format):
    layout = sequence_layout(path, data_format)
    return all(layout[key] is not None for key in ("frames", "events", "voxels"))


def discover_sequences(root, data_format):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if is_sequence(root, data_format):
        return [root]
    # The two public datasets have fixed shallow layouts. Avoid recursively
    # probing their many frame/event folders, which is unnecessarily slow.
    if data_format == "efturb":
        candidates = (
            sequence
            for scenario in root.iterdir()
            if scenario.is_dir()
            for sequence in scenario.iterdir()
            if sequence.is_dir()
        )
    elif data_format == "ctth":
        candidates = (path for path in root.iterdir() if path.is_dir())
    else:
        candidates = (path for path in root.rglob("*") if path.is_dir())
    sequences = [path for path in candidates if is_sequence(path, data_format)]
    # A detected sequence cannot contain another independent sequence in the
    # supported layouts; sorting makes selection/output deterministic.
    return sorted(set(sequences))


def relative_name(sequence, root):
    try:
        relative = sequence.relative_to(root)
    except ValueError:
        relative = Path(sequence.name)
    if str(relative) == ".":
        relative = Path(sequence.name)
    return relative.as_posix(), "__".join(relative.parts)


def load_sequence_list(path):
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip().replace("\\", "/").strip("/")
        if value:
            entries.append(value)
    if not entries:
        raise ValueError(f"test sequence list is empty: {path}")
    return entries


def select_manifest_sequences(sequences, root, list_path):
    requested = load_sequence_list(list_path)
    aliases = {}
    for sequence in sequences:
        relative, sequence_id = relative_name(sequence, root)
        for alias in (relative, sequence_id, sequence.name):
            aliases.setdefault(alias, set()).add(sequence)

    selected = []
    unresolved = []
    ambiguous = []
    for value in requested:
        matches = aliases.get(value, set())
        if len(matches) == 1:
            selected.append(next(iter(matches)))
        elif not matches:
            unresolved.append(value)
        else:
            ambiguous.append(value)
    if unresolved or ambiguous:
        raise ValueError(
            f"invalid --test-list; unresolved={unresolved}, ambiguous={ambiguous}. "
            "Use paths relative to --data-root (for example Scenario/seq_001)."
        )
    return sorted(set(selected))


def split_public_sequences(sequences, root, data_format, split, seed, train_ratio):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train ratio must be between 0 and 1")
    if split == "all":
        return sorted(sequences)

    if data_format == "efturb":
        groups = {}
        for sequence in sequences:
            relative, _ = relative_name(sequence, root)
            scenario = relative.split("/", 1)[0]
            groups.setdefault(scenario, []).append(sequence)
        selected = []
        for scenario, group in sorted(groups.items()):
            group = sorted(group)
            random.Random(f"{seed}:{scenario}").shuffle(group)
            cut = round(len(group) * train_ratio)
            if len(group) > 1:
                cut = min(max(cut, 1), len(group) - 1)
            selected.extend(group[:cut] if split == "train" else group[cut:])
        return sorted(selected)

    group = sorted(sequences)
    random.Random(seed).shuffle(group)
    cut = round(len(group) * train_ratio)
    return sorted(group[:cut] if split == "train" else group[cut:])


def build_samples(sequences, root, data_format, frames, gt_mode):
    radius = frames // 2
    samples = []
    diagnostics = []
    for sequence in sequences:
        layout = sequence_layout(sequence, data_format)
        images = numeric_images(layout["frames"])
        events = numeric_npz(layout["events"])
        voxels = numeric_npz(layout["voxels"])
        gt_images = numeric_images(layout["gt"]) if layout["gt"] else {}
        relative, sequence_id = relative_name(sequence, root)
        accepted = 0
        for center in sorted(voxels):
            frame_ids = list(range(center - radius, center + (frames - radius)))
            if not all(index in images and index in events for index in frame_ids):
                continue
            gt_path = gt_images.get(center)
            gt_window = [gt_images.get(index) for index in frame_ids]
            if gt_mode == "required" and any(path is None for path in gt_window):
                continue
            if gt_mode == "none":
                gt_path = None
                gt_window = [None] * frames
            samples.append(
                {
                    "sequence": sequence_id,
                    "relative_sequence": relative,
                    "center": center,
                    "frame_ids": frame_ids,
                    "images": [images[index] for index in frame_ids],
                    "events": [events[index] for index in frame_ids],
                    "voxel": voxels[center],
                    "gt": gt_path,
                    "gt_window": gt_window,
                }
            )
            accepted += 1
        diagnostics.append(
            {
                "sequence": relative,
                "images": len(images),
                "events": len(events),
                "voxels": len(voxels),
                "gt": len(gt_images),
                "samples": accepted,
            }
        )
    if not samples:
        raise RuntimeError(
            "No valid inference windows were found. Each output center needs one "
            "event_voxel NPZ and a complete frame/event window. Diagnostics: "
            + json.dumps(diagnostics, ensure_ascii=False)
        )
    return samples, diagnostics


def select_temporal_samples(samples, temporal_mode):
    """Select sliding windows or a greedy set of frame-disjoint windows."""
    if temporal_mode == "sliding-center":
        return list(samples)
    selected = []
    last_frame_by_sequence = {}
    for sample in samples:
        sequence = sample["sequence"]
        first_frame = sample["frame_ids"][0]
        if first_frame <= last_frame_by_sequence.get(sequence, -math.inf):
            continue
        selected.append(sample)
        last_frame_by_sequence[sequence] = sample["frame_ids"][-1]
    return selected


def grayscale_array(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_voxel(path, key, scale):
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"missing {key!r} in {path}; available keys={data.files}")
        voxel = np.asarray(data[key], dtype=np.float32)
    if voxel.ndim == 5 and voxel.shape[0] == 1:
        voxel = voxel[0]
    if voxel.ndim != 4 or voxel.shape[:2] != (2, 10):
        raise ValueError(f"expected voxel [2,10,H,W], got {voxel.shape} in {path}")
    if not np.isfinite(voxel).all():
        raise ValueError(f"non-finite event voxel: {path}")
    voxel = np.sign(voxel) * np.log1p(np.abs(voxel)) / np.log1p(scale)
    return np.clip(voxel, -1.0, 1.0).astype(np.float32, copy=False)


def load_events(path, top, left, crop, height, width):
    with np.load(path) as data:
        missing = {"x", "y", "p", "t"} - set(data.files)
        if missing:
            raise KeyError(f"missing event keys {sorted(missing)} in {path}")
        x = np.asarray(data["x"])
        y = np.asarray(data["y"])
        p = np.asarray(data["p"])
        t = np.asarray(data["t"])
    if not (x.shape == y.shape == p.shape == t.shape):
        raise ValueError(f"event arrays have inconsistent shapes in {path}")
    bottom = min(top + crop, height)
    right = min(left + crop, width)
    valid = (x >= left) & (x < right) & (y >= top) & (y < bottom)
    x = x[valid].astype(np.float32) - left
    y = y[valid].astype(np.float32) - top
    polarity = p[valid].astype(np.float32)
    if polarity.size and polarity.min() >= 0 and polarity.max() <= 1:
        polarity = polarity * 2.0 - 1.0
    else:
        polarity = np.sign(polarity)
        polarity[polarity == 0] = 1.0
    timestamp = t[valid].astype(np.float32)
    if x.size == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.from_numpy(
        np.stack((x, y, polarity, timestamp), axis=1).astype(np.float32, copy=False)
    )


def pad_spatial(array, target_height, target_width, constant=False):
    pad_h = target_height - array.shape[-2]
    pad_w = target_width - array.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("target padding shape is smaller than the input")
    if pad_h == 0 and pad_w == 0:
        return array
    pads = [(0, 0)] * array.ndim
    pads[-2] = (0, pad_h)
    pads[-1] = (0, pad_w)
    if constant:
        return np.pad(array, pads, mode="constant")
    mode = "reflect" if array.shape[-2] > 1 and array.shape[-1] > 1 else "edge"
    return np.pad(array, pads, mode=mode)


def positions(length, tile, overlap):
    if tile >= length:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def blend_window(tile, device):
    if tile <= 2:
        return torch.ones(1, 1, tile, tile, device=device)
    axis = torch.hann_window(tile, periodic=False, device=device).clamp_min(0.05)
    return (axis[:, None] * axis[None, :]).view(1, 1, tile, tile)


@torch.inference_mode()
def infer_sample(
    sample,
    et_model,
    epaw_model,
    restoration,
    device,
    tile_size,
    overlap,
    voxel_scale,
    voxel_key,
    amp,
):
    images_full = [grayscale_array(path) for path in sample["images"]]
    height, width = images_full[0].shape
    if any(image.shape != (height, width) for image in images_full):
        raise ValueError(f"frame sizes differ in {sample['relative_sequence']}")
    voxel_full = load_voxel(sample["voxel"], voxel_key, voxel_scale)
    if voxel_full.shape[-2:] != (height, width):
        raise ValueError(
            f"voxel/image size mismatch in {sample['relative_sequence']} center="
            f"{sample['center']}: voxel={voxel_full.shape[-2:]}, image={(height, width)}"
        )

    padded_height = int(math.ceil(height / 4) * 4)
    padded_width = int(math.ceil(width / 4) * 4)
    images_full = [pad_spatial(image, padded_height, padded_width) for image in images_full]
    voxel_full = pad_spatial(voxel_full, padded_height, padded_width, constant=True)
    tile = min(tile_size, padded_height, padded_width)
    if tile % 4:
        raise ValueError(f"effective tile size must be divisible by 4, got {tile}")
    overlap = min(overlap, tile - 1)

    frame_count = len(sample["images"])
    output = torch.zeros(
        1, frame_count, 1, padded_height, padded_width, device=device
    )
    guide_output = torch.zeros_like(output)
    weight = torch.zeros(
        1, 1, 1, padded_height, padded_width, device=device
    )
    window_weight = blend_window(tile, device).unsqueeze(1)
    for top in positions(padded_height, tile, overlap):
        for left in positions(padded_width, tile, overlap):
            images = torch.stack(
                [
                    torch.from_numpy(
                        image[top : top + tile, left : left + tile].copy()
                    ).unsqueeze(0)
                    for image in images_full
                ]
            ).unsqueeze(0).to(device)
            voxel = torch.from_numpy(
                voxel_full[:, :, top : top + tile, left : left + tile].copy()
            ).unsqueeze(0).to(device)
            events = [[
                load_events(path, top, left, tile, height, width).to(device)
                for path in sample["events"]
            ]]

            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                et_feature = et_model(voxel)
                guide = epaw_model(images, events, et_feature)[0]
                if guide.ndim == 4:
                    guide = guide.unsqueeze(1).expand(
                        -1, images.shape[1], -1, -1, -1
                    )
                restored = restoration(images, guide)
            region = (..., slice(top, top + tile), slice(left, left + tile))
            output[region] += restored.float() * window_weight
            guide_output[region] += guide.float() * window_weight
            weight[region] += window_weight

    output = output.div(weight.clamp_min(1e-6))[..., :height, :width].clamp(0, 1)
    guide_output = guide_output.div(weight.clamp_min(1e-6))[..., :height, :width].clamp(0, 1)
    return output, guide_output


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    Image.fromarray((image * 255.0).round().astype(np.uint8)).save(path)


def sample_metrics(restored, gt_path, device, lpips_model):
    gt_array = grayscale_array(gt_path)
    if gt_array.shape != tuple(restored.shape[-2:]):
        raise ValueError(
            f"GT/restoration size mismatch: GT={gt_array.shape}, "
            f"restoration={tuple(restored.shape[-2:])}, path={gt_path}"
        )
    gt = torch.from_numpy(gt_array.copy()).view(1, 1, *gt_array.shape).to(device)
    mse = (restored - gt).square().mean().clamp_min(1e-12)
    restored_rgb = restored.repeat(1, 3, 1, 1) * 2.0 - 1.0
    gt_rgb = gt.repeat(1, 3, 1, 1) * 2.0 - 1.0
    lpips_value = lpips_model(restored_rgb, gt_rgb).mean()
    return (
        float((-10.0 * torch.log10(mse)).item()),
        float(ssim_pytorch(restored, gt).item()),
        float(lpips_value.item()),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final event-guided restoration using one complete fine-tune checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-format", choices=("efturb", "ctth", "real"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/final_inference")
    parser.add_argument(
        "--split", choices=("test", "train", "all"), default="test",
        help="For EFTurb/CTTH, test is the held-out sequence subset (default)"
    )
    parser.add_argument(
        "--test-list",
        help="Optional exact test-sequence manifest, one path relative to data-root per line"
    )
    parser.add_argument(
        "--allow-non-test", action="store_true",
        help="Required confirmation before --split train/all on EFTurb or CTTH"
    )
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--voxel-scale", type=float)
    parser.add_argument("--voxel-key", default="voxel")
    parser.add_argument("--gt", choices=("auto", "required", "none"), default="auto")
    parser.add_argument(
        "--temporal-mode",
        choices=("sliding-center", "nonoverlap-all"),
        default="sliding-center",
        help=(
            "sliding-center uses stride 1 and saves only each center frame; "
            "nonoverlap-all uses frame-disjoint windows and saves every frame"
        ),
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Maximum temporal windows after mode selection; 0 processes all"
    )
    parser.add_argument("--save-guide", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.tile_size <= 0 or args.tile_size % 4:
        parser.error("--tile-size must be positive and divisible by 4")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        parser.error("--overlap must satisfy 0 <= overlap < tile-size")
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")
    if args.data_format in {"efturb", "ctth"} and args.split != "test" and not args.allow_non_test:
        parser.error(
            "Refusing to infer training/all public-dataset sequences. Keep --split test, "
            "or explicitly add --allow-non-test."
        )
    if args.data_format == "real" and args.test_list:
        parser.error("--test-list is intended for EFTurb/CTTH; use a real-data root containing only desired sequences")
    return args


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Ref_MambaTM uses CUDA-only causal_conv1d/selective-scan kernels; "
            "run inference with --device cuda on an available GPU."
        )
    checkpoint, et_model, epaw_model, restoration = load_complete_models(
        args.checkpoint, device
    )
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    lpips_model.requires_grad_(False)
    saved_args = checkpoint_args(checkpoint)
    frames = args.frames or saved_args.get("eval_frames") or saved_args.get("train_frames") or 5
    voxel_scale = args.voxel_scale or saved_args.get("voxel_scale") or 4.0
    split_seed = args.split_seed if args.split_seed is not None else saved_args.get("split_seed", 42)
    train_ratio = args.train_ratio if args.train_ratio is not None else saved_args.get("train_ratio", 0.9)
    frames = int(frames)
    voxel_scale = float(voxel_scale)
    split_seed = int(split_seed)
    train_ratio = float(train_ratio)
    if frames < 1:
        raise ValueError("frames must be positive")
    if voxel_scale <= 0:
        raise ValueError("voxel scale must be positive")

    root = Path(args.data_root).resolve()
    discovered = discover_sequences(root, args.data_format)
    if not discovered:
        raise RuntimeError(f"no {args.data_format} sequence folders found under {root}")

    if args.data_format == "real":
        selected = discovered
        selection = "all real-data sequences (no public-dataset split applied)"
    elif args.test_list:
        if args.split != "test":
            raise ValueError("--test-list can only be used with --split test")
        selected = select_manifest_sequences(discovered, root, args.test_list)
        selection = f"exact test manifest: {Path(args.test_list).name}"
    else:
        selected = split_public_sequences(
            discovered, root, args.data_format, args.split, split_seed, train_ratio
        )
        selection = (
            f"deterministic {args.split} split: seed={split_seed}, "
            f"train_ratio={train_ratio}"
        )
    if not selected:
        raise RuntimeError(f"sequence selection is empty ({selection})")

    candidate_samples, _diagnostics = build_samples(
        selected, root, args.data_format, frames, args.gt
    )
    samples = select_temporal_samples(candidate_samples, args.temporal_mode)
    if args.max_samples:
        samples = samples[: args.max_samples]
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] format={args.data_format} selection={selection}; "
        f"sequences={len(selected)}/{len(discovered)} "
        f"candidate_windows={len(candidate_samples)} selected_windows={len(samples)} "
        f"temporal_mode={args.temporal_mode} frames={frames} GT={args.gt}",
        flush=True,
    )

    rows = []
    for index, sample in enumerate(samples):
        restored_sequence, guide_sequence = infer_sample(
            sample,
            et_model,
            epaw_model,
            restoration,
            device,
            args.tile_size,
            args.overlap,
            voxel_scale,
            args.voxel_key,
            args.amp,
        )
        if args.temporal_mode == "sliding-center":
            output_indices = [len(sample["frame_ids"]) // 2]
        else:
            output_indices = list(range(len(sample["frame_ids"])))
        sample_dir = output_root / sample["sequence"]
        window_metrics = []
        for time_index in output_indices:
            frame_id = sample["frame_ids"][time_index]
            restored = restored_sequence[:, time_index]
            guide = guide_sequence[:, time_index]
            restored_path = sample_dir / f"{frame_id:05d}_restored.png"
            save_tensor_image(restored, restored_path)
            guide_path = ""
            if args.save_guide:
                guide_path = sample_dir / f"{frame_id:05d}_guide.png"
                save_tensor_image(guide, guide_path)
            gt_path = sample["gt_window"][time_index]
            if gt_path is not None:
                psnr, ssim, lpips_value = sample_metrics(
                    restored, gt_path, device, lpips_model
                )
                window_metrics.append((psnr, ssim, lpips_value))
            else:
                psnr, ssim, lpips_value = None, None, None
            rows.append(
                {
                    "temporal_mode": args.temporal_mode,
                    "sequence": sample["relative_sequence"],
                    "window_center": sample["center"],
                    "frame": frame_id,
                    "is_window_center": int(time_index == len(sample["frame_ids"]) // 2),
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS": lpips_value,
                }
            )
        if window_metrics:
            metric_text = (
                f"mean_PSNR={np.mean([value[0] for value in window_metrics]):.3f} "
                f"mean_SSIM={np.mean([value[1] for value in window_metrics]):.4f} "
                f"mean_LPIPS={np.mean([value[2] for value in window_metrics]):.4f}"
            )
        else:
            metric_text = "GT=none"
        print(
            f"[{index + 1}/{len(samples)}] {sample['relative_sequence']} "
            f"window_center={sample['center']} outputs={len(output_indices)} "
            f"{metric_text}",
            flush=True,
        )

    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    valid_psnr = [row["PSNR"] for row in rows if row["PSNR"] is not None]
    valid_ssim = [row["SSIM"] for row in rows if row["SSIM"] is not None]
    valid_lpips = [row["LPIPS"] for row in rows if row["LPIPS"] is not None]
    summary = {
        "data_format": args.data_format,
        "selection": selection,
        "discovered_sequences": len(discovered),
        "selected_sequences": len(selected),
        "temporal_mode": args.temporal_mode,
        "candidate_windows": len(candidate_samples),
        "processed_windows": len(samples),
        "restored_frames": len(rows),
        "samples": len(rows),
        "samples_with_gt": len(valid_psnr),
        "frames": frames,
        "voxel_scale": voxel_scale,
        "mean_psnr": float(np.mean(valid_psnr)) if valid_psnr else None,
        "mean_ssim": float(np.mean(valid_ssim)) if valid_ssim else None,
        "mean_lpips": float(np.mean(valid_lpips)) if valid_lpips else None,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
