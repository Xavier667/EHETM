"""Dataset loader for the four-scenario EFTurb dataset.

The public sample interface intentionally matches ``Data.ctth_dataset`` so the
three training stages can reuse their existing model and checkpoint code.
EFTurb RGB frames are decoded as grayscale. Flow supervision is read from the
precomputed L2 magnitude of ``clean_flow_npz`` and then receives the same fixed
log normalization used for CTTH+.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFTURB_ROOT = REPOSITORY_ROOT / "datasets" / "EFTurb"


def numeric_files(folder, suffix):
    """Return numeric-stem files as ``{frame_index: path}``."""
    return {
        int(path.stem): path
        for path in Path(folder).glob(f"*{suffix}")
        if path.stem.isdigit()
    }


def _scenario_sequences(root, excluded_scenarios=()):
    """Find sequence folders one level below each EFTurb scenario folder."""
    root = Path(root)
    excluded_scenarios = set(excluded_scenarios or ())
    scenarios = {}
    for scenario in sorted(path for path in root.iterdir() if path.is_dir()):
        if scenario.name in excluded_scenarios:
            continue
        sequences = sorted(
            path
            for path in scenario.iterdir()
            if path.is_dir()
            and (path / "Turb").is_dir()
            and (path / "GT").is_dir()
            and (path / "Optical_Flow").is_dir()
        )
        if sequences:
            scenarios[scenario.name] = sequences
    return scenarios


def split_sequences(
    root,
    split,
    split_seed=42,
    train_ratio=0.9,
    excluded_scenarios=(),
):
    """Deterministically split sequences within every scenario.

    A per-scenario split keeps all four EFTurb scene types represented in both
    training and validation, including ``Dynamic_Scene_800FPS`` which has only
    nine sequences.
    """
    if split not in {"train", "val", "validation", "test"}:
        raise ValueError(f"Unsupported split={split!r}; expected 'train' or 'val'")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    selected = []
    for scenario, sequences in _scenario_sequences(root, excluded_scenarios).items():
        sequences = list(sequences)
        random.Random(f"{split_seed}:{scenario}").shuffle(sequences)
        cut = round(len(sequences) * train_ratio)
        if len(sequences) > 1:
            cut = min(max(cut, 1), len(sequences) - 1)
        selected.extend(sequences[:cut] if split == "train" else sequences[cut:])
    return selected


class EFTurbDataset(Dataset):
    """Five-frame windows with CTTH-compatible tensors and sample keys."""

    def __init__(
        self,
        root=DEFAULT_EFTURB_ROOT,
        split="train",
        crop_size=128,
        frames=5,
        random_crop=None,
        flow_scale=4000.0,
        voxel_scale=4.0,
        load_images=True,
        load_events=True,
        load_gt=True,
        split_seed=42,
        train_ratio=0.9,
        noise_std=0.0,
        excluded_scenarios=None,
    ):
        if frames < 1:
            raise ValueError("frames must be positive")
        if crop_size < 1 or crop_size % 4:
            raise ValueError("crop_size must be positive and divisible by 4")
        if flow_scale <= 0 or voxel_scale <= 0:
            raise ValueError("flow_scale and voxel_scale must be positive")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"EFTurb root does not exist: {self.root}")
        self.split = split
        self.crop_size = crop_size
        self.frames = frames
        self.radius = frames // 2
        self.random_crop = split == "train" if random_crop is None else random_crop
        self.flow_scale = flow_scale
        self.voxel_scale = voxel_scale
        self.load_images = load_images
        self.load_events = load_events
        self.load_gt = load_gt
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.noise_std = noise_std
        self.excluded_scenarios = frozenset(excluded_scenarios or ())
        self.samples = []

        sequences = split_sequences(
            self.root,
            split,
            split_seed,
            train_ratio,
            excluded_scenarios=self.excluded_scenarios,
        )
        if not sequences:
            raise RuntimeError(f"No EFTurb sequence folders found under {self.root}")

        for sequence_path in sequences:
            images = numeric_files(sequence_path / "Turb" / "frames", ".png")
            events = numeric_files(sequence_path / "Turb" / "events", ".npz")
            voxels = numeric_files(sequence_path / "Turb" / "event_voxel", ".npz")
            gt_images = numeric_files(sequence_path / "GT" / "frames", ".png")
            flow = numeric_files(
                sequence_path
                / "Optical_Flow"
                / "raw_gradient_flow_new_scalar_npz",
                ".npz",
            )
            gradient = numeric_files(
                sequence_path / "Optical_Flow" / "raw_gradient_png_8bit", ".png"
            )

            relative = sequence_path.relative_to(self.root)
            sequence_id = "__".join(relative.parts)
            for center in sorted(set(flow) & set(gradient) & set(voxels)):
                window = range(
                    center - self.radius,
                    center + (self.frames - self.radius),
                )
                if all(i in images and i in events for i in window) and center in gt_images:
                    self.samples.append(
                        {
                            "sequence": sequence_id,
                            "sequence_path": sequence_path,
                            "center": center,
                            "images": [images[i] for i in window],
                            "events": [events[i] for i in window],
                            "voxel": voxels[center],
                            "flow": flow[center],
                            "gradient": gradient[center],
                            "gt": gt_images[center],
                        }
                    )
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} for split={split}")

    def __len__(self):
        return len(self.samples)

    def _crop_box(self, height, width):
        crop = min(self.crop_size, height, width)
        if crop % 4:
            raise ValueError(
                f"effective crop size {crop} must be divisible by 4; image is {width}x{height}"
            )
        if self.random_crop:
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        else:
            top = (height - crop) // 2
            left = (width - crop) // 2
        return top, left, crop

    @staticmethod
    def _image(path, top, left, crop):
        # EFTurb frames are RGB; CTTH+ and all four training scripts expect 1 channel.
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        image = image[top : top + crop, left : left + crop]
        return torch.from_numpy(image.copy()).unsqueeze(0)

    def _noisy(self, image):
        if self.noise_std <= 0 or self.split != "train":
            return image
        return (image + torch.randn_like(image) * self.noise_std).clamp_(0, 1)

    @staticmethod
    def normalize_voxel(voxel, scale=4.0):
        """Use the same signed-log event normalization as CTTH+."""
        voxel = voxel.astype(np.float32)
        voxel = np.sign(voxel) * np.log1p(np.abs(voxel)) / np.log1p(scale)
        return np.clip(voxel, -1.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def normalize_flow(flow, scale=4000.0):
        """Apply the same non-negative fixed-log target mapping as CTTH+."""
        flow = np.maximum(flow.astype(np.float32), 0.0)
        flow = np.log1p(flow) / np.log1p(scale)
        return np.clip(flow, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _events(path, top, left, crop):
        try:
            with np.load(path) as data:
                missing = {"x", "y", "p", "t"} - set(data.files)
                if missing:
                    raise KeyError(f"missing event keys: {sorted(missing)}")

                x = data["x"]
                y = data["y"]
                valid = (
                    (x >= left)
                    & (x < left + crop)
                    & (y >= top)
                    & (y < top + crop)
                )
                x = x[valid].astype(np.float32) - left
                y = y[valid].astype(np.float32) - top

                polarity = data["p"][valid].astype(np.float32)
                if polarity.size > 0 and polarity.min() >= 0 and polarity.max() <= 1:
                    polarity = polarity * 2.0 - 1.0
                else:
                    polarity = np.sign(polarity)
                    polarity[polarity == 0] = 1.0

                timestamp = data["t"][valid].astype(np.float32)
                events = np.stack([x, y, polarity, timestamp], axis=1).astype(np.float32)
                return torch.from_numpy(events)
        except Exception as exc:
            print(
                f"[WARN] failed to load event file {path}: "
                f"{type(exc).__name__}: {exc}; using empty events",
                flush=True,
            )
            return torch.empty((0, 4), dtype=torch.float32)

    @staticmethod
    def _load_voxel(path, top, left, crop):
        with np.load(path) as data:
            if "voxel" not in data:
                raise KeyError(f"missing 'voxel' array in {path}")
            voxel = data["voxel"]
            if voxel.ndim != 4 or voxel.shape[:2] != (2, 10):
                raise ValueError(f"expected voxel [2,10,H,W], got {voxel.shape} in {path}")
            return voxel[:, :, top : top + crop, left : left + crop]

    @staticmethod
    def _load_flow(path, top, left, crop):
        with np.load(path) as data:
            if "raw_gradient_flow" not in data:
                raise KeyError(f"missing 'raw_gradient_flow' array in {path}")
            flow = data["raw_gradient_flow"]
            if flow.ndim != 3 or flow.shape[0] != 1:
                raise ValueError(f"expected flow [1,H,W], got {flow.shape} in {path}")
            return flow[:, top : top + crop, left : left + crop]

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["images"][0]) as image:
            width, height = image.size
        top, left, crop = self._crop_box(height, width)

        voxel = self._load_voxel(sample["voxel"], top, left, crop)
        flow = self._load_flow(sample["flow"], top, left, crop)
        if not np.isfinite(voxel).all() or not np.isfinite(flow).all():
            raise ValueError(
                f"Non-finite training data in sequence={sample['sequence']} "
                f"center={sample['center']}"
            )

        output = {
            "sequence": sample["sequence"],
            "center": sample["center"],
            "crop": torch.tensor([top, left, crop]),
            "event_voxel": torch.from_numpy(
                self.normalize_voxel(voxel, self.voxel_scale)
            ),
            "flow_target": torch.from_numpy(
                self.normalize_flow(flow, self.flow_scale)
            ),
        }
        if self.load_images:
            output["images"] = torch.stack(
                [
                    self._noisy(self._image(path, top, left, crop))
                    for path in sample["images"]
                ]
            )
        output["gradient_target"] = self._image(sample["gradient"], top, left, crop)
        if self.load_events:
            output["warped_events"] = [
                self._events(path, top, left, crop) for path in sample["events"]
            ]
        if self.load_gt:
            output["gt"] = self._image(sample["gt"], top, left, crop)
        return output


def collate_efturb(samples):
    """CTTH-compatible collation, including variable-length raw events."""
    batch = {}
    for key in ["event_voxel", "flow_target", "images", "gradient_target", "gt", "crop"]:
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples])
    if "warped_events" in samples[0]:
        batch["warped_events"] = [sample["warped_events"] for sample in samples]
    batch["sequence"] = [sample["sequence"] for sample in samples]
    batch["center"] = [sample["center"] for sample in samples]
    return batch


# Convenient drop-in alias for auxiliary code that still imports the old name.
collate_ctth = collate_efturb
