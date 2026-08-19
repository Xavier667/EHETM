"""Dataset loader for the CTTH+ dataset."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CTTH_ROOT = REPOSITORY_ROOT / "datasets" / "CTTH+_Dataset"


def numeric_files(folder, suffix):
    return {int(path.stem): path for path in Path(folder).glob(f"*{suffix}") if path.stem.isdigit()}


def split_sequences(root, split, split_seed=42, train_ratio=0.9):
    sequences = sorted(Path(root).glob("seq_*"))
    random.Random(split_seed).shuffle(sequences)
    cut = round(len(sequences) * train_ratio)
    return sequences[:cut] if split == "train" else sequences[cut:]


class CTTHDataset(Dataset):
    """Five-frame windows with center-frame flow, gradient and restoration targets."""

    def __init__(
        self,
        root=DEFAULT_CTTH_ROOT,
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
    ):
        if frames < 1:
            raise ValueError("frames must be positive")
        self.root = Path(root)
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
        self.samples = []

        for sequence in split_sequences(self.root, split, split_seed, train_ratio):
            images = numeric_files(sequence / "Turb" / "frames_gray", ".png")
            events = numeric_files(sequence / "Turb" / "events", ".npz")
            voxels = numeric_files(sequence / "Turb" / "event_voxel", ".npz")
            gt_images = numeric_files(sequence / "GT" / "frames_gray", ".png")
            flow = numeric_files(sequence / "Flow" / "raw_gradient_flow_scalar_npz", ".npz")
            gradient = numeric_files(sequence / "Flow" / "raw_gradient_png_8bit", ".png")
            for center in sorted(set(flow) & set(gradient) & set(voxels)):
                window = range(center - self.radius, center + (self.frames - self.radius))
                if all(i in images and i in events for i in window) and center in gt_images:
                    self.samples.append(
                        {
                            "sequence": sequence.name,
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
            raise RuntimeError(f"No samples found under {root} for split={split}")

    def __len__(self):
        return len(self.samples)

    def _crop_box(self, height, width):
        crop = min(self.crop_size, height, width)
        if crop % 4:
            raise ValueError("crop_size must be divisible by 4")
        if self.random_crop:
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        else:
            top = (height - crop) // 2
            left = (width - crop) // 2
        return top, left, crop

    @staticmethod
    def _image(path, top, left, crop):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        image = image[top : top + crop, left : left + crop]
        return torch.from_numpy(image.copy()).unsqueeze(0)

    def _noisy(self, image):
        if self.noise_std <= 0 or self.split != "train":
            return image
        return (image + torch.randn_like(image) * self.noise_std).clamp_(0, 1)

    @staticmethod
    def normalize_voxel(voxel, scale=4.0):
        """Compress sparse event-count spikes while preserving absolute density."""
        voxel = voxel.astype(np.float32)
        voxel = np.sign(voxel) * np.log1p(np.abs(voxel)) / np.log1p(scale)
        return np.clip(voxel, -1.0, 1.0)

    @staticmethod
    def normalize_flow(flow, scale=4000.0):
        """Map positive motion magnitudes to [0, 1] with a robust fixed scale."""
        flow = np.maximum(flow.astype(np.float32), 0.0)
        flow = np.log1p(flow) / np.log1p(scale)
        return np.clip(flow, 0.0, 1.0)

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
                    (x >= left) & (x < left + crop) &
                    (y >= top) & (y < top + crop)
                )

                x = x[valid].astype(np.float32) - left
                y = y[valid].astype(np.float32) - top

                polarity = data["p"][valid].astype(np.float32)

                # Compatible with both 0/1 and -1/+1 polarity formats.
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
                f"[WARN] failed to load event file {path}: {type(exc).__name__}: {exc}; using empty events",
                flush=True,
            )
            return torch.empty((0, 4), dtype=torch.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["images"][0]) as image:
            width, height = image.size
        top, left, crop = self._crop_box(height, width)
        voxel = np.load(sample["voxel"])["voxel"][:, :, top : top + crop, left : left + crop]
        flow = np.load(sample["flow"])["raw_gradient_flow"][:, top : top + crop, left : left + crop]
        if not np.isfinite(voxel).all() or not np.isfinite(flow).all():
            raise ValueError(
                f"Non-finite training data in sequence={sample['sequence']} center={sample['center']}"
            )
        voxel = self.normalize_voxel(voxel, self.voxel_scale)
        flow = self.normalize_flow(flow, self.flow_scale)
        output = {
            "sequence": sample["sequence"],
            "center": sample["center"],
            "crop": torch.tensor([top, left, crop]),
            "event_voxel": torch.from_numpy(voxel),
            "flow_target": torch.from_numpy(flow),
        }
        if self.load_images:
            output["images"] = torch.stack([self._noisy(self._image(path, top, left, crop)) for path in sample["images"]])
        output["gradient_target"] = self._image(sample["gradient"], top, left, crop)
        if self.load_events:
            output["warped_events"] = [self._events(path, top, left, crop) for path in sample["events"]]
        if self.load_gt:
            output["gt"] = self._image(sample["gt"], top, left, crop)
        return output


def collate_ctth(samples):
    batch = {}
    for key in ["event_voxel", "flow_target", "images", "gradient_target", "gt", "crop"]:
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples])
    if "warped_events" in samples[0]:
        batch["warped_events"] = [sample["warped_events"] for sample in samples]
    batch["sequence"] = [sample["sequence"] for sample in samples]
    batch["center"] = [sample["center"] for sample in samples]
    return batch
