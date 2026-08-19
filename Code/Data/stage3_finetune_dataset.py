"""Efficient data pipeline for Stage 3 event-guided fine-tuning.

The loader reads only tensors consumed by ET -> EPAW -> Restoration:

* five degraded grayscale frames;
* five matching GT grayscale frames;
* the center event voxel;
* five raw event streams.

It never opens optical-flow targets or gradient supervision.  Samples are
indexed once, neighbouring temporal windows can be sampled in short shuffled
chunks, and small bounded in-process caches avoid decoding the same overlapping
PNG/NPZ files repeatedly.  No dataset file is copied to local storage.
"""

from __future__ import annotations

import random
import time
import os
import pickle
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, Sampler

from Data.ctth_dataset import (
    DEFAULT_CTTH_ROOT,
    split_sequences as split_ctth_sequences,
)
from Data.efturb_dataset import (
    DEFAULT_EFTURB_ROOT,
    split_sequences as split_efturb_sequences,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = REPOSITORY_ROOT / ".cache" / "stage3_finetune"
INDEX_VERSION = 2

_INDEX_CACHE = {}
_MISSING = object()


def numeric_files(folder, suffix):
    """Match the fast Stage 1/2 glob path without an NFS stat per entry."""
    return {
        int(path.stem): path
        for path in Path(folder).glob(f"*{suffix}")
        if path.stem.isdigit()
    }


def _value_nbytes(value):
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (tuple, list)):
        return sum(_value_nbytes(item) for item in value)
    return 0


class ByteLRU:
    """Per-worker, byte-bounded decoded-array cache."""

    def __init__(self, capacity_mb):
        self.capacity = max(0, int(float(capacity_mb) * 1024 * 1024))
        self.size = 0
        self.values = OrderedDict()

    def get_or_load(self, key, loader):
        if self.capacity <= 0:
            return loader(), False
        value = self.values.pop(key, _MISSING)
        if value is not _MISSING:
            self.values[key] = value
            return value, True

        value = loader()
        value_size = _value_nbytes(value)
        if 0 < value_size <= self.capacity:
            while self.values and self.size + value_size > self.capacity:
                _, removed = self.values.popitem(last=False)
                self.size -= _value_nbytes(removed)
            self.values[key] = value
            self.size += value_size
        return value, False


def _domain_layout(root, domain, split, split_seed, train_ratio):
    if domain == "efturb":
        return (
            split_efturb_sequences(root, split, split_seed, train_ratio),
            Path("Turb/frames"),
            Path("GT/frames"),
        )
    if domain == "ctth":
        return (
            split_ctth_sequences(root, split, split_seed, train_ratio),
            Path("Turb/frames_gray"),
            Path("GT/frames_gray"),
        )
    raise ValueError(f"unsupported fine-tuning domain: {domain}")


def _manifest_path(index_dir, root, domain, split, frames, split_seed, train_ratio):
    identity = "|".join(
        (
            str(root.absolute()),
            domain,
            split,
            str(int(frames)),
            str(int(split_seed)),
            f"{float(train_ratio):.8f}",
        )
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return Path(index_dir) / f"{domain}_{split}_f{frames}_{digest}.pkl"


def _build_index(
    root,
    domain,
    split,
    frames,
    split_seed,
    train_ratio,
    index_dir,
    rebuild_index=False,
):
    key = (
        str(root.absolute()),
        domain,
        split,
        int(frames),
        int(split_seed),
        float(train_ratio),
    )
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    manifest = _manifest_path(
        index_dir, root, domain, split, frames, split_seed, train_ratio
    )
    if manifest.is_file() and not rebuild_index:
        try:
            with manifest.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") != INDEX_VERSION:
                raise ValueError("index version changed")
            result = (payload["samples"], payload["sequence_groups"])
            _INDEX_CACHE[key] = result
            return result
        except Exception as exc:
            print(
                f"[WARN] rebuilding unreadable Stage 3 index {manifest}: {exc}",
                flush=True,
            )

    sequences, input_relative, gt_relative = _domain_layout(
        root, domain, split, split_seed, train_ratio
    )
    radius = frames // 2
    samples = []
    sequence_groups = []

    for sequence_path in sequences:
        images = numeric_files(sequence_path / input_relative, ".png")
        gt_images = numeric_files(sequence_path / gt_relative, ".png")
        events = numeric_files(sequence_path / "Turb/events", ".npz")
        voxels = numeric_files(sequence_path / "Turb/event_voxel", ".npz")
        if not images or not gt_images or not events or not voxels:
            continue

        first_image = next(iter(images.values()))
        with Image.open(first_image) as image:
            width, height = image.size
        sequence_id = "__".join(sequence_path.relative_to(root).parts)
        group_start = len(samples)

        for center in sorted(voxels):
            window = tuple(
                range(center - radius, center + (frames - radius))
            )
            if not all(
                frame in images and frame in gt_images and frame in events
                for frame in window
            ):
                continue
            samples.append(
                {
                    "domain": domain,
                    "sequence": sequence_id,
                    "center": center,
                    "width": width,
                    "height": height,
                    "images": tuple(images[frame] for frame in window),
                    "gt": tuple(gt_images[frame] for frame in window),
                    "events": tuple(events[frame] for frame in window),
                    "voxel": voxels[center],
                }
            )

        if len(samples) > group_start:
            sequence_groups.append(tuple(range(group_start, len(samples))))

    if not samples:
        raise RuntimeError(
            f"no Stage 3 fine-tuning samples under {root}; "
            f"domain={domain} split={split} frames={frames}"
        )
    result = (samples, sequence_groups)
    _INDEX_CACHE[key] = result
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(
        f".{manifest.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "version": INDEX_VERSION,
                    "samples": samples,
                    "sequence_groups": sequence_groups,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return result


class Stage3FineTuneDataset(Dataset):
    """One CTTH+ or EFTurb domain with exact Stage 3 fine-tuning inputs."""

    def __init__(
        self,
        root,
        domain,
        split="train",
        crop_size=512,
        frames=5,
        split_seed=42,
        train_ratio=0.9,
        voxel_scale=4.0,
        noise_std=0.0,
        load_events=True,
        image_cache_mb=64,
        event_cache_mb=128,
        index_dir=DEFAULT_INDEX_DIR,
        rebuild_index=False,
    ):
        if frames < 1:
            raise ValueError("frames must be positive")
        if crop_size < 1 or crop_size % 4:
            raise ValueError("crop_size must be positive and divisible by 4")
        if voxel_scale <= 0:
            raise ValueError("voxel_scale must be positive")
        if min(image_cache_mb, event_cache_mb) < 0:
            raise ValueError("cache sizes must be non-negative")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{domain} root does not exist: {self.root}")
        self.domain = domain.lower()
        self.split = split
        self.crop_size = int(crop_size)
        self.random_crop = split == "train"
        self.voxel_scale = float(voxel_scale)
        self.noise_std = float(noise_std) if split == "train" else 0.0
        self.load_events = bool(load_events)
        self.image_cache = ByteLRU(image_cache_mb)
        self.event_cache = ByteLRU(event_cache_mb)
        index_args = (
            self.root,
            self.domain,
            split,
            frames,
            split_seed,
            train_ratio,
            index_dir,
        )
        manifest = _manifest_path(
            index_dir,
            self.root,
            self.domain,
            split,
            frames,
            split_seed,
            train_ratio,
        )
        coordinate_build = (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and (rebuild_index or not manifest.is_file())
        )
        if coordinate_build and dist.get_rank() != 0:
            dist.barrier()
            self.samples, self.sequence_groups = _build_index(
                *index_args, rebuild_index=False
            )
        else:
            self.samples, self.sequence_groups = _build_index(
                *index_args, rebuild_index=rebuild_index
            )
            if coordinate_build:
                dist.barrier()

    def __len__(self):
        return len(self.samples)

    def _crop_box(self, height, width):
        crop = min(self.crop_size, height, width)
        if crop % 4:
            raise ValueError(
                f"effective crop {crop} must be divisible by 4; "
                f"image is {width}x{height}"
            )
        if self.random_crop:
            return (
                random.randint(0, height - crop),
                random.randint(0, width - crop),
                crop,
            )
        return (height - crop) // 2, (width - crop) // 2, crop

    def _gray_array(self, path):
        path = Path(path)

        def load():
            with Image.open(path) as image:
                return np.asarray(image.convert("L"), dtype=np.uint8).copy()

        image, _ = self.image_cache.get_or_load(str(path), load)
        return image

    def _image(self, path, top, left, crop):
        image = self._gray_array(path)
        cropped = image[top : top + crop, left : left + crop]
        tensor = torch.from_numpy(np.ascontiguousarray(cropped)).unsqueeze(0)
        return tensor.to(dtype=torch.float32).div_(255.0)

    def _voxel(self, path, top, left, crop):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "voxel" not in data:
                raise KeyError(f"missing 'voxel' array in {path}")
            voxel = np.ascontiguousarray(
                data["voxel"][:, :, top : top + crop, left : left + crop],
                dtype=np.float32,
            )
        voxel = np.sign(voxel) * np.log1p(np.abs(voxel))
        voxel /= np.log1p(self.voxel_scale)
        np.clip(voxel, -1.0, 1.0, out=voxel)
        return torch.from_numpy(voxel)

    @staticmethod
    def _read_event_arrays(path):
        with np.load(path, allow_pickle=False) as data:
            missing = {"x", "y", "p", "t"} - set(data.files)
            if missing:
                raise KeyError(f"missing event keys {sorted(missing)} in {path}")
            return tuple(np.asarray(data[key]) for key in ("x", "y", "p", "t"))

    def _events(self, path, top, left, crop):
        path = Path(path)
        arrays, _ = self.event_cache.get_or_load(
            str(path), lambda: self._read_event_arrays(path)
        )
        x_all, y_all, polarity_all, timestamp_all = arrays
        valid = (
            (x_all >= left)
            & (x_all < left + crop)
            & (y_all >= top)
            & (y_all < top + crop)
        )
        x = x_all[valid].astype(np.float32, copy=False) - left
        y = y_all[valid].astype(np.float32, copy=False) - top
        polarity = polarity_all[valid].astype(np.float32, copy=False)
        timestamp = timestamp_all[valid].astype(np.float32, copy=False)
        if polarity.size and polarity.min() >= 0 and polarity.max() <= 1:
            polarity = polarity * 2.0 - 1.0
        else:
            polarity = np.sign(polarity)
            polarity[polarity == 0] = 1.0
        events = np.stack((x, y, polarity, timestamp), axis=1).astype(
            np.float32, copy=False
        )
        return torch.from_numpy(events)

    def __getitem__(self, index):
        sample = self.samples[index]
        top, left, crop = self._crop_box(sample["height"], sample["width"])
        images = torch.stack(
            [self._image(path, top, left, crop) for path in sample["images"]]
        )
        if self.noise_std > 0:
            images = (
                images + torch.randn_like(images) * self.noise_std
            ).clamp_(0, 1)
        output = {
            "domain": self.domain,
            "sequence": sample["sequence"],
            "center": sample["center"],
            "crop": torch.tensor((top, left, crop), dtype=torch.int64),
            "event_voxel": self._voxel(sample["voxel"], top, left, crop),
            "images": images,
            "gt": torch.stack(
                [self._image(path, top, left, crop) for path in sample["gt"]]
            ),
        }
        if self.load_events:
            output["warped_events"] = [
                self._events(path, top, left, crop) for path in sample["events"]
            ]
        return output


class JointStage3FineTuneDataset(Dataset):
    """Concat-like wrapper that preserves sequence groups for local sampling."""

    def __init__(self, datasets):
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("joint dataset requires at least one domain")
        self.offsets = []
        self.sequence_groups = []
        offset = 0
        for dataset in self.datasets:
            self.offsets.append(offset)
            self.sequence_groups.extend(
                tuple(offset + index for index in group)
                for group in dataset.sequence_groups
            )
            offset += len(dataset)
        self.length = offset

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for offset, dataset in zip(reversed(self.offsets), reversed(self.datasets)):
            if index >= offset:
                return dataset[index - offset]
        raise IndexError(index)


class SequenceChunkSampler(Sampler):
    """Shuffle short temporal chunks while retaining cache locality."""

    def __init__(
        self,
        dataset,
        chunk_size=8,
        shuffle=True,
        seed=42,
        rank=0,
        world_size=1,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank/world_size")
        self.dataset = dataset
        self.chunk_size = max(1, int(chunk_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.indices = []
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch * 1000003)
        chunks = []
        for group in self.dataset.sequence_groups:
            for start in range(0, len(group), self.chunk_size):
                chunk = list(group[start : start + self.chunk_size])
                if self.shuffle and rng.random() < 0.5:
                    chunk.reverse()
                chunks.append(chunk)
        if self.shuffle:
            rng.shuffle(chunks)

        rank_chunks = [[] for _ in range(self.world_size)]
        rank_lengths = [0] * self.world_size
        for chunk in chunks:
            target_rank = min(
                range(self.world_size), key=lambda item: rank_lengths[item]
            )
            rank_chunks[target_rank].extend(chunk)
            rank_lengths[target_rank] += len(chunk)

        target_length = max(rank_lengths, default=0)
        for rank_indices in rank_chunks:
            original = tuple(rank_indices)
            if not original:
                continue
            while len(rank_indices) < target_length:
                missing = target_length - len(rank_indices)
                rank_indices.extend(original[:missing])
        self.indices = rank_chunks[self.rank]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_stage3_finetune(samples):
    batch = {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("event_voxel", "images", "gt", "crop")
    }
    if "warped_events" in samples[0]:
        batch["warped_events"] = [sample["warped_events"] for sample in samples]
    for key in ("domain", "sequence", "center"):
        batch[key] = [sample[key] for sample in samples]
    return batch


def stage3_worker_init(_worker_id):
    """Keep each worker single-threaded and independently reproducible."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
