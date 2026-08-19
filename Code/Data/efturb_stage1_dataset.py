"""Stage 1 EFTurb loader using only scenes with non-static motion.

Stage 1 learns the event-to-motion mapping. ``Static_Scene_Image`` is excluded
from both training and validation because its converted clean-flow targets are
all zero and its 220 sequences would dominate the three motion scenarios.
Other training stages continue to use :mod:`Data.efturb_dataset` unchanged.
"""

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Data.efturb_dataset import (
    DEFAULT_EFTURB_ROOT,
    EFTurbDataset,
    collate_efturb,
    numeric_files,
)


EXCLUDED_STAGE1_SCENARIOS = frozenset({"Static_Scene_Image"})


def _sample_scenario(sample, root):
    relative = Path(sample["sequence_path"]).relative_to(Path(root))
    return relative.parts[0]


def stratified_validation_subset(samples, root, count):
    """Select deterministic, evenly spaced samples across all Stage 1 scenarios."""
    if count <= 0 or count >= len(samples):
        return list(samples)

    grouped = {}
    for sample in samples:
        grouped.setdefault(_sample_scenario(sample, root), []).append(sample)
    scenarios = sorted(grouped)
    quotas = {
        scenario: count // len(scenarios) + (index < count % len(scenarios))
        for index, scenario in enumerate(scenarios)
    }

    selected_by_scenario = {}
    for scenario in scenarios:
        scenario_samples = grouped[scenario]
        quota = min(quotas[scenario], len(scenario_samples))
        if quota <= 0:
            selected_by_scenario[scenario] = []
            continue
        # Midpoints of equal-width bins spread selections across sequences and
        # frame indices instead of taking adjacent windows from one sequence.
        indices = [
            min(int((index + 0.5) * len(scenario_samples) / quota), len(scenario_samples) - 1)
            for index in range(quota)
        ]
        selected_by_scenario[scenario] = [scenario_samples[index] for index in indices]

    selected = []
    offset = 0
    while len(selected) < count:
        added = False
        for scenario in scenarios:
            items = selected_by_scenario[scenario]
            if offset < len(items):
                selected.append(items[offset])
                added = True
        if not added:
            break
        offset += 1
    return selected[:count]


class DynamicStage1EFTurbDataset(EFTurbDataset):
    """Base EFTurb samples after removing Stage 1's static scenario."""

    def __init__(self, *args, **kwargs):
        kwargs["excluded_scenarios"] = EXCLUDED_STAGE1_SCENARIOS
        super().__init__(*args, **kwargs)
        if not self.samples:
            raise RuntimeError(
                f"No non-static EFTurb samples found under {self.root} for split={self.split}"
            )

        kept_sequences = {sample["sequence_path"] for sample in self.samples}
        scenarios = {_sample_scenario(sample, self.root) for sample in self.samples}
        if scenarios & EXCLUDED_STAGE1_SCENARIOS:
            raise RuntimeError(
                f"Stage 1 scenario exclusion failed: {sorted(scenarios & EXCLUDED_STAGE1_SCENARIOS)}"
            )
        print(
            "[INFO] stage1 scenario filter: "
            f"split={self.split} excluded={sorted(EXCLUDED_STAGE1_SCENARIOS)} "
            f"kept_scenarios={sorted(scenarios)} kept_sequences={len(kept_sequences)} "
            f"kept_samples={len(self.samples)}",
            flush=True,
        )


class MultiFrameFlowEFTurbDataset(DynamicStage1EFTurbDataset):
    """Non-static EFTurb samples with voxel and flow targets for every frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        voxel_cache = {}
        flow_cache = {}
        filtered = []
        for sample in self.samples:
            sequence_path = sample["sequence_path"]
            if sequence_path not in voxel_cache:
                voxel_cache[sequence_path] = numeric_files(
                    sequence_path / "Turb" / "event_voxel", ".npz"
                )
                flow_cache[sequence_path] = numeric_files(
                    sequence_path
                    / "Optical_Flow"
                    / "raw_gradient_flow_new_scalar_npz",
                    ".npz",
                )
            voxels = voxel_cache[sequence_path]
            flows = flow_cache[sequence_path]
            frame_ids = [int(Path(path).stem) for path in sample["images"]]
            if all(frame_id in voxels and frame_id in flows for frame_id in frame_ids):
                sample = dict(sample)
                sample["voxel_window"] = [voxels[frame_id] for frame_id in frame_ids]
                sample["flow_window"] = [flows[frame_id] for frame_id in frame_ids]
                filtered.append(sample)
        if not filtered:
            raise RuntimeError(
                f"No multi-frame flow samples found under {self.root} for split={self.split}"
            )
        dropped = len(self.samples) - len(filtered)
        self.samples = filtered
        if dropped:
            print(
                f"[INFO] dropped {dropped} samples without full {self.frames}-frame flow supervision "
                f"for split={self.split}",
                flush=True,
            )

    def _voxel(self, path, top, left, crop):
        with np.load(path) as data:
            voxel = data["voxel"][:, :, top : top + crop, left : left + crop]
        if not np.isfinite(voxel).all():
            raise ValueError(f"Non-finite event voxel in {path}")
        voxel = self.normalize_voxel(voxel, self.voxel_scale)
        return torch.from_numpy(voxel)

    def _flow(self, path, top, left, crop):
        with np.load(path) as data:
            flow = data["raw_gradient_flow"][:, top : top + crop, left : left + crop]
        if not np.isfinite(flow).all():
            raise ValueError(f"Non-finite flow target in {path}")
        flow = self.normalize_flow(flow, self.flow_scale)
        return torch.from_numpy(flow)

    def __getitem__(self, index):
        output = super().__getitem__(index)
        top, left, crop = [int(value) for value in output["crop"].tolist()]
        sample = self.samples[index]
        output["event_voxel"] = torch.stack(
            [self._voxel(path, top, left, crop) for path in sample["voxel_window"]]
        )
        output["flow_target"] = torch.stack(
            [self._flow(path, top, left, crop) for path in sample["flow_window"]]
        )
        return output


def make_multiframe_flow_loader(
    args,
    split,
    load_events=True,
    load_gt=True,
    shuffle=None,
    crop_size=None,
    frames=None,
    batch_size=None,
):
    """Build the distributed Stage 1 loader without static-scene samples."""
    dataset = MultiFrameFlowEFTurbDataset(
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
    validation_samples = getattr(args, "validation_samples", 0)
    if split != "train" and validation_samples > 0:
        full_count = len(dataset.samples)
        dataset.samples = stratified_validation_subset(
            dataset.samples,
            dataset.root,
            validation_samples,
        )
        scenarios = sorted({_sample_scenario(sample, dataset.root) for sample in dataset.samples})
        print(
            "[INFO] stage1 stratified validation subset: "
            f"selected={len(dataset.samples)}/{full_count} scenarios={scenarios}",
            flush=True,
        )
    use_shuffle = split == "train" if shuffle is None else shuffle
    sampler = (
        DistributedSampler(dataset, shuffle=use_shuffle)
        if dist.is_initialized() and split == "train"
        else None
    )
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
