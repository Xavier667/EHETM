# EHETM

Official PyTorch implementation of EHETM for event-guided turbulence mitigation. This release contains the complete three-stage training pipeline, event-guided fine-tuning, unified inference for EFTurb/CTTH+/real data, and three final training-resume checkpoints.

## Release contents

```text
EHETM/
├── Data/                          # EFTurb and CTTH+ dataset loaders
│   └── splits/                    # Released deterministic test manifests
├── Model/                         # ET, EPAW, and Ref-MambaTM networks
├── Utils/                         # Metrics, training, Hilbert, and Mamba utilities
├── checkpoints/                   # Three final inference/resume checkpoints
├── train_stage1_et.py
├── train_stage2_epaw.py
├── train_stage3_restoration.py
├── train_stage3_finetuning.py
├── infer_final.py
└── requirements.txt
```

The released checkpoints preserve `et_model`, `epaw_model`, `restoration_model`, optimizer, scheduler, AMP scaler, and training-progress state. Private training paths and machine-specific fields are replaced with portable values, so the same files support both direct inference and continued fine-tuning.

## Environment

The code was validated with:

- Ubuntu 18.04
- Python 3.11.13
- PyTorch 2.1.1 + CUDA 12.1
- NVIDIA RTX 3090

An NVIDIA GPU is required by the selective-scan and causal-convolution kernels used in Ref-MambaTM. A matching CUDA toolkit and a working C/C++ compiler are required when `mamba-ssm` and `causal-conv1d` are built.

Create the environment:

```bash
conda create -n ehetm python=3.11 -y
conda activate ehetm

# Install PyTorch first so the Mamba extensions can compile against it.
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
```

Verify the CUDA installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

LPIPS initializes an AlexNet backbone on first use and may download its pretrained weights if they are not already cached.

## Checkpoints

| File | Training domain | SHA-256 |
| --- | --- | --- |
| `restoration_event_guided_finetune_CTTH.pt` | CTTH+ | `d18d66bccff01b1215226d4526a10e50ad5d29b6f6d2a4e03df212870ef68ee7` |
| `restoration_event_guided_finetune_EFTURB.pt` | EFTurb | `8356afcd9c8ae6621d675f31046cf3696189298e66f35467332f70249031afd2` |
| `restoration_event_guided_finetune_Joint.pt` | CTTH+ and EFTurb | `c172a7b187feb6a05d5661b09b237ec339b8f20abebf42b40c0d56aa33974021` |

Each file can be passed directly to `infer_final.py` or to `train_stage3_finetuning.py --finetune-resume`. Put downloaded weights in `checkpoints/` if they are distributed separately from the source repository.

## Dataset layout

The datasets are not included. Pass their locations explicitly; no server-specific path is embedded in the code.

EFTurb:

```text
EFTurb/
└── <scenario>/<sequence>/
    ├── Turb/
    │   ├── frames/<frame>.png
    │   ├── events/<frame>.npz
    │   └── event_voxel/<frame>.npz
    ├── GT/frames/<frame>.png
    └── Optical_Flow/
        ├── raw_gradient_flow_new_scalar_npz/<frame>.npz
        └── raw_gradient_png_8bit/<frame>.png
```

CTTH+:

```text
CTTH+_Dataset/
└── seq_*/
    ├── Turb/
    │   ├── frames_gray/<frame>.png
    │   ├── events/<frame>.npz
    │   └── event_voxel/<frame>.npz
    ├── GT/frames_gray/<frame>.png
    └── Flow/
        ├── raw_gradient_flow_scalar_npz/<frame>.npz
        └── raw_gradient_png_8bit/<frame>.png
```

Expected NPZ fields:

- event stream: `x`, `y`, `p`, `t`;
- event voxel: `voxel`, shaped `[2, 10, H, W]`;
- scalar flow target: `raw_gradient_flow`, shaped `[1, H, W]`.

Frame stems must be numeric. The default split is deterministic with seed `42` and a `0.9` train ratio. Exact released test sequence lists are in `Data/splits/`.

## Inference

Run all commands from the repository root.

EFTurb checkpoint:

```bash
python infer_final.py \
  --checkpoint checkpoints/restoration_event_guided_finetune_EFTURB.pt \
  --data-format efturb \
  --data-root /path/to/EFTurb \
  --test-list Data/splits/efturb_test_seed42_ratio0.9.txt \
  --output outputs/efturb \
  --tile-size 512 --overlap 64 --amp
```

CTTH+ checkpoint:

```bash
python infer_final.py \
  --checkpoint checkpoints/restoration_event_guided_finetune_CTTH.pt \
  --data-format ctth \
  --data-root /path/to/CTTH+_Dataset \
  --test-list Data/splits/ctth_test_seed42_ratio0.9.txt \
  --output outputs/ctth \
  --tile-size 512 --overlap 0 --amp
```

The joint checkpoint uses the same command with the joint checkpoint path and either matching `--data-format`.

Real event-camera data without ground truth:

```bash
python infer_final.py \
  --checkpoint checkpoints/restoration_event_guided_finetune_Joint.pt \
  --data-format real \
  --data-root /path/to/real_sequences \
  --gt none \
  --output outputs/real \
  --tile-size 512 --overlap 64 --amp --save-guide
```

For real data, each sequence may use `frames/`, `events/`, and `event_voxel/`; the two public layouts are also accepted. If ground truth is available, use `gt/` or `ground_truth/` and keep `--gt auto` (the default).

Inference writes restored PNG files, optional guide PNG files, per-frame `metrics.csv`, and `summary.json`. The reported quality metrics are:

- PSNR on grayscale images in `[0, 1]`;
- SSIM on grayscale images in `[0, 1]`;
- LPIPS after repeating grayscale inputs to three channels and mapping them to `[-1, 1]`.

When ground truth is unavailable, metric values are left empty.

Useful inference options:

- `--temporal-mode sliding-center`: stride-one windows; save the center output;
- `--temporal-mode nonoverlap-all`: disjoint windows; save every frame;
- `--max-samples N`: short smoke test;
- `--save-guide`: also export EPAW guidance maps;
- `--frames`, `--voxel-scale`: override metadata stored in the checkpoint.

## Training

Training is organized into four commands. Checkpoints and output directories can always be overridden from the command line.

### Stage 1: ET

```bash
python train_stage1_et.py \
  --data-root /path/to/EFTurb \
  --output checkpoints/stage1_et.pt
```

Stage 1 uses the non-static EFTurb scenarios and supervises multi-frame event-to-motion prediction.

### Stage 2: EPAW

```bash
python train_stage2_epaw.py \
  --data-root /path/to/EFTurb \
  --et-checkpoint checkpoints/stage1_et.pt \
  --output checkpoints/stage2_epaw.pt
```

ET is frozen in Stage 2. A Stage 2 checkpoint contains both ET and EPAW weights.

### Stage 3: restoration pretraining

```bash
python train_stage3_restoration.py \
  --efturb-root /path/to/EFTurb \
  --ctth-root /path/to/CTTH+_Dataset \
  --output checkpoints/stage3_restoration.pt
```

This stage jointly pretrains Ref-MambaTM with degraded-image gradient guidance.

### Stage 3: event-guided fine-tuning

```bash
python train_stage3_finetuning.py \
  --finetune-dataset joint \
  --efturb-root /path/to/EFTurb \
  --ctth-root /path/to/CTTH+_Dataset \
  --epaw-checkpoint checkpoints/stage2_epaw.pt \
  --restoration-pretrained checkpoints/stage3_restoration.pt \
  --output checkpoints/stage3_finetune_joint.pt
```

Use `--finetune-dataset efturb`, `ctth`, or `joint`. For an interrupted run, replace `--restoration-pretrained` with `--finetune-resume <full-training-checkpoint>`; the resume file must include optimizer, scheduler, and scaler state.

Continue from a released checkpoint; use `--epochs` to set the desired stopping point:

```bash
python train_stage3_finetuning.py \
  --finetune-dataset joint \
  --efturb-root /path/to/EFTurb \
  --ctth-root /path/to/CTTH+_Dataset \
  --finetune-resume checkpoints/restoration_event_guided_finetune_Joint.pt \
  --output checkpoints/stage3_finetune_joint.pt
```

Multi-GPU training uses `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 train_stage3_finetuning.py \
  --gpus 0,1 \
  --finetune-dataset joint \
  --efturb-root /path/to/EFTurb \
  --ctth-root /path/to/CTTH+_Dataset \
  --epaw-checkpoint checkpoints/stage2_epaw.pt \
  --restoration-pretrained checkpoints/stage3_restoration.pt
```

The number of processes must match the number of GPUs selected with `--gpus`.

Training logs contain optimization losses and only three image-quality metrics: PSNR, SSIM, and LPIPS. Validation previews and CSV logs are written under `outputs/` by default.

## Reproducibility notes

- Random seeds default to `42`.
- Event voxels use signed log normalization with `voxel_scale=4.0`.
- Flow targets use non-negative log normalization with `flow_scale=4000.0`.
- Crops must be positive and divisible by four.
- The released test manifests should be used for directly comparable evaluation.

## Acknowledgements

The implementation uses PyTorch, Mamba selective scan, causal-conv1d, timm, LPIPS, and a generalized 3D Hilbert/Gilbert curve implementation. Third-party notices retained in source files remain applicable.
