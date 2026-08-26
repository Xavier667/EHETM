# High-Quality and Efficient Turbulence Mitigation with Events [CVPR 2026 Highlight]
# High Temporal Resolution Matters: High-Quality and Efficient Turbulence Mitigation with Events [TPAMI Under Review]

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_High-Quality_and_Efficient_Turbulence_Mitigation_with_Events_CVPR_2026_paper.pdf"><img src="https://img.shields.io/badge/CVPR_2026-Paper-1f6feb.svg" alt="CVPR 2026 paper"></a>
  <a href="https://youtu.be/oZrPr5Mmn6c?si=PvmLqcTke8P2lcFD"><img src="https://img.shields.io/badge/Video-YouTube-ff0000.svg?logo=youtube" alt="Video"></a>
  <a href="#datasets"><img src="https://img.shields.io/badge/Datasets-Events%20%26%20Images-2ea44f.svg" alt="Datasets"></a>
  <a href="#code-and-models"><img src="https://img.shields.io/badge/Code-PyTorch-ee4c2c.svg?logo=pytorch" alt="Code"></a>
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_High-Quality_and_Efficient_Turbulence_Mitigation_with_Events_CVPR_2026_paper.pdf">Paper</a> ·
  <a href="#video">Video</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#environment">Installation</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#training">Training</a> ·
  <a href="#citation">Citation</a>
</p>

Official PyTorch implementation of **EHETM**, an event-guided framework for high-quality and efficient turbulence mitigation. The repository includes the complete staged training pipeline, unified inference for simulated and real-world data, pretrained checkpoints, and resources for the CVPR 2026 datasets **CTTH** and **LATH** and the TPAMI-extension datasets **EFTSim** and **CTTH+**. **LATH+** will be released later.

## News

- **2026-08-25:** **EFTSim and CTTH+ are now publicly available** from the [dataset downloads](#datasets). LATH+ will be released later.
- **2026-08-19:** Training and inference code is now available.
- **2026-06-26:** An extended version of this work has been submitted to **TPAMI** and is currently **under review**. The extension introduces EFTSim, CTTH+, and LATH+.
- **2026-06-05:** The official CVPR 2026 paper is now available from [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_High-Quality_and_Efficient_Turbulence_Mitigation_with_Events_CVPR_2026_paper.pdf).
- **2026-03-27:** The CTTH and LATH datasets are released.
- **2026-03-25:** The paper is published on [arXiv](https://arxiv.org/abs/2603.20708).
- **2026-02-23:** EHETM is accepted by CVPR 2026 as a Highlight paper.

## Video

<p align="center">
  <a href="https://youtu.be/oZrPr5Mmn6c?si=PvmLqcTke8P2lcFD">
    <img src="https://img.youtube.com/vi/oZrPr5Mmn6c/maxresdefault.jpg" width="80%" alt="EHETM project video">
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/oZrPr5Mmn6c?si=PvmLqcTke8P2lcFD"><strong>▶ Watch the EHETM project video on YouTube</strong></a>
</p>

## Datasets

This repository covers five event-based turbulence datasets. CTTH and LATH were introduced with the CVPR 2026 paper. EFTSim, CTTH+, and LATH+ form the dataset suite developed for the TPAMI extension, which is currently under review.

| Dataset | Release group | Event representation | Availability |
| --- | --- | --- | --- |
| **CTTH** | CVPR 2026 | Time-sliced events | [Baidu Netdisk](https://pan.baidu.com/s/1XsDaJTYYfcgNENzEL0_wqw?pwd=qaz3) (code: `qaz3`) |
| **LATH** | CVPR 2026 | Time-sliced events | [Baidu Netdisk](https://pan.baidu.com/s/1XsDaJTYYfcgNENzEL0_wqw?pwd=qaz3) (code: `qaz3`) |
| **EFTSim** | TPAMI extension | Complete event streams + IMU ego-motion | [Event_Turb_Datasets](https://pan.baidu.com/s/1anYvXzc6in3YCowZ2SBBGw?pwd=qaz3) (code: `qaz3`) |
| **CTTH+** | TPAMI extension | Complete event streams + IMU ego-motion | [Event_Turb_Datasets](https://pan.baidu.com/s/1anYvXzc6in3YCowZ2SBBGw?pwd=qaz3) (code: `qaz3`) |
| **LATH+** | TPAMI extension | Complete event streams + IMU ego-motion | **Coming soon** |

### CTTH

The **CTTH** dataset contains static and dynamic-object scenes captured under controlled turbulence. It provides synchronized intensity frames, time-sliced events, timestamps, ground truth, and optical flow for dynamic-object scenes.

### LATH

The **LATH** dataset contains synchronized frame and time-sliced event measurements captured over real atmospheric paths ranging from 3.5 km to 8 km. It is intended for evaluating generalization under realistic long-range turbulence.

### EFTSim

**EFTSim** is a simulated event-based turbulence dataset developed for the TPAMI extension. It provides complete event streams, IMU-based platform ego-motion measurements, clean reference frames, degraded frames, and optical-flow or clean-image-gradient supervision. For compatibility, EFTSim retains the internal identifier **EFTurb** (`efturb`) in the released code, command-line options, split filenames, and checkpoints.

### CTTH+

**CTTH+** extends the controlled real-world turbulence data for the TPAMI study. In contrast to the time-sliced events in CTTH, CTTH+ provides complete event streams, IMU-based platform ego-motion measurements, clean and degraded frames, and motion or gradient supervision where available.

### LATH+

**LATH+** is the extended long-range real-world dataset developed for the TPAMI study. It provides complete event streams and IMU-based platform ego-motion measurements for long-range atmospheric-turbulence sequences.

> **Release status:** LATH+ is not yet publicly available. The download will be added here—stay tuned.

### Data formats

#### CTTH and LATH: time-sliced events

```text
Dataset/
├── CTTH/
│   ├── Dynamic_Object/
│   │   ├── Train/
│   │   │   ├── seq_000/
│   │   │   │   ├── GT/
│   │   │   │   │   ├── frames/
│   │   │   │   │   ├── events/
│   │   │   │   │   ├── frame_timestamp.txt
│   │   │   │   │   └── event_timestamp.txt
│   │   │   │   ├── Turb/
│   │   │   │   │   ├── frames/
│   │   │   │   │   ├── events/
│   │   │   │   │   ├── frame_timestamp.txt
│   │   │   │   │   └── event_timestamp.txt
│   │   │   │   └── Flow/
│   │   │   └── ...
│   │   └── Test/
│   │       ├── seq_000/
│   │       └── ...
│   └── Static/
│       ├── Train/
│       │   ├── seq_000/
│       │   │   ├── turb/
│       │   │   ├── event/
│       │   │   ├── frame_timestamp.txt
│       │   │   ├── event_timestamp.txt
│       │   │   └── gt.jpg
│       │   └── ...
│       └── Test/
│           ├── seq_000/
│           └── ...
└── LATH/
    ├── seq_000/
    │   ├── turb/
    │   ├── events/
    │   ├── frame_timestamp.txt
    │   └── event_timestamp.txt
    └── ...
```

Each CTTH or LATH sequence contains synchronized frame images, event data, and corresponding timestamps. Directory names vary slightly across subsets, as shown above.

- **`frames/` or `turb/`:** intensity images captured at a fixed frame rate of 25 Hz.
- **`events/` or `event/`:** time-sliced event-camera measurements encoded as positive event = `200`, negative event = `100`, and background = `0`.
- **`frame_timestamp.txt`:** timestamps for the intensity frames.
- **`event_timestamp.txt`:** timestamps for the event slices.
- **`Flow/`** (when available): ground-truth optical flow for dynamic-object scenes.

CTTH and LATH were captured with the [ALPIX-Pizol camera](https://www.alpsentek.com/), which outputs events in a time-sliced format rather than as fully asynchronous streams. All events in one slice share a timestamp and represent the events accumulated over a short 1 ms window. This representation should be considered during voxelization and temporal alignment. The provided [`tools/event_processing.py`](tools/event_processing.py) script converts the time-sliced events into voxels and polarity-alternation statistics.

#### EFTSim, CTTH+, and LATH+: complete event streams with IMU

The TPAMI-extension datasets share the following high-level organization. Exact subdirectory names may vary slightly between `Flow/` and `Optical_Flow/`, but the contents follow the same logic.

```text
<dataset>/
└── <sequence>/
    ├── GT/
    │   └── frames/                         # Clean reference images
    ├── Flow/ or Optical_Flow/
    │   ├── optical_flow/                   # Motion of dynamic scenes or objects
    │   └── clean_image_gradient/           # Gradients of the clean images
    └── Turb/
        ├── frames/                         # Turbulence-degraded images
        ├── events_raw/                     # Original complete event stream
        ├── voxel_raw/                      # Voxels built from the original events
        ├── events_ego_motion_compensated/  # Ego-motion-compensated events
        ├── voxel_ego_motion_compensated/   # Voxels built after compensation
        └── ego_motion/                     # IMU platform ego-motion measurements
```

The `Flow/` or `Optical_Flow/` directory may be omitted when optical-flow or gradient supervision is not applicable. The `Turb/` directory preserves both the original and ego-motion-compensated event representations so that compensation and alignment strategies can be evaluated consistently.

## Code and Models

### Repository structure

```text
EHETM/
├── Code/
│   ├── Data/                         # EFTSim (EFTurb in code) and CTTH+ loaders
│   │   └── splits/                   # Deterministic test manifests
│   ├── Model/                        # ET, EPAW, and Ref-MambaTM
│   ├── Utils/                        # Metrics and training utilities
│   ├── infer_final.py                # Unified inference and evaluation
│   ├── train_stage1_et.py
│   ├── train_stage2_epaw.py
│   ├── train_stage3_restoration.py
│   ├── train_stage3_finetuning.py
│   └── requirements.txt
├── assets/                            # README figures and examples
└── tools/
    └── event_processing.py            # Event preprocessing utility
```

### Environment

The code has been validated with:

- Ubuntu 18.04
- Python 3.11.13
- PyTorch 2.1.1 with CUDA 12.1
- NVIDIA RTX 3090

An NVIDIA GPU is required by the selective-scan and causal-convolution kernels used in Ref-MambaTM. A matching CUDA toolkit and a working C/C++ compiler are required when building `mamba-ssm` and `causal-conv1d`.

```bash
git clone https://github.com/Xavier667/EHETM.git
cd EHETM/Code

conda create -n ehetm python=3.11 -y
conda activate ehetm

# Install PyTorch first so the CUDA extensions can compile against it.
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

LPIPS initializes an AlexNet backbone on first use and may download its pretrained weights when they are not already cached.

### Checkpoints

The three released checkpoints retain the model parameters and the optimizer, scheduler, AMP scaler, and training-progress states required to continue training. The public filenames intentionally do not encode training epochs.

| Checkpoint | Training domain | Baidu Netdisk |
| --- | --- | --- |
| `restoration_event_guided_finetune_CTTH.pt` | CTTH+ | [Baidu Netdisk](https://pan.baidu.com/s/1l4Dglqffld-TfRbDI9E_qw) (code: `qaz3`) |
| `restoration_event_guided_finetune_EFTURB.pt` | EFTSim (`EFTurb` in code) | [Baidu Netdisk](https://pan.baidu.com/s/1l4Dglqffld-TfRbDI9E_qw) (code: `qaz3`) |
| `restoration_event_guided_finetune_Joint.pt` | Joint | [Baidu Netdisk](https://pan.baidu.com/s/1l4Dglqffld-TfRbDI9E_qw) (code: `qaz3`) |

Download and extract the checkpoint package, then place the three `.pt` files in `Code/checkpoints/`. Every checkpoint can be used directly for inference or passed to the fine-tuning script with `--finetune-resume`.

### Model-ready data layout

The released training loaders consume the following preprocessed subsets of EFTSim and CTTH+. The complete dataset packages additionally provide the original event streams, ego-motion-compensated events, corresponding voxels, and IMU ego-motion measurements described in [Data formats](#data-formats). EFTSim retains the internal identifier `EFTurb` for compatibility with the code and checkpoints. Dataset locations are supplied through command-line arguments.

<details>
<summary><strong>EFTSim layout (EFTurb in code)</strong></summary>

```text
EFTSim/
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

</details>

<details>
<summary><strong>CTTH+ layout</strong></summary>

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

</details>

Expected NPZ fields:

- Event stream: `x`, `y`, `p`, and `t`.
- Event voxel: `voxel`, with shape `[2, 10, H, W]`.
- Scalar flow target: `raw_gradient_flow`, with shape `[1, H, W]`.

Frame stems must be numeric. The default deterministic split uses seed `42` and a `0.9` training ratio. The released test sequence lists are in `Code/Data/splits/`. See `tools/event_processing.py` for event preprocessing.

## Inference

Run the following commands from `EHETM/Code`.

### EFTSim (EFTurb in code)

```bash
python infer_final.py \
  --checkpoint checkpoints/restoration_event_guided_finetune_EFTURB.pt \
  --data-format efturb \
  --data-root /path/to/EFTSim \
  --test-list Data/splits/efturb_test_seed42_ratio0.9.txt \
  --output outputs/efturb \
  --tile-size 512 --overlap 64 --amp
```

### CTTH+

```bash
python infer_final.py \
  --checkpoint checkpoints/restoration_event_guided_finetune_CTTH.pt \
  --data-format ctth \
  --data-root /path/to/CTTH+_Dataset \
  --test-list Data/splits/ctth_test_seed42_ratio0.9.txt \
  --output outputs/ctth \
  --tile-size 512 --overlap 0 --amp
```

The joint checkpoint uses the same command with `restoration_event_guided_finetune_Joint.pt` and either supported `--data-format`.

### Real event-camera data

Ground truth is optional for real measurements:

```bash
python infer_final.py \
  --checkpoint checkpoints/restoration_event_guided_finetune_Joint.pt \
  --data-format real \
  --data-root /path/to/real_sequences \
  --gt none \
  --output outputs/real \
  --tile-size 512 --overlap 64 --amp --save-guide
```

Each real sequence may use `frames/`, `events/`, and `event_voxel/`; the EFTSim and CTTH+ layouts are also accepted. If ground truth is present, use `gt/` or `ground_truth/` and retain the default `--gt auto`.

Inference writes restored PNG files, optional guidance maps, per-frame `metrics.csv`, and `summary.json`. Only the following image-quality metrics are reported:

- PSNR on grayscale images in `[0, 1]`;
- SSIM on grayscale images in `[0, 1]`;
- LPIPS after converting grayscale inputs to three channels in `[-1, 1]`.

When ground truth is unavailable, metric values are left empty.

Useful options include `--temporal-mode sliding-center`, `--temporal-mode nonoverlap-all`, `--max-samples N`, `--save-guide`, `--frames`, and `--voxel-scale`. Run `python infer_final.py --help` for the complete interface.

## Training

All training scripts support command-line data and output paths. Run commands from `EHETM/Code`.

<details open>
<summary><strong>Stage 1: Event-to-motion network (ET)</strong></summary>

```bash
python train_stage1_et.py \
  --data-root /path/to/EFTSim \
  --output checkpoints/stage1_et.pt
```

Stage 1 uses the non-static EFTSim scenarios and supervises multi-frame event-to-motion prediction.

</details>

<details>
<summary><strong>Stage 2: Event prior alignment and weighting (EPAW)</strong></summary>

```bash
python train_stage2_epaw.py \
  --data-root /path/to/EFTSim \
  --et-checkpoint checkpoints/stage1_et.pt \
  --output checkpoints/stage2_epaw.pt
```

ET is frozen in Stage 2. The resulting checkpoint contains both ET and EPAW weights.

</details>

<details>
<summary><strong>Stage 3: Restoration pretraining</strong></summary>

```bash
python train_stage3_restoration.py \
  --efturb-root /path/to/EFTSim \
  --ctth-root /path/to/CTTH+_Dataset \
  --output checkpoints/stage3_restoration.pt
```

This stage jointly pretrains Ref-MambaTM with degraded-image gradient guidance.

</details>

<details>
<summary><strong>Stage 3: Event-guided fine-tuning</strong></summary>

```bash
python train_stage3_finetuning.py \
  --finetune-dataset joint \
  --efturb-root /path/to/EFTSim \
  --ctth-root /path/to/CTTH+_Dataset \
  --epaw-checkpoint checkpoints/stage2_epaw.pt \
  --restoration-pretrained checkpoints/stage3_restoration.pt \
  --output checkpoints/stage3_finetune_joint.pt
```

Set `--finetune-dataset` to `efturb`, `ctth`, or `joint`.

</details>

### Continue training from a released checkpoint

Use `--finetune-resume` to restore the model, optimizer, scheduler, and scaler states. Set `--epochs` to the desired stopping point when needed.

```bash
python train_stage3_finetuning.py \
  --finetune-dataset joint \
  --efturb-root /path/to/EFTSim \
  --ctth-root /path/to/CTTH+_Dataset \
  --finetune-resume checkpoints/restoration_event_guided_finetune_Joint.pt \
  --output checkpoints/stage3_finetune_joint.pt
```

### Multi-GPU training

```bash
torchrun --standalone --nproc_per_node=2 train_stage3_finetuning.py \
  --gpus 0,1 \
  --finetune-dataset joint \
  --efturb-root /path/to/EFTSim \
  --ctth-root /path/to/CTTH+_Dataset \
  --epaw-checkpoint checkpoints/stage2_epaw.pt \
  --restoration-pretrained checkpoints/stage3_restoration.pt
```

The number of `torchrun` processes must match the number of GPUs selected with `--gpus`. Training logs and validation outputs report PSNR, SSIM, and LPIPS only.

## Citation

If this work or the released datasets are useful in your research, please cite our paper and consider starring the repository.

```bibtex
@InProceedings{Zhang_2026_CVPR,
    author    = {Zhang, Xiaoran and Ding, Jian and Duan, Yuxing and Liu, Haoyue and Chen, Gang and Chang, Yi and Yan, Luxin},
    title     = {High-Quality and Efficient Turbulence Mitigation with Events},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {29514-29525}
}
```
