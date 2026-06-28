# DFF-MambaNet

### Adaptive Attention-Based Dynamic Feature Fusion in Hybrid Mamba-CNN Networks for High-Resolution Satellite Image Segmentation

A research-grade, fully-implemented PyTorch project combining a CNN encoder
(ResNet34) with a Vision Mamba (Selective State Space Model) encoder via a
novel **Adaptive Attention-Based Dynamic Feature Fusion (AADFF)** module, for
semantic segmentation of high-resolution satellite imagery on the **ISPRS
Potsdam** dataset.

Built for a B.Tech summer internship project, structured so it can be
extended directly into a research paper.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation & Gap](#2-research-motivation--gap)
3. [Architecture](#3-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Training](#7-training)
8. [Testing / Evaluation](#8-testing--evaluation)
9. [Inference](#9-inference)
10. [Results](#10-results)
11. [Future Improvements](#11-future-improvements)
12. [References](#12-references)

---

## 1. Project Overview

High-resolution satellite/aerial image segmentation requires a model that can
simultaneously resolve **fine local detail** (sharp object boundaries, small
objects like cars) and **long-range global context** (large, texturally
homogeneous regions like rooftops, fields, or water bodies). CNNs are
excellent at the former but limited by their local receptive field; Vision
Mamba (a vision adaptation of Selective State Space Models) excels at the
latter via near-linear-complexity long-range sequence modeling, but is
comparatively weaker at precise local detail.

**DFF-MambaNet** runs both branches in parallel over the same input image and
fuses them at every pyramid scale using **AADFF**, a learned, input-dependent
gating mechanism — rather than the static concatenation/summation used by
most existing hybrid CNN-SSM segmentation works.

```
Fusion = alpha ⊙ CNN_refined + (1 - alpha) ⊙ Mamba_refined
```

where `alpha` is a full per-pixel, per-channel map predicted dynamically from
both branches' (attention-refined) features — not a fixed scalar.

---

## 2. Research Motivation & Gap

| Limitation                                            | Addressed by |
|--------------------------------------------------------|--------------|
| CNNs have limited receptive field → weak global context | Vision Mamba branch (selective-scan SSM, near-linear complexity in sequence length) |
| Vision Mamba is comparatively weak at fine local detail / sharp boundaries | Parallel ResNet34 CNN branch supplies local detail |
| Existing hybrid CNN-Mamba segmentation works (e.g. RS3Mamba, Samba, and related Vision-Mamba remote-sensing models) generally fuse the two branches with a **static** rule (fixed concatenation + conv, or a fixed-weight sum) | **AADFF**: a channel+spatial attention-refined, *dynamically predicted*, per-pixel/per-channel mixing coefficient |

**Research contribution of this project:** the **AADFF module**
(`models/fusion.py`), which is the only architecturally novel component —
the CNN branch, the Vision Mamba branch, and the U-Net decoder are all
faithful, clearly-separated reference implementations of established
techniques, so that the contribution of AADFF can be cleanly isolated and
ablated.

---

## 3. Architecture

```
                                 Input Image (B, 3, H, W)
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
     ┌─────────────────────┐                           ┌─────────────────────────┐
     │   CNN Encoder        │                           │  Vision Mamba Encoder    │
     │   (ResNet34,          │                           │  (Selective-Scan SSM,   │
     │    ImageNet-pretrained)│                          │   bidirectional blocks)  │
     │                       │                           │                          │
     │  stage1: 64ch,  s=4   │                           │  stage1: 64ch,  s=4      │
     │  stage2: 128ch, s=8   │                           │  stage2: 128ch, s=8      │
     │  stage3: 256ch, s=16  │                           │  stage3: 256ch, s=16     │
     │  stage4: 512ch, s=32  │                           │  stage4: 512ch, s=32     │
     └──────────┬────────────┘                           └────────────┬─────────────┘
                │  f1..f4 (local)                                     │ m1..m4 (global)
                └───────────────────────┬─────────────────────────────┘
                                         v
                       ┌───────────────────────────────────────┐
                       │   AADFF  (one instance per stage)       │
                       │  ─────────────────────────────────────  │
                       │  1. Channel Attention  (per branch)      │
                       │  2. Spatial Attention  (per branch)      │
                       │  3. Adaptive gate: alpha = f(concat)     │
                       │  4. Fuse = alpha*CNN + (1-alpha)*Mamba   │
                       └───────────────────┬───────────────────┘
                                fused1..fused4 (stride 4,8,16,32)
                                           │
                                           v
                            ┌───────────────────────────┐
                            │     U-Net Decoder           │
                            │  (skip connections from      │
                            │   fused1, fused2, fused3)    │
                            └─────────────┬─────────────┘
                                           v
                         Per-pixel logits (B, 6, H, W)
                                           │
                                       argmax
                                           v
                  Segmentation Mask: Impervious / Building /
                  Low Vegetation / Tree / Car / Clutter
```

### 3.1 CNN Branch (`models/encoder.py`)

A standard, ImageNet-pretrained **ResNet34**, exposing `layer1..layer4` as a
4-level feature pyramid (strides 4/8/16/32, channels 64/128/256/512).

### 3.2 Vision Mamba Branch (`models/vision_mamba.py`)

A from-scratch Vision Mamba encoder built to mirror the CNN branch's pyramid
exactly (same 4 strides/channel widths, so the two can be fused
element-wise). Each stage flattens its feature map into a token sequence and
runs it through bidirectional **selective-scan SSM blocks**.

Two backends are supported automatically:

- **Official kernel**: if [`mamba-ssm`](https://github.com/state-spaces/mamba)
  is importable (requires a CUDA GPU), it is used directly for maximum speed.
- **Pure-PyTorch fallback**: otherwise, `SimpleSelectiveSSM` implements the
  *exact same* selective-scan recurrence using an **O(log L) Hillis-Steele
  parallel prefix scan** (rather than a naive O(L) Python loop), which is
  mathematically exact and tractable in both compute and memory on CPU,
  Windows, or any GPU. A token-count cap (`MAMBA_MAX_TOKENS_FALLBACK`)
  additionally pools very long sequences down before the scan and upsamples
  the result back, purely to bound memory when the official kernel isn't
  available.

### 3.3 AADFF — Adaptive Attention-Based Dynamic Feature Fusion (`models/fusion.py`)

**This is the project's research contribution.** At every one of the 4
pyramid stages:

1. `BranchAttentionRefiner` applies CBAM-style **channel attention**
   (squeeze-and-excite, average+max pooled) and **spatial attention**
   (average+max channel statistics → conv) independently to the CNN feature
   map and the Mamba feature map.
2. The two refined maps are concatenated and fed through a lightweight
   gating network (1x1 convs) ending in a sigmoid, producing a per-pixel,
   per-channel coefficient **alpha ∈ (0, 1)**.
3. The branches are fused as a convex combination:
   `fused = alpha * cnn_refined + (1 - alpha) * mamba_refined`.
4. A final 3x3 conv + BatchNorm + ReLU smooths the result.

Because `alpha` is predicted fresh for every image (and varies spatially and
per-channel), the network can, e.g., lean on the CNN branch right at a
building edge while leaning on the Mamba branch in the middle of a large
homogeneous field — something a fixed fusion rule cannot do. `alpha` maps can
be visualized directly (`network(x, return_alpha_maps=True)`) for qualitative
analysis of what the model has learned to rely on.

### 3.4 U-Net Decoder (`models/decoder.py`)

A standard U-Net decoder: 3 `UpBlock`s (transposed-conv upsample → concat
skip connection from the matching fused stage → double-conv), followed by a
final 4x upsampling head and a 1x1 classifier producing 6-class logits at the
input resolution.

---

## 4. Repository Structure

```
DFF-MambaNet/
├── configs/
│   ├── config.py            # All hyperparameters, paths, and device/Drive helpers
│   └── __init__.py
├── dataset/
│   ├── dataset.py           # PotsdamDataset (PyTorch Dataset) + dataloader factory
│   ├── preprocess.py         # Tiling (256x256) + RGB-label -> class-index conversion
│   ├── augmentations.py      # Albumentations train/val/inference pipelines
│   └── __init__.py
├── models/
│   ├── encoder.py            # ResNet34 CNN branch
│   ├── vision_mamba.py       # Vision Mamba branch (selective-scan SSM)
│   ├── fusion.py             # *** AADFF — the research contribution ***
│   ├── decoder.py            # U-Net decoder
│   ├── network.py            # DFFMambaNet — wires everything together
│   ├── losses.py             # CrossEntropy / Dice / Hybrid losses
│   ├── metrics.py            # Confusion-matrix-based segmentation metrics
│   ├── utils.py              # Seeding, checkpoints, early stopping, plotting
│   └── __init__.py
├── train.py                   # Training entry point
├── test.py                    # Evaluation entry point
├── inference.py                # Single-image inference (with sliding-window tiling)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Three additional directories are created automatically (and are
`.gitignore`d apart from a `.gitkeep`):

```
checkpoints/   # best_model.pth, last_model.pth
logs/          # TensorBoard event files
results/       # training curves, confusion matrices, qualitative samples, CSV metrics
```

---

## 5. Installation

### 5.1 Google Colab

```python
!git clone <your-fork-url> DFF-MambaNet
%cd DFF-MambaNet
!pip install -r requirements.txt
```

To use your dataset stored on Google Drive:

```python
from configs.config import Config
Config.mount_google_drive()   # mounts at /content/drive, no-ops outside Colab
```

### 5.2 Ubuntu Linux / Windows

```bash
git clone <your-fork-url> DFF-MambaNet
cd DFF-MambaNet
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 (Optional) Official Mamba CUDA kernel

The project runs correctly everywhere using a pure-PyTorch fallback SSM (see
[§3.2](#32-vision-mamba-branch-modelsvision_mambapy)). For maximum training
speed on a CUDA GPU, you may additionally install the official kernel:

```bash
pip install causal-conv1d>=1.1.0
pip install mamba-ssm>=1.2.0
```

This requires a CUDA-enabled GPU and a matching toolkit; if installation
fails or no GPU is present, simply skip this step — the fallback is used
automatically (with a one-time printed notice).

GPU/CPU/MPS device selection is automatic via `Config.get_device()` — no
manual configuration required.

---

## 6. Dataset Preparation

1. Download the **ISPRS Potsdam** dataset (2D Semantic Labeling benchmark).
   It ships as a set of full-resolution `6000x6000` RGB "TOP" tiles and
   matching RGB-coded ground-truth label tiles.
2. Place them as:

   ```
   data/raw/images/top_potsdam_2_10.tif
   data/raw/labels/top_potsdam_2_10_label.tif
   ...
   ```

3. Run the preprocessing script to tile everything into `256x256` patches,
   convert RGB labels into class-index masks, and build the train/val split:

   ```bash
   python -m dataset.preprocess \
       --raw_image_dir data/raw/images \
       --raw_label_dir data/raw/labels \
       --patch_size 256 \
       --val_ratio 0.2
   ```

   This writes patches to `data/patches/{images,labels}/` and two manifest
   files, `data/patches/train.csv` and `data/patches/val.csv` (split at the
   *source-tile* level to avoid leaking adjacent patches between splits).

### Class palette

| Class | RGB Color | Index |
|---|---|---|
| Impervious Surface | `(255, 255, 255)` | 0 |
| Building | `(0, 0, 255)` | 1 |
| Low Vegetation | `(0, 255, 255)` | 2 |
| Tree | `(0, 255, 0)` | 3 |
| Car | `(255, 255, 0)` | 4 |
| Clutter / Background | `(255, 0, 0)` | 5 |

---

## 7. Training

```bash
python train.py
```

Common overrides:

```bash
python train.py --epochs 100 --batch_size 8 --lr 3e-4 --loss_type hybrid --optimizer adamw
python train.py --resume checkpoints/last_model.pth     # resume training
python train.py --no_pretrained                          # offline / no internet
python train.py --no_amp                                 # disable mixed precision
```

All other hyperparameters (loss weights, scheduler type, warmup, gradient
clipping, early-stopping patience, AADFF reduction ratio, Mamba SSM
dimensions, etc.) live in `configs/config.py` — edit directly for anything
not exposed as a CLI flag.

What you get during/after training:

- **TensorBoard logs** under `logs/<experiment_name>/` — run
  `tensorboard --logdir logs` to view live curves.
- **`checkpoints/best_model.pth`** (highest validation mIoU so far) and
  **`checkpoints/last_model.pth`** (for resuming), each containing model,
  optimizer, scheduler, and AMP-scaler state.
- **`results/epoch_metrics.csv`** — one row per epoch (loss, mIoU, pixel
  accuracy, precision/recall/F1, learning rate, epoch time).
- **`results/training_curves.png`** and **`results/confusion_matrix.png`**,
  generated automatically once training finishes (or early-stops).

---

## 8. Testing / Evaluation

```bash
python test.py --checkpoint checkpoints/best_model.pth
python test.py --checkpoint checkpoints/best_model.pth --csv data/patches/val.csv --num_samples 8
```

Produces:

- Console summary: mean IoU, pixel accuracy, mean precision/recall/F1, and
  full per-class IoU breakdown.
- `results/test_classification_report.csv` — per-class IoU/precision/recall/F1.
- `results/test_confusion_matrix.png` — normalized confusion matrix heatmap.
- `results/test_qualitative_samples.png` — a grid of
  `[input image | ground truth | prediction | overlay]` rows.

---

## 9. Inference

Run the trained model on a single satellite image of **any size** (the
script automatically applies overlap-feathered sliding-window tiling for
images larger than the training patch size):

```bash
python inference.py --image path/to/satellite_tile.tif --checkpoint checkpoints/best_model.pth
python inference.py --image path/to/large_tile.tif --tile_size 256 --overlap 32 --output_dir results/inference
```

Outputs (saved to `--output_dir`, default `results/inference/`):

- `<name>_mask.png` — single-channel predicted class-index mask.
- `<name>_mask_color.png` — colorized segmentation map (ISPRS palette).
- `<name>_overlay.png` — colorized prediction alpha-blended over the input.

A per-class pixel-coverage summary is also printed to the console.

---

## 10. Results

This repository ships fully implemented and tested (including a full
synthetic end-to-end smoke test of preprocessing → training → checkpointing
→ evaluation → inference), but does **not** ship pretrained weights or
benchmark numbers, since these depend on the actual ISPRS Potsdam download
and your chosen training budget.

After training on the real dataset, this section is intended to be filled in
with:

- A table of final mean IoU / pixel accuracy / per-class IoU on the
  validation split.
- The `results/training_curves.png` and `results/confusion_matrix.png` plots.
- A few rows from `results/test_qualitative_samples.png`.
- An ablation comparing AADFF against a static-fusion baseline (e.g. plain
  concatenation + conv) — the natural next experiment given the project's
  research framing.

---

## 11. Future Improvements

- **Ablation studies**: AADFF vs. static concatenation, channel-attention-only
  vs. spatial-attention-only, scalar vs. per-pixel vs. per-channel `alpha`.
- **Stronger backbones**: swap ResNet34 for ResNet50/EfficientNet via `timm`,
  or swap the hand-rolled Vision Mamba branch for a pretrained Vim/VMamba
  checkpoint.
- **Multi-dataset evaluation**: ISPRS Vaihingen, LoveDA, or other remote-
  sensing benchmarks, to test generalization of the fusion mechanism.
- **Boundary-aware losses** (e.g. boundary IoU, Lovász-Softmax) to further
  sharpen object edges, complementing AADFF's spatial attention.
- **Test-time augmentation** and **multi-scale inference** for a further
  accuracy boost at evaluation time.
- **Quantitative alpha-map analysis**: correlate AADFF's predicted mixing
  coefficient with object class / boundary proximity, to quantitatively
  support the qualitative motivation in §2.

---

## 12. References

This implementation draws conceptual inspiration from the following lines of
work (no code was copied from any of them — every file in this repository is
an original implementation):

1. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with
   Selective State Spaces.*
2. Zhu, L. et al. (2024). *Vision Mamba: Efficient Visual Representation
   Learning with Bidirectional State Space Model.*
3. Liu, Y. et al. (2024). *VMamba: Visual State Space Model.*
4. Ma, X. et al. (2024). *RS3Mamba: Visual State Space Model for Remote
   Sensing Image Semantic Segmentation.*
5. *Samba: Semantic Segmentation of Remotely Sensed Images with State Space
   Model.*
6. Woo, S. et al. (2018). *CBAM: Convolutional Block Attention Module.*
   (channel + spatial attention design used inside AADFF)
7. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for
   Biomedical Image Segmentation.*
8. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.*
   (ResNet)
9. ISPRS 2D Semantic Labeling Contest — Potsdam dataset.

---

## License

Released under the [MIT License](LICENSE).
