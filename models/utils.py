"""
models/utils.py
================
General-purpose utilities shared across ``train.py``, ``test.py`` and
``inference.py``:

    * Reproducibility (seeding every RNG involved)
    * Checkpoint save / load / resume
    * Early stopping
    * Running-average tracking for losses/metrics
    * Visualization: training curves, confusion matrix heatmap, prediction
      overlays, side-by-side ground-truth comparisons
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless backend — safe for Colab / servers without a display
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    """Seed every RNG that could affect reproducibility: Python's own
    `random`, NumPy, and PyTorch (CPU + all CUDA devices).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # `True` would hurt throughput substantially
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Running average tracker
# ---------------------------------------------------------------------------
class AverageMeter:
    """Tracks the running average of a scalar (e.g. loss) across an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stops training once a monitored metric (assumed "higher is better",
    e.g. mean IoU) fails to improve by at least `min_delta` for `patience`
    consecutive epochs.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update state with the latest score. Returns True if training
        should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
            else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_metric: float = 0.0,
    scaler=None,
    extra: Optional[Dict] = None,
):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if extra:
        checkpoint["extra"] = extra
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a checkpoint saved by :func:`save_checkpoint` back into the
    given model/optimizer/scheduler/scaler. Returns the raw checkpoint dict
    (useful for reading back ``epoch`` / ``best_metric``).
    """
    checkpoint = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def mask_to_color(mask: np.ndarray, class_id_to_color: Dict[int, tuple]) -> np.ndarray:
    """Convert an (H, W) integer class-index mask into an (H, W, 3) RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_id_to_color.items():
        color_mask[mask == class_id] = color
    return color_mask


def overlay_mask_on_image(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a colored segmentation mask on top of an RGB image.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) uint8 RGB image.
    color_mask : np.ndarray
        (H, W, 3) uint8 RGB colorized mask, same size as `image`.
    alpha : float
        Blend factor — 0 shows only the original image, 1 shows only the mask.
    """
    image = image.astype(np.float32)
    color_mask = color_mask.astype(np.float32)
    blended = (1 - alpha) * image + alpha * color_mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """Plot training/validation loss and mIoU/accuracy curves side-by-side
    and save the figure to `save_path`.

    `history` is expected to contain (at least) the keys:
    ``train_loss``, ``val_loss``, ``val_miou``, ``val_pixel_accuracy``.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "train_loss" in history and "val_loss" in history:
        axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
        axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    if "val_miou" in history:
        axes[0, 1].plot(epochs, history["val_miou"], label="Val mIoU", color="tab:green")
        axes[0, 1].set_title("Mean IoU")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mIoU")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    if "val_pixel_accuracy" in history:
        axes[1, 0].plot(epochs, history["val_pixel_accuracy"], label="Val Pixel Accuracy", color="tab:red")
        axes[1, 0].set_title("Pixel Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    if "val_mean_f1" in history:
        axes[1, 1].plot(epochs, history["val_mean_f1"], label="Val Mean F1", color="tab:purple")
        axes[1, 1].set_title("Mean F1 Score")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm_normalized: np.ndarray, class_names: List[str], save_path: str):
    """Plot a row-normalized confusion matrix heatmap and save to disk."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_normalized[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_prediction_grid(
    images: List[np.ndarray],
    gt_masks_color: List[np.ndarray],
    pred_masks_color: List[np.ndarray],
    overlays: List[np.ndarray],
    save_path: str,
    max_samples: int = 4,
):
    """Save a grid of [input image | ground truth | prediction | overlay]
    rows, one row per sample, for qualitative inspection.
    """
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 4)

    col_titles = ["Input Image", "Ground Truth", "Prediction", "Overlay"]
    for row in range(n):
        panels = [images[row], gt_masks_color[row], pred_masks_color[row], overlays[row]]
        for col in range(4):
            axes[row, col].imshow(panels[col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_alpha_map(alpha_map: np.ndarray, save_path: str, title: str = "AADFF mixing coefficient (alpha)"):
    """Visualize a single-channel-averaged AADFF alpha map (CNN vs Mamba
    contribution) as a heatmap — useful for qualitatively inspecting what
    the fusion module has learned to rely on, region by region.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(alpha_map, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("alpha (1 = CNN, 0 = Vision Mamba)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    seed_everything(42)
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(2.0)
    print(f"AverageMeter avg: {meter.avg}")

    stopper = EarlyStopping(patience=2)
    print(stopper.step(0.5), stopper.step(0.4), stopper.step(0.3))
