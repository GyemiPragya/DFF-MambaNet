"""
test.py
=======
Evaluation entry point for DFF-MambaNet.

Loads a trained checkpoint, runs inference over an evaluation split
(defaults to the validation split — point ``--csv`` at a held-out test CSV
if you have one), and reports/saves:

    * Per-class and mean IoU, precision, recall, F1
    * Pixel / overall accuracy
    * A confusion matrix heatmap
    * A CSV classification report
    * A qualitative grid of [input | ground truth | prediction | overlay]
      samples

Usage
-----
    python test.py --checkpoint checkpoints/best_model.pth
    python test.py --checkpoint checkpoints/best_model.pth --csv data/patches/val.csv
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from configs.config import Config
from dataset.dataset import PotsdamDataset
from dataset.augmentations import get_val_augmentations
from torch.utils.data import DataLoader

from models.network import build_model
from models.metrics import SegmentationMetrics
from models.utils import (
    load_checkpoint,
    mask_to_color,
    overlay_mask_on_image,
    plot_confusion_matrix,
    save_prediction_grid,
)


def denormalize_image(image_tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Undo ImageNet normalization on a (3, H, W) tensor and return an
    (H, W, 3) uint8 RGB numpy array, ready for visualization.
    """
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = (image * std + mean) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def write_classification_report_csv(results: dict, class_names, save_path: str):
    """Write a per-class metrics table (IoU / precision / recall / F1) to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou", "precision", "recall", "f1"])
        for name in class_names:
            writer.writerow([
                name,
                f"{results['per_class_iou'][name]:.4f}",
                f"{results['per_class_precision'][name]:.4f}",
                f"{results['per_class_recall'][name]:.4f}",
                f"{results['per_class_f1'][name]:.4f}",
            ])
        writer.writerow([])
        writer.writerow(["mean_iou", f"{results['mean_iou']:.4f}"])
        writer.writerow(["pixel_accuracy", f"{results['pixel_accuracy']:.4f}"])
        writer.writerow(["mean_precision", f"{results['mean_precision']:.4f}"])
        writer.writerow(["mean_recall", f"{results['mean_recall']:.4f}"])
        writer.writerow(["mean_f1", f"{results['mean_f1']:.4f}"])


@torch.no_grad()
def run_evaluation(config: Config, checkpoint_path: str, csv_path: str, num_qualitative_samples: int = 8):
    device = config.DEVICE
    print(f"[test] Using device: {device}")

    dataset = PotsdamDataset(csv_path, transform=get_val_augmentations(config.PATCH_SIZE))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # The checkpoint already contains fully fine-tuned weights for every
    # sub-module (including the CNN backbone), so there is no need to
    # additionally download ImageNet-pretrained initialization weights —
    # this also means evaluation/inference works in fully offline
    # environments.
    config.ENCODER_PRETRAINED = False
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print(f"[test] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(best_metric={checkpoint.get('best_metric', '?')})")

    metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES, class_names=config.CLASS_NAMES)

    qualitative_images, qualitative_gt, qualitative_pred, qualitative_overlay = [], [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=config.USE_AMP):
            logits = model(images)
        preds = logits.argmax(dim=1)

        metrics.update(preds, masks)

        if len(qualitative_images) < num_qualitative_samples:
            for i in range(images.size(0)):
                if len(qualitative_images) >= num_qualitative_samples:
                    break
                rgb_image = denormalize_image(images[i], config.IMAGE_MEAN, config.IMAGE_STD)
                gt_color = mask_to_color(masks[i].cpu().numpy(), config.CLASS_ID_TO_COLOR)
                pred_color = mask_to_color(preds[i].cpu().numpy(), config.CLASS_ID_TO_COLOR)
                overlay = overlay_mask_on_image(rgb_image, pred_color, alpha=0.5)

                qualitative_images.append(rgb_image)
                qualitative_gt.append(gt_color)
                qualitative_pred.append(pred_color)
                qualitative_overlay.append(overlay)

    results = metrics.compute()

    print("\n[test] ===== Evaluation Results =====")
    print(f"  Mean IoU:        {results['mean_iou']:.4f}")
    print(f"  Pixel Accuracy:  {results['pixel_accuracy']:.4f}")
    print(f"  Mean Precision:  {results['mean_precision']:.4f}")
    print(f"  Mean Recall:     {results['mean_recall']:.4f}")
    print(f"  Mean F1:         {results['mean_f1']:.4f}")
    print("  Per-class IoU:")
    for name in config.CLASS_NAMES:
        print(f"    {name:25s}: {results['per_class_iou'][name]:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    write_classification_report_csv(
        results, config.CLASS_NAMES, os.path.join(config.RESULTS_DIR, "test_classification_report.csv")
    )
    plot_confusion_matrix(
        metrics.get_confusion_matrix_normalized(), config.CLASS_NAMES,
        os.path.join(config.RESULTS_DIR, "test_confusion_matrix.png"),
    )
    if qualitative_images:
        save_prediction_grid(
            qualitative_images, qualitative_gt, qualitative_pred, qualitative_overlay,
            os.path.join(config.RESULTS_DIR, "test_qualitative_samples.png"),
            max_samples=num_qualitative_samples,
        )

    print(f"[test] Results saved under: {config.RESULTS_DIR}")
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DFF-MambaNet on a held-out split.")
    parser.add_argument("--checkpoint", type=str, default=Config.BEST_MODEL_PATH)
    parser.add_argument("--csv", type=str, default=Config.VAL_SPLIT_CSV,
                         help="Manifest CSV of the split to evaluate on (defaults to the val split).")
    parser.add_argument("--num_samples", type=int, default=8,
                         help="Number of qualitative prediction samples to save.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    run_evaluation(cfg, checkpoint_path=args.checkpoint, csv_path=args.csv,
                   num_qualitative_samples=args.num_samples)
