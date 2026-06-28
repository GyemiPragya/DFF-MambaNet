"""
models/metrics.py
==================
Segmentation evaluation metrics, all derived from a running confusion
matrix accumulated batch-by-batch across an epoch:

    * Per-class IoU and mean IoU (mIoU)
    * Pixel accuracy (overall accuracy)
    * Per-class precision, recall, F1
    * Macro-averaged precision / recall / F1
    * The raw confusion matrix itself (for plotting)

Usage
-----
    metrics = SegmentationMetrics(num_classes=6, class_names=[...])
    for batch in loader:
        preds = model(batch["image"]).argmax(dim=1)
        metrics.update(preds, batch["mask"])
    results = metrics.compute()
    metrics.reset()
"""

from typing import List, Optional, Dict

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates a confusion matrix over multiple batches and derives all
    standard semantic-segmentation metrics from it.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Zero out the accumulated confusion matrix (call at epoch start)."""
        self.confusion_matrix.fill(0)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Accumulate one batch's predictions into the running confusion matrix.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted class indices, shape (B, H, W) — i.e. already
            ``argmax``-ed over the channel dimension.
        targets : torch.Tensor
            Ground-truth class indices, shape (B, H, W).
        """
        preds_np = preds.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)

        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        indices = self.num_classes * targets_np.astype(np.int64) + preds_np.astype(np.int64)
        batch_cm = np.bincount(indices, minlength=self.num_classes ** 2)
        batch_cm = batch_cm.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += batch_cm

    def compute(self) -> Dict:
        """Derive all metrics from the current confusion matrix.

        Returns
        -------
        dict with keys:
            mean_iou, pixel_accuracy, overall_accuracy,
            mean_precision, mean_recall, mean_f1,
            per_class_iou, per_class_precision, per_class_recall, per_class_f1,
            confusion_matrix
        """
        cm = self.confusion_matrix.astype(np.float64)
        eps = 1e-10

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp   # predicted as class c, but isn't
        fn = cm.sum(axis=1) - tp   # actually class c, but predicted otherwise

        per_class_iou = tp / (tp + fp + fn + eps)
        per_class_precision = tp / (tp + fp + eps)
        per_class_recall = tp / (tp + fn + eps)
        per_class_f1 = (
            2 * per_class_precision * per_class_recall
            / (per_class_precision + per_class_recall + eps)
        )

        pixel_accuracy = tp.sum() / (cm.sum() + eps)

        return {
            "mean_iou": float(np.mean(per_class_iou)),
            "pixel_accuracy": float(pixel_accuracy),
            "overall_accuracy": float(pixel_accuracy),  # identical for single-label segmentation
            "mean_precision": float(np.mean(per_class_precision)),
            "mean_recall": float(np.mean(per_class_recall)),
            "mean_f1": float(np.mean(per_class_f1)),
            "per_class_iou": {self.class_names[i]: float(per_class_iou[i]) for i in range(self.num_classes)},
            "per_class_precision": {self.class_names[i]: float(per_class_precision[i]) for i in range(self.num_classes)},
            "per_class_recall": {self.class_names[i]: float(per_class_recall[i]) for i in range(self.num_classes)},
            "per_class_f1": {self.class_names[i]: float(per_class_f1[i]) for i in range(self.num_classes)},
            "confusion_matrix": cm.astype(np.int64).tolist(),
        }

    def get_confusion_matrix_normalized(self) -> np.ndarray:
        """Row-normalized confusion matrix (each row sums to 1), useful for
        plotting since it shows per-class recall directly on the diagonal.
        """
        cm = self.confusion_matrix.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cm / row_sums


if __name__ == "__main__":
    torch.manual_seed(0)
    metrics = SegmentationMetrics(num_classes=6)
    preds = torch.randint(0, 6, (2, 32, 32))
    targets = torch.randint(0, 6, (2, 32, 32))
    metrics.update(preds, targets)
    results = metrics.compute()
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f}")
