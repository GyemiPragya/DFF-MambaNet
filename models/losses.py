"""
models/losses.py
=================
Loss functions for semantic segmentation: Cross Entropy, Dice, and a
configurable weighted Hybrid combination of the two.

Selection is driven by ``configs.config.Config.LOSS_TYPE`` (one of
``"ce"``, ``"dice"``, ``"hybrid"``), via the :func:`build_loss_fn` factory
so ``train.py`` only needs a single call site.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss.

    Dice directly optimizes the overlap between predicted and ground-truth
    regions, which makes it considerably more robust than plain Cross
    Entropy to the heavy class imbalance typical of remote-sensing
    segmentation (e.g. "Car" pixels are vastly outnumbered by "Impervious
    Surface" pixels in ISPRS Potsdam).

    Parameters
    ----------
    num_classes : int
    smooth : float
        Additive smoothing term to avoid division by zero for classes
        absent from a given batch.
    ignore_index : int, optional
        Label value to exclude from the loss (e.g. unlabeled pixels).
    """

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Shape (B, num_classes, H, W), raw (pre-softmax) network output.
        targets : torch.Tensor
            Shape (B, H, W), integer class labels.
        """
        probs = F.softmax(logits, dim=1)

        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # placeholder, masked out below

        targets_onehot = F.one_hot(targets_clamped, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        dims = (0, 2, 3)  # reduce over batch + spatial dims, keep per-class
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CrossEntropyLoss(nn.Module):
    """Thin wrapper around ``nn.CrossEntropyLoss`` supporting optional
    per-class weights (useful for class-imbalanced datasets like Potsdam)
    and an ``ignore_index``, kept consistent with :class:`DiceLoss`'s API.
    """

    def __init__(self, class_weights: Optional[Sequence[float]] = None,
                 ignore_index: int = -100):
        super().__init__()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)

    def to(self, *args, **kwargs):
        # Ensure the internal class-weight tensor moves with the module.
        module = super().to(*args, **kwargs)
        return module


class HybridLoss(nn.Module):
    """Weighted combination of Cross Entropy and Dice losses.

    ``total = ce_weight * CE(logits, targets) + dice_weight * Dice(logits, targets)``

    Combining the two is a common, effective recipe for segmentation: CE
    provides stable per-pixel gradient signal early in training, while Dice
    directly targets region-level overlap and helps with class imbalance.
    """

    def __init__(self, num_classes: int, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 class_weights: Optional[Sequence[float]] = None,
                 ignore_index: Optional[int] = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        ce_ignore_index = ignore_index if ignore_index is not None else -100
        self.ce_loss = CrossEntropyLoss(class_weights=class_weights, ignore_index=ce_ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def build_loss_fn(config) -> nn.Module:
    """Factory selecting the configured loss function.

    Reads ``config.LOSS_TYPE`` (``"ce"``, ``"dice"``, or ``"hybrid"``) plus
    the associated weighting/class-weight settings from the config object.
    """
    loss_type = config.LOSS_TYPE.lower()

    if loss_type == "ce":
        return CrossEntropyLoss(class_weights=config.CLASS_WEIGHTS)
    elif loss_type == "dice":
        return DiceLoss(num_classes=config.NUM_CLASSES)
    elif loss_type == "hybrid":
        return HybridLoss(
            num_classes=config.NUM_CLASSES,
            ce_weight=config.CE_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            class_weights=config.CLASS_WEIGHTS,
        )
    else:
        raise ValueError(f"Unknown LOSS_TYPE '{config.LOSS_TYPE}'. Expected 'ce', 'dice', or 'hybrid'.")


if __name__ == "__main__":
    logits = torch.randn(2, 6, 64, 64)
    targets = torch.randint(0, 6, (2, 64, 64))

    for name, loss_fn in [
        ("CrossEntropyLoss", CrossEntropyLoss()),
        ("DiceLoss", DiceLoss(num_classes=6)),
        ("HybridLoss", HybridLoss(num_classes=6)),
    ]:
        value = loss_fn(logits, targets)
        print(f"{name}: {value.item():.4f}")
