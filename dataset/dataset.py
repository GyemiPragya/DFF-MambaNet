"""
dataset/dataset.py
===================
PyTorch ``Dataset`` implementation for the pre-processed (tiled) ISPRS
Potsdam Semantic Segmentation patches.

Expects patches to already exist on disk (see ``dataset/preprocess.py``):

    data/patches/images/<patch_name>.png   -> RGB uint8, 256x256x3
    data/patches/labels/<patch_name>.png   -> single-channel uint8, values 0..5

and a manifest CSV (``train.csv`` / ``val.csv``) with at least the columns
``image_path`` and ``label_path``.
"""

import os
from typing import Optional, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import Config
from dataset.augmentations import get_train_augmentations, get_val_augmentations


class PotsdamDataset(Dataset):
    """ISPRS Potsdam patch-level semantic segmentation dataset.

    Parameters
    ----------
    csv_path : str
        Path to a manifest CSV produced by ``dataset/preprocess.py`` with
        ``image_path`` and ``label_path`` columns.
    transform : albumentations.Compose, optional
        Joint image/mask transform. If ``None``, a deterministic
        normalize-only transform is used.
    """

    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Manifest CSV not found at {csv_path}. "
                "Run `python -m dataset.preprocess` first to generate patches."
            )
        self.manifest = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else get_val_augmentations()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]

        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read label mask: {row['label_path']}")

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"].float()
        mask_tensor = augmented["mask"].long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patch_name": row["patch_name"] if "patch_name" in row else os.path.basename(row["image_path"]),
        }

    def get_class_distribution(self) -> np.ndarray:
        """Compute the per-class pixel frequency across the whole split.

        Useful for deriving class weights to counter the heavy class
        imbalance typical of remote-sensing segmentation (e.g. "Car" pixels
        are far rarer than "Impervious Surface").
        """
        counts = np.zeros(Config.NUM_CLASSES, dtype=np.int64)
        for _, row in self.manifest.iterrows():
            mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
            for c in range(Config.NUM_CLASSES):
                counts[c] += int(np.sum(mask == c))
        return counts


def build_dataloaders(
    train_csv: str = Config.TRAIN_SPLIT_CSV,
    val_csv: str = Config.VAL_SPLIT_CSV,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    image_size: int = Config.PATCH_SIZE,
):
    """Convenience factory that returns (train_loader, val_loader)."""
    train_dataset = PotsdamDataset(train_csv, transform=get_train_augmentations(image_size))
    val_dataset = PotsdamDataset(val_csv, transform=get_val_augmentations(image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke test when run directly: `python -m dataset.dataset`
    train_loader, val_loader = build_dataloaders(batch_size=2, num_workers=0)
    batch = next(iter(train_loader))
    print(f"[dataset] image batch shape: {batch['image'].shape}")
    print(f"[dataset] mask  batch shape: {batch['mask'].shape}")
    print(f"[dataset] unique mask values: {torch.unique(batch['mask'])}")
