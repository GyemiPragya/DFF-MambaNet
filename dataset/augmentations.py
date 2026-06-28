"""
dataset/augmentations.py
=========================
Albumentations-based augmentation pipelines for the ISPRS Potsdam dataset.

Two pipelines are exposed:

* `get_train_augmentations(image_size)` — geometric + photometric
  augmentations suitable for training (random flips, rotations, scale jitter,
  brightness/contrast/HSV jitter, occasional blur/noise), followed by
  ImageNet normalization and conversion to tensors.

* `get_val_augmentations(image_size)` — deterministic resize + normalization
  only, so that validation/test numbers are comparable across epochs.

Both pipelines apply identically to the image and its segmentation mask
(Albumentations handles this natively via `additional_targets`/mask support).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import Config


def get_train_augmentations(image_size: int = None) -> A.Compose:
    """Build the training-time augmentation pipeline.

    Parameters
    ----------
    image_size : int, optional
        Target patch size (defaults to ``Config.PATCH_SIZE``).

    Returns
    -------
    albumentations.Compose
        Composed transform expecting `image` (HWC uint8) and `mask` (HW int)
        keys and returning normalized float tensors.
    """
    image_size = image_size or Config.PATCH_SIZE

    return A.Compose(
        [
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05), scale=(0.85, 1.15), rotate=(-15, 15),
                border_mode=0, p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                ],
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4), hole_height_range=(8, 16), hole_width_range=(8, 16),
                fill=0, p=0.1,
            ),
            A.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
            ToTensorV2(),
        ]
    )


def get_val_augmentations(image_size: int = None) -> A.Compose:
    """Build the deterministic validation/test-time transform pipeline."""
    image_size = image_size or Config.PATCH_SIZE

    return A.Compose(
        [
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
            ToTensorV2(),
        ]
    )


def get_inference_augmentations(image_size: int = None) -> A.Compose:
    """Transform used in inference.py — resize-only + normalize, no crop."""
    image_size = image_size or Config.PATCH_SIZE
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
            ToTensorV2(),
        ]
    )
