"""
dataset/preprocess.py
======================
Pre-processing utilities for the ISPRS Potsdam Semantic Segmentation dataset.

Responsibilities
-----------------
1. Tile full-resolution "TOP" RGB images and their RGB-coded ground-truth
   label images into fixed-size (default 256x256) patches.
2. Convert RGB-coded labels into single-channel class-index masks using the
   official ISPRS Potsdam color table (see ``configs.config.Config.CLASS_COLORS``).
3. Persist patches to disk under ``data/patches/{images,labels}``.
4. Build a stratified train/validation split and write it out as two CSV
   files (`train.csv`, `val.csv`) consumed by ``dataset/dataset.py``.

This module can be run as a standalone script:

    python -m dataset.preprocess --raw_image_dir data/raw/images \
                                  --raw_label_dir data/raw/labels \
                                  --patch_size 256

or imported and called programmatically.
"""

import argparse
import os
import glob

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from configs.config import Config


def rgb_label_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) RGB-coded label image into an (H, W) class-index map.

    Any pixel color not present in ``Config.CLASS_COLORS`` is mapped to the
    "Clutter / Background" class (index 5) as a safe fallback.
    """
    h, w, _ = label_rgb.shape
    class_map = np.full((h, w), fill_value=Config.CLASS_COLORS[(255, 0, 0)], dtype=np.uint8)

    for color, class_id in Config.CLASS_COLORS.items():
        matches = np.all(label_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_map[matches] = class_id

    return class_map


def class_index_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_label_to_class_index`, used for visualization."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in Config.CLASS_ID_TO_COLOR.items():
        rgb[class_map == class_id] = color
    return rgb


def _tile_pair(image: np.ndarray, label: np.ndarray, patch_size: int, overlap: int):
    """Yield (image_patch, label_patch, row, col) tuples covering the full image.

    The final row/column of patches is shifted inward (rather than padded)
    so that every patch is exactly `patch_size x patch_size`, at the cost of
    a small amount of overlap at the image borders.
    """
    h, w = image.shape[:2]
    stride = patch_size - overlap

    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not ys or ys[-1] != h - patch_size:
        ys.append(max(h - patch_size, 0))
    if not xs or xs[-1] != w - patch_size:
        xs.append(max(w - patch_size, 0))

    for y in ys:
        for x in xs:
            img_patch = image[y:y + patch_size, x:x + patch_size]
            lbl_patch = label[y:y + patch_size, x:x + patch_size]
            if img_patch.shape[0] != patch_size or img_patch.shape[1] != patch_size:
                continue  # skip degenerate edge patches on tiny source images
            yield img_patch, lbl_patch, y, x


def tile_dataset(
    raw_image_dir: str,
    raw_label_dir: str,
    out_image_dir: str,
    out_label_dir: str,
    patch_size: int = 256,
    overlap: int = 0,
    image_ext: str = ".tif",
    label_suffix: str = "_label",
):
    """Tile every full-resolution image/label pair into patches saved to disk.

    Parameters
    ----------
    raw_image_dir, raw_label_dir : str
        Directories containing the original ISPRS Potsdam TOP images and
        their corresponding RGB ground-truth label images.
    out_image_dir, out_label_dir : str
        Destination directories for the generated patches.
    patch_size : int
        Side length (in pixels) of the square patches to extract.
    overlap : int
        Overlap, in pixels, between adjacent patches.
    image_ext : str
        File extension of the raw images (Potsdam ships ``.tif``).
    label_suffix : str
        Suffix used to match an image file to its label file
        (e.g. ``top_potsdam_2_10.tif`` <-> ``top_potsdam_2_10_label.tif``).
    """
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(raw_image_dir, f"*{image_ext}")))
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images with extension '{image_ext}' found in {raw_image_dir}. "
            "Verify the ISPRS Potsdam dataset has been extracted there."
        )

    manifest_rows = []
    patch_counter = 0

    for image_path in tqdm(image_paths, desc="Tiling Potsdam tiles"):
        base_name = os.path.splitext(os.path.basename(image_path))[0].replace("_IRRG", "")
        label_path = os.path.join(raw_label_dir, f"{base_name}{label_suffix}{image_ext}")
        if not os.path.exists(label_path):
            # Fall back to a label file sharing the exact same stem.
            label_path = os.path.join(raw_label_dir, f"{base_name}{image_ext}")
        if not os.path.exists(label_path):
            print(f"[preprocess] WARNING: no label found for {image_path}, skipping.")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_rgb = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)

        if image.shape[:2] != label_rgb.shape[:2]:
            raise ValueError(f"Shape mismatch between {image_path} and {label_path}")

        label_idx = rgb_label_to_class_index(label_rgb)

        for img_patch, lbl_patch, y, x in _tile_pair(image, label_idx, patch_size, overlap):
            patch_name = f"{base_name}_y{y}_x{x}.png"
            img_out_path = os.path.join(out_image_dir, patch_name)
            lbl_out_path = os.path.join(out_label_dir, patch_name)

            cv2.imwrite(img_out_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(lbl_out_path, lbl_patch)  # single channel, values 0..5

            manifest_rows.append(
                {
                    "patch_name": patch_name,
                    "source_tile": base_name,
                    "image_path": img_out_path,
                    "label_path": lbl_out_path,
                }
            )
            patch_counter += 1

    print(f"[preprocess] Generated {patch_counter} patches from {len(image_paths)} source tiles.")
    return pd.DataFrame(manifest_rows)


def build_train_val_split(manifest_df: "pd.DataFrame", val_ratio: float, seed: int):
    """Split the patch manifest into train/val sets, grouped by source tile.

    Splitting at the *source tile* level (rather than per-patch) avoids
    leaking near-duplicate, spatially-adjacent patches between train and
    validation, which would otherwise inflate validation metrics.
    """
    source_tiles = manifest_df["source_tile"].unique()
    train_tiles, val_tiles = train_test_split(
        source_tiles, test_size=val_ratio, random_state=seed
    )
    train_df = manifest_df[manifest_df["source_tile"].isin(train_tiles)].reset_index(drop=True)
    val_df = manifest_df[manifest_df["source_tile"].isin(val_tiles)].reset_index(drop=True)
    return train_df, val_df


def run_preprocessing(
    raw_image_dir: str = Config.RAW_IMAGE_DIR,
    raw_label_dir: str = Config.RAW_LABEL_DIR,
    patch_size: int = Config.PATCH_SIZE,
    overlap: int = Config.PATCH_OVERLAP,
    val_ratio: float = Config.VAL_SPLIT_RATIO,
    seed: int = Config.SEED,
):
    """End-to-end pre-processing: tile the dataset and write the split CSVs."""
    manifest_df = tile_dataset(
        raw_image_dir=raw_image_dir,
        raw_label_dir=raw_label_dir,
        out_image_dir=Config.PATCH_IMAGE_DIR,
        out_label_dir=Config.PATCH_LABEL_DIR,
        patch_size=patch_size,
        overlap=overlap,
    )
    train_df, val_df = build_train_val_split(manifest_df, val_ratio=val_ratio, seed=seed)

    os.makedirs(os.path.dirname(Config.TRAIN_SPLIT_CSV), exist_ok=True)
    train_df.to_csv(Config.TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(Config.VAL_SPLIT_CSV, index=False)

    print(f"[preprocess] Train patches: {len(train_df)} | Val patches: {len(val_df)}")
    print(f"[preprocess] Train CSV -> {Config.TRAIN_SPLIT_CSV}")
    print(f"[preprocess] Val CSV   -> {Config.VAL_SPLIT_CSV}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Tile ISPRS Potsdam dataset into patches.")
    parser.add_argument("--raw_image_dir", type=str, default=Config.RAW_IMAGE_DIR)
    parser.add_argument("--raw_label_dir", type=str, default=Config.RAW_LABEL_DIR)
    parser.add_argument("--patch_size", type=int, default=Config.PATCH_SIZE)
    parser.add_argument("--overlap", type=int, default=Config.PATCH_OVERLAP)
    parser.add_argument("--val_ratio", type=float, default=Config.VAL_SPLIT_RATIO)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_preprocessing(
        raw_image_dir=args.raw_image_dir,
        raw_label_dir=args.raw_label_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
