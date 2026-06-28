"""
inference.py
=============
Single-image inference entry point for DFF-MambaNet.

Given one RGB satellite image (of any size), produces:

    * The predicted class-index segmentation mask (saved as a single-
      channel PNG, values 0..NUM_CLASSES-1)
    * A colorized segmentation map (RGB, using the ISPRS Potsdam palette)
    * An overlay of the colorized prediction on top of the original image

For images larger than the model's training patch size, a sliding-window
("tiled") inference strategy with overlap-averaging is used so that
arbitrarily large satellite tiles can be processed without manual
pre-tiling.

Usage
-----
    python inference.py --image path/to/satellite_image.png --checkpoint checkpoints/best_model.pth
    python inference.py --image path/to/large_tile.tif --tile_size 256 --overlap 32
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from configs.config import Config
from models.network import build_model
from models.utils import load_checkpoint, mask_to_color, overlay_mask_on_image


def load_image_rgb(path: str) -> np.ndarray:
    """Read an image from disk and return it as an (H, W, 3) uint8 RGB array."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_tile(tile_rgb: np.ndarray, mean, std, device) -> torch.Tensor:
    """Normalize an (H, W, 3) uint8 RGB tile and convert it to a (1, 3, H, W)
    float tensor ready for the model.
    """
    tile = tile_rgb.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    tile = (tile - mean) / std
    tile = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float()
    return tile.to(device)


@torch.no_grad()
def sliding_window_inference(
    model,
    image_rgb: np.ndarray,
    num_classes: int,
    device,
    tile_size: int = 256,
    overlap: int = 32,
    mean=Config.IMAGE_MEAN,
    std=Config.IMAGE_STD,
    use_amp: bool = True,
) -> np.ndarray:
    """Run the model over an arbitrarily large image using overlapping tiles,
    averaging per-pixel class probabilities across overlapping predictions.

    Parameters
    ----------
    image_rgb : np.ndarray
        (H, W, 3) uint8 RGB input image, any size.
    tile_size : int
        Side length of each square tile fed to the model (should match the
        resolution the model was trained at, e.g. 256).
    overlap : int
        Overlap, in pixels, between adjacent tiles. Larger overlap reduces
        seam artefacts at tile boundaries at the cost of more compute.

    Returns
    -------
    np.ndarray
        (H, W) int64 array of predicted class indices for the full image.
    """
    h, w, _ = image_rgb.shape
    stride = tile_size - overlap

    # Pad the image so it's tileable, then crop the result back down.
    pad_h = (-(h - tile_size)) % stride if h > tile_size else max(tile_size - h, 0)
    pad_w = (-(w - tile_size)) % stride if w > tile_size else max(tile_size - w, 0)
    padded = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw, _ = padded.shape

    prob_accum = np.zeros((num_classes, ph, pw), dtype=np.float32)
    weight_accum = np.zeros((ph, pw), dtype=np.float32)

    # A simple raised-cosine-like weight (here: uniform with feathered
    # edges) reduces visible seams where tiles overlap.
    tile_weight = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        tile_weight[:overlap, :] *= ramp[:, None]
        tile_weight[-overlap:, :] *= ramp[::-1][:, None]
        tile_weight[:, :overlap] *= ramp[None, :]
        tile_weight[:, -overlap:] *= ramp[::-1][None, :]

    ys = list(range(0, ph - tile_size + 1, stride)) or [0]
    xs = list(range(0, pw - tile_size + 1, stride)) or [0]
    if ys[-1] != ph - tile_size:
        ys.append(ph - tile_size)
    if xs[-1] != pw - tile_size:
        xs.append(pw - tile_size)

    model.eval()
    for y in ys:
        for x in xs:
            tile = padded[y:y + tile_size, x:x + tile_size]
            tile_tensor = preprocess_tile(tile, mean, std, device)
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                logits = model(tile_tensor)
            probs = F.softmax(logits, dim=1)[0].float().cpu().numpy()  # (C, tile, tile)

            prob_accum[:, y:y + tile_size, x:x + tile_size] += probs * tile_weight[None, :, :]
            weight_accum[y:y + tile_size, x:x + tile_size] += tile_weight

    weight_accum = np.clip(weight_accum, a_min=1e-6, a_max=None)
    prob_accum /= weight_accum[None, :, :]

    pred_full = np.argmax(prob_accum, axis=0)
    return pred_full[:h, :w]  # crop back to the original (unpadded) size


def run_inference(
    image_path: str,
    checkpoint_path: str,
    output_dir: str,
    config: Config,
    tile_size: int = 256,
    overlap: int = 32,
):
    """End-to-end single-image inference: load model, run prediction, and
    save the mask / colorized mask / overlay to `output_dir`.
    """
    device = config.DEVICE
    os.makedirs(output_dir, exist_ok=True)

    print(f"[inference] Loading model from {checkpoint_path} ...")
    # No need for ImageNet-pretrained init weights — the checkpoint already
    # contains fully fine-tuned weights, and this keeps inference usable
    # fully offline.
    config.ENCODER_PRETRAINED = False
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    print(f"[inference] Reading image: {image_path}")
    image_rgb = load_image_rgb(image_path)
    print(f"[inference] Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]} (WxH)")

    pred_mask = sliding_window_inference(
        model, image_rgb, num_classes=config.NUM_CLASSES, device=device,
        tile_size=tile_size, overlap=overlap,
        mean=config.IMAGE_MEAN, std=config.IMAGE_STD, use_amp=config.USE_AMP,
    )

    color_mask = mask_to_color(pred_mask, config.CLASS_ID_TO_COLOR)
    overlay = overlay_mask_on_image(image_rgb, color_mask, alpha=0.5)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    color_path = os.path.join(output_dir, f"{base_name}_mask_color.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")

    cv2.imwrite(mask_path, pred_mask.astype(np.uint8))
    cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[inference] Saved class-index mask -> {mask_path}")
    print(f"[inference] Saved colorized mask   -> {color_path}")
    print(f"[inference] Saved overlay image    -> {overlay_path}")

    # Quick per-class pixel coverage summary, handy for a sanity check.
    print("[inference] Predicted class coverage:")
    total_pixels = pred_mask.size
    for class_id, class_name in enumerate(config.CLASS_NAMES):
        coverage = float(np.mean(pred_mask == class_id)) * 100
        print(f"    {class_name:25s}: {coverage:5.2f}%  ({int(total_pixels * coverage / 100)} px)")

    return mask_path, color_path, overlay_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Run DFF-MambaNet inference on a single satellite image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input RGB image.")
    parser.add_argument("--checkpoint", type=str, default=Config.BEST_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=os.path.join(Config.RESULTS_DIR, "inference"))
    parser.add_argument("--tile_size", type=int, default=Config.PATCH_SIZE)
    parser.add_argument("--overlap", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config=cfg,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
