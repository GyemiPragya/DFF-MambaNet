"""
train.py
========
Main training entry point for DFF-MambaNet on the ISPRS Potsdam dataset.

Implements:
    * AdamW / SGD optimizer selection with a differential learning rate for
      the pretrained CNN backbone vs. the rest of the network
    * Cosine / step / plateau learning-rate scheduling with linear warmup
    * Mixed-precision (AMP) training
    * Gradient clipping
    * Early stopping on validation mIoU
    * Best + last checkpoint saving, and resume-from-checkpoint support
    * TensorBoard logging
    * Per-epoch metrics exported to CSV
    * Training-curve and confusion-matrix plots saved at the end of training

Usage
-----
    python train.py
    python train.py --epochs 50 --batch_size 16 --lr 1e-4
    python train.py --resume checkpoints/last_model.pth
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import Config
from dataset.dataset import build_dataloaders
from models.network import build_model
from models.losses import build_loss_fn
from models.metrics import SegmentationMetrics
from models.utils import (
    seed_everything,
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    plot_training_curves,
    plot_confusion_matrix,
)


def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Build the configured optimizer with a differential learning rate for
    the pretrained CNN backbone (lower LR) vs. the rest of the network.
    """
    param_groups = model.get_trainable_parameter_groups(
        base_lr=config.LEARNING_RATE, backbone_lr_mult=0.1
    )

    if config.OPTIMIZER.lower() == "adamw":
        return torch.optim.AdamW(param_groups, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(
            param_groups, lr=config.LEARNING_RATE, momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer '{config.OPTIMIZER}'. Expected 'adamw' or 'sgd'.")


def build_scheduler(optimizer: torch.optim.Optimizer, config, steps_per_epoch: int):
    """Build the configured LR scheduler, with optional linear warmup."""
    scheduler_type = config.LR_SCHEDULER.lower()

    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.EPOCHS - config.WARMUP_EPOCHS, 1)
        )
    elif scheduler_type == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )
    elif scheduler_type == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.LR_GAMMA, patience=5
        )
    elif scheduler_type == "none":
        main_scheduler = None
    else:
        raise ValueError(f"Unknown LR_SCHEDULER '{config.LR_SCHEDULER}'.")

    if config.WARMUP_EPOCHS > 0 and main_scheduler is not None and scheduler_type != "plateau":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=config.WARMUP_EPOCHS
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.WARMUP_EPOCHS],
        )
    return main_scheduler


def get_amp_device_type(device: torch.device) -> str:
    """`torch.amp.autocast`/`GradScaler` need an explicit device-type string
    ("cuda" or "cpu") rather than inferring it, under the current API.
    """
    return "cuda" if device.type == "cuda" else "cpu"


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, config, epoch, writer, amp_device_type):
    """Run a single training epoch. Returns the average training loss."""
    model.train()
    loss_meter = AverageMeter()

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for step, batch in enumerate(progress_bar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device_type, enabled=config.USE_AMP):
            logits = model(images)
            loss = loss_fn(logits, masks)

        if config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))
        progress_bar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        global_step = epoch * len(loader) + step
        if global_step % config.LOG_EVERY_N_STEPS == 0:
            writer.add_scalar("train/step_loss", loss.item(), global_step)

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, loss_fn, device, config, epoch, writer, metrics: SegmentationMetrics, amp_device_type):
    """Run a full validation pass. Returns (avg_val_loss, metrics_dict)."""
    model.eval()
    loss_meter = AverageMeter()
    metrics.reset()

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type=amp_device_type, enabled=config.USE_AMP):
            logits = model(images)
            loss = loss_fn(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        loss_meter.update(loss.item(), n=images.size(0))
        progress_bar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    results = metrics.compute()
    writer.add_scalar("val/loss", loss_meter.avg, epoch)
    writer.add_scalar("val/mean_iou", results["mean_iou"], epoch)
    writer.add_scalar("val/pixel_accuracy", results["pixel_accuracy"], epoch)
    writer.add_scalar("val/mean_f1", results["mean_f1"], epoch)

    return loss_meter.avg, results


def append_epoch_csv(csv_path: str, row: dict):
    """Append one epoch's metrics as a row to a CSV file, writing the
    header on first use.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(config: Config):
    seed_everything(config.SEED)
    config.create_directories()

    device = config.DEVICE
    amp_device_type = get_amp_device_type(device)
    print(f"[train] Using device: {device}")

    train_loader, val_loader = build_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        image_size=config.PATCH_SIZE,
    )
    print(f"[train] Train batches/epoch: {len(train_loader)} | Val batches/epoch: {len(val_loader)}")

    model = build_model(config).to(device)
    print(f"[train] Trainable parameters: {count_parameters(model):,}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    loss_fn = build_loss_fn(config).to(device)
    scaler = GradScaler(device=amp_device_type, enabled=config.USE_AMP)
    metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES, class_names=config.CLASS_NAMES)
    early_stopper = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max",
    )

    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME))

    start_epoch = 0
    best_metric = 0.0
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pixel_accuracy": [], "val_mean_f1": []}

    if config.RESUME_TRAINING and os.path.exists(config.RESUME_CHECKPOINT):
        print(f"[train] Resuming from checkpoint: {config.RESUME_CHECKPOINT}")
        checkpoint = load_checkpoint(
            config.RESUME_CHECKPOINT, model, optimizer, scheduler, scaler, device=device
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metric = checkpoint.get("best_metric", 0.0)

    metrics_csv_path = os.path.join(config.RESULTS_DIR, "epoch_metrics.csv")

    for epoch in range(start_epoch, config.EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, config, epoch, writer, amp_device_type)
        val_loss, val_results = validate(model, val_loader, loss_fn, device, config, epoch, writer, metrics, amp_device_type)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results["mean_iou"])
            else:
                scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mIoU={val_results['mean_iou']:.4f} pixel_acc={val_results['pixel_accuracy']:.4f} "
            f"F1={val_results['mean_f1']:.4f} lr={current_lr:.2e} time={elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_results["mean_iou"])
        history["val_pixel_accuracy"].append(val_results["pixel_accuracy"])
        history["val_mean_f1"].append(val_results["mean_f1"])

        append_epoch_csv(metrics_csv_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_iou": val_results["mean_iou"],
            "val_pixel_accuracy": val_results["pixel_accuracy"],
            "val_mean_precision": val_results["mean_precision"],
            "val_mean_recall": val_results["mean_recall"],
            "val_mean_f1": val_results["mean_f1"],
            "learning_rate": current_lr,
            "epoch_time_sec": elapsed,
        })

        # Always keep a "last" checkpoint so training can be resumed.
        save_checkpoint(
            config.LAST_MODEL_PATH, model, optimizer, scheduler, epoch=epoch,
            best_metric=best_metric, scaler=scaler,
        )

        # Save a new "best" checkpoint whenever validation mIoU improves.
        if val_results["mean_iou"] > best_metric:
            best_metric = val_results["mean_iou"]
            save_checkpoint(
                config.BEST_MODEL_PATH, model, optimizer, scheduler, epoch=epoch,
                best_metric=best_metric, scaler=scaler,
            )
            print(f"[train] New best model saved (mIoU={best_metric:.4f}) -> {config.BEST_MODEL_PATH}")

        if early_stopper.step(val_results["mean_iou"]):
            print(f"[train] Early stopping triggered at epoch {epoch} (best mIoU={early_stopper.best_score:.4f}).")
            break

    writer.close()

    # Final visualizations saved into the results directory.
    plot_training_curves(history, os.path.join(config.RESULTS_DIR, "training_curves.png"))
    plot_confusion_matrix(
        metrics.get_confusion_matrix_normalized(),
        config.CLASS_NAMES,
        os.path.join(config.RESULTS_DIR, "confusion_matrix.png"),
    )
    print(f"[train] Training complete. Best validation mIoU: {best_metric:.4f}")
    print(f"[train] Results saved under: {config.RESULTS_DIR}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train DFF-MambaNet on ISPRS Potsdam.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss_type", type=str, default=None, choices=["ce", "dice", "hybrid"])
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Disable ImageNet-pretrained CNN backbone weights (e.g. for offline environments).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()

    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.LEARNING_RATE = args.lr
    if args.loss_type is not None:
        cfg.LOSS_TYPE = args.loss_type
    if args.optimizer is not None:
        cfg.OPTIMIZER = args.optimizer
    if args.resume is not None:
        cfg.RESUME_TRAINING = True
        cfg.RESUME_CHECKPOINT = args.resume
    if args.no_amp:
        cfg.USE_AMP = False
    if args.no_pretrained:
        cfg.ENCODER_PRETRAINED = False

    main(cfg)
