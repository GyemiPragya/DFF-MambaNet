"""
configs/config.py
==================
Central configuration module for DFF-MambaNet.

Every tunable hyper-parameter, path, and switch used across the project is
defined here as a single dataclass-style object so that `train.py`,
`test.py` and `inference.py` all read from one source of truth.

Usage
-----
    from configs.config import Config
    cfg = Config()
    print(cfg.DEVICE)

The class also exposes a couple of convenience helpers (GPU auto-detection,
Google Drive mounting for Colab, and directory creation) so that the same
config object can be dropped into Colab, Windows, or Ubuntu without edits.
"""

import os
import torch


class Config:
    """Single source of truth for all project hyper-parameters and paths."""

    # ------------------------------------------------------------------
    # Project identity
    # ------------------------------------------------------------------
    PROJECT_NAME = "DFF-MambaNet"
    EXPERIMENT_NAME = "aadff_resnet34_visionmamba_potsdam"
    SEED = 42

    # ------------------------------------------------------------------
    # Paths (edit ROOT_DIR if you relocate the repository)
    # ------------------------------------------------------------------
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Raw ISPRS Potsdam data (full-resolution TOP/Label tiles as downloaded)
    RAW_IMAGE_DIR = os.path.join(ROOT_DIR, "data", "raw", "images")
    RAW_LABEL_DIR = os.path.join(ROOT_DIR, "data", "raw", "labels")

    # Pre-processed 256x256 patches (created by dataset/preprocess.py)
    PATCH_IMAGE_DIR = os.path.join(ROOT_DIR, "data", "patches", "images")
    PATCH_LABEL_DIR = os.path.join(ROOT_DIR, "data", "patches", "labels")

    # CSV files listing the train / val split (created by preprocess.py)
    TRAIN_SPLIT_CSV = os.path.join(ROOT_DIR, "data", "patches", "train.csv")
    VAL_SPLIT_CSV = os.path.join(ROOT_DIR, "data", "patches", "val.csv")

    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")

    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")

    # ------------------------------------------------------------------
    # Dataset / classes
    # ------------------------------------------------------------------
    # ISPRS Potsdam ships labels as RGB-coded masks. We map every color to
    # an integer class id used internally by the network and losses.
    NUM_CLASSES = 6
    CLASS_NAMES = [
        "Impervious_Surface",
        "Building",
        "Low_Vegetation",
        "Tree",
        "Car",
        "Clutter_Background",
    ]
    # RGB color -> class index mapping, following the official ISPRS
    # Potsdam color convention.
    CLASS_COLORS = {
        (255, 255, 255): 0,  # Impervious surface (white)
        (0, 0, 255): 1,      # Building (blue)
        (0, 255, 255): 2,    # Low vegetation (cyan)
        (0, 255, 0): 3,      # Tree (green)
        (255, 255, 0): 4,    # Car (yellow)
        (255, 0, 0): 5,      # Clutter / background (red)
    }
    # Reverse mapping used purely for visualization (class id -> RGB)
    CLASS_ID_TO_COLOR = {v: k for k, v in CLASS_COLORS.items()}

    # ------------------------------------------------------------------
    # Image / patch geometry
    # ------------------------------------------------------------------
    PATCH_SIZE = 256
    PATCH_OVERLAP = 0          # overlap (in pixels) used while tiling
    IMAGE_MEAN = (0.485, 0.456, 0.406)   # ImageNet statistics (ResNet34)
    IMAGE_STD = (0.229, 0.224, 0.225)

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    VAL_SPLIT_RATIO = 0.2

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    ENCODER_NAME = "resnet34"
    ENCODER_PRETRAINED = True
    # Channel widths produced by each of the 4 pyramid stages. Both the CNN
    # branch (ResNet34 layer1..layer4) and the Vision Mamba branch are
    # designed to output features at these exact widths/strides so that the
    # AADFF module can fuse them stage-by-stage.
    STAGE_CHANNELS = (64, 128, 256, 512)
    STAGE_STRIDES = (4, 8, 16, 32)

    # Vision Mamba branch
    # NOTE on these defaults: they are deliberately modest so that the
    # project trains end-to-end even on the pure-PyTorch fallback SSM (i.e.
    # without the official `mamba_ssm` CUDA kernel) on CPU or a small GPU.
    # If `mamba_ssm` is installed and a CUDA GPU is available, the official
    # kernel is dramatically more memory-efficient per element, so feel free
    # to raise MAMBA_D_STATE / MAMBA_EXPAND / MAMBA_BLOCKS_PER_STAGE for a
    # stronger model once you've confirmed the official kernel is in use
    # (check the printed "[VisionMambaEncoder]" warning at startup).
    MAMBA_D_STATE = 8               # SSM state dimension (N)
    MAMBA_EXPAND = 1                # inner expansion factor for the SSM block
    MAMBA_BLOCKS_PER_STAGE = (1, 1, 1, 1)   # depth per pyramid stage
    MAMBA_CONV_KERNEL = 3
    MAMBA_USE_OFFICIAL_KERNEL = True  # try `mamba_ssm` first, else fallback
    # Token-count cap used ONLY by the pure-PyTorch fallback SSM: stages
    # whose flattened token count (H*W) exceeds this are processed on an
    # average-pooled grid and bilinearly upsampled back, to keep memory
    # bounded. Has no effect when the official `mamba_ssm` kernel is active.
    MAMBA_MAX_TOKENS_FALLBACK = 256

    # AADFF fusion module
    FUSION_REDUCTION = 8            # channel-attention squeeze ratio

    # Decoder
    DECODER_CHANNELS = (256, 128, 64, 32)

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = "adamw"              # one of: "adamw", "sgd"
    SGD_MOMENTUM = 0.9
    LR_SCHEDULER = "cosine"          # one of: "cosine", "step", "plateau", "none"
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1
    WARMUP_EPOCHS = 5

    LOSS_TYPE = "hybrid"             # one of: "ce", "dice", "hybrid"
    CE_WEIGHT = 0.5                  # weight of CE inside the hybrid loss
    DICE_WEIGHT = 0.5                # weight of Dice inside the hybrid loss
    CLASS_WEIGHTS = None              # optional list[float] of length NUM_CLASSES

    USE_AMP = True                    # mixed precision training
    GRAD_CLIP_NORM = 5.0
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 1e-4

    RESUME_TRAINING = False
    RESUME_CHECKPOINT = LAST_MODEL_PATH

    LOG_EVERY_N_STEPS = 20

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------
    @staticmethod
    def get_device() -> torch.device:
        """Automatically select CUDA if available, else CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    DEVICE = get_device.__func__()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def create_directories(cls):
        """Create every directory referenced in the config if missing."""
        dirs = [
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
            os.path.dirname(cls.PATCH_IMAGE_DIR),
            cls.PATCH_IMAGE_DIR,
            cls.PATCH_LABEL_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def mount_google_drive(mount_point: str = "/content/drive"):
        """
        Mount Google Drive when running inside Google Colab.

        Safe to call from any environment: it silently no-ops if the
        `google.colab` package is not importable (i.e. not running on Colab).
        """
        try:
            from google.colab import drive  # type: ignore
            drive.mount(mount_point)
            print(f"[Config] Google Drive mounted at {mount_point}")
        except ImportError:
            print("[Config] Not running on Google Colab — skipping Drive mount.")

    def __repr__(self):
        attrs = {k: v for k, v in vars(Config).items() if not k.startswith("_") and k.isupper()}
        lines = [f"{k} = {v}" for k, v in attrs.items()]
        return "Config(\n  " + "\n  ".join(lines) + "\n)"


if __name__ == "__main__":
    cfg = Config()
    cfg.create_directories()
    print(f"[Config] Using device: {cfg.DEVICE}")
    print(cfg)
