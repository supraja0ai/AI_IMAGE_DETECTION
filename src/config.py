"""
====================================================
AI IMAGE DETECTION V2
Configuration File
====================================================
"""

import os
import torch
from pathlib import Path

# ====================================================
# PROJECT PATHS
# ====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ====================================================
# MODEL PATHS
# ====================================================

EFFNET_MODEL_PATH = MODEL_DIR / "effnet_b0_best.pth"

RESNET_MODEL_PATH = MODEL_DIR / "resnet18_best.pth"

# ====================================================
# DEVICE
# ====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ====================================================
# IMAGE SETTINGS
# ====================================================

IMAGE_SIZE = 128

NUM_CLASSES = 1

# ====================================================
# TRAINING SETTINGS
# ====================================================

BATCH_SIZE = 128

EPOCHS = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Windows CPU → 0 is usually fastest
NUM_WORKERS = 0

PIN_MEMORY = False

# ====================================================
# FAST MODE
# ====================================================

FAST_MODE = True

FAST_TRAIN_IMAGES = 5000

FAST_TEST_IMAGES = 2500

# ====================================================
# NORMALIZATION
# ====================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]

# ====================================================
# DATA AUGMENTATION
# ====================================================

RANDOM_HORIZONTAL_FLIP = True

RANDOM_ROTATION = 0

COLOR_JITTER = False

# ====================================================
# PREDICTION
# ====================================================

THRESHOLD = 0.5

SUPPORTED_FORMATS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

# ====================================================
# ENSEMBLE
# ====================================================

EFFNET_WEIGHT = 0.5

RESNET_WEIGHT = 0.5

# ====================================================
# PROJECT INFO
# ====================================================

PROJECT_NAME = "AI Generated Image Detection"

VERSION = "2.0"

PRIMARY_MODEL = "EfficientNet-B0"

SECONDARY_MODEL = "ResNet18"

# ====================================================
# CONFIG SUMMARY
# ====================================================

if __name__ == "__main__":

    print("=" * 60)

    print(PROJECT_NAME)

    print("=" * 60)

    print(f"Version              : {VERSION}")
    print(f"Device               : {DEVICE}")

    print()

    print(f"Training Folder      : {TRAIN_DIR}")
    print(f"Testing Folder       : {TEST_DIR}")

    print()

    print(f"EfficientNet Model   : {EFFNET_MODEL_PATH}")
    print(f"ResNet18 Model       : {RESNET_MODEL_PATH}")

    print()

    print(f"Image Size           : {IMAGE_SIZE}")
    print(f"Batch Size           : {BATCH_SIZE}")
    print(f"Epochs               : {EPOCHS}")
    print(f"Learning Rate        : {LEARNING_RATE}")

    print()

    print(f"FAST MODE            : {FAST_MODE}")
    print(f"Train Images         : {FAST_TRAIN_IMAGES}")
    print(f"Test Images          : {FAST_TEST_IMAGES}")

    print("=" * 60)