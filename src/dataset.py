"""
====================================================
AI IMAGE DETECTION V2
Dataset Loader
====================================================
"""

import random

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import datasets
from torchvision import transforms

from src.config import *

# ====================================================
# Reproducibility
# ====================================================

random.seed(42)

# ====================================================
# TRAIN TRANSFORMS
# ====================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])

# ====================================================
# TEST TRANSFORMS
# ====================================================

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])

# ====================================================
# DATASET
# ====================================================

train_dataset = datasets.ImageFolder(

    TRAIN_DIR,

    transform=train_transform

)

test_dataset = datasets.ImageFolder(

    TEST_DIR,

    transform=test_transform

)

# ====================================================
# FAST MODE
# ====================================================

if FAST_MODE:

    print("\nFAST MODE ENABLED")

    train_indices = random.sample(

        range(len(train_dataset)),

        min(
            FAST_TRAIN_IMAGES,
            len(train_dataset)
        )

    )

    test_indices = random.sample(

        range(len(test_dataset)),

        min(
            FAST_TEST_IMAGES,
            len(test_dataset)
        )

    )

    train_dataset = Subset(

        train_dataset,

        train_indices

    )

    test_dataset = Subset(

        test_dataset,

        test_indices

    )

# ====================================================
# DATALOADER
# ====================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)

# ====================================================
# GET DATALOADERS
# ====================================================

def get_dataloaders():

    return train_loader, test_loader


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":

    print("=" * 60)

    print("DATASET INFORMATION")

    print("=" * 60)

    print(f"Training Images : {len(train_dataset)}")

    print(f"Testing Images  : {len(test_dataset)}")

    print(f"Training Batches : {len(train_loader)}")

    print(f"Testing Batches  : {len(test_loader)}")

    print("=" * 60)