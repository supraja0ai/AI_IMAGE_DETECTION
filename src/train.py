"""
=========================================================
AI IMAGE DETECTION V2
Train EfficientNet-B0 + ResNet18
=========================================================
"""

import copy
import torch
import torch.nn as nn

from tqdm import tqdm

from src.config import *
from src.dataset import get_dataloaders
from src.model import get_model


# ============================================================
# Common Training Function
# ============================================================

def train_model(model_name, save_path):

    print("\n" + "=" * 70)
    print(f"Training {model_name.upper()}")
    print("=" * 70)

    # ---------------------------------------
    # Dataset
    # ---------------------------------------

    train_loader, test_loader = get_dataloaders()

    print(f"Training Images : {len(train_loader.dataset)}")
    print(f"Testing Images  : {len(test_loader.dataset)}")

    # ---------------------------------------
    # Model
    # ---------------------------------------

    model = get_model(model_name).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(

        filter(

            lambda p: p.requires_grad,

            model.parameters()

        ),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    best_accuracy = 0.0

    best_model = copy.deepcopy(

        model.state_dict()

    )

    # ====================================================
    # Training Loop
    # ====================================================

    for epoch in range(EPOCHS):

        print()

        print("-" * 70)

        print(f"Epoch {epoch+1}/{EPOCHS}")

        print("-" * 70)

        model.train()

        running_loss = 0

        progress = tqdm(

            train_loader,

            desc=f"{model_name.upper()}"

        )

        for images, labels in progress:

            images = images.to(DEVICE)

            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(

                outputs,

                labels

            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(

                Loss=f"{loss.item():.4f}"

            )

        train_loss = running_loss / len(train_loader)

        # ============================================
        # Validation
        # ============================================

        model.eval()

        correct = 0

        total = 0

        validation_loss = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(DEVICE)

                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(images)

                loss = criterion(

                    outputs,

                    labels

                )

                validation_loss += loss.item()

                predictions = (

                    torch.sigmoid(outputs) >= THRESHOLD

                ).float()

                correct += (

                    predictions == labels

                ).sum().item()

                total += labels.size(0)

        validation_loss /= len(test_loader)

        accuracy = correct / total
        # ====================================================
        # Epoch Summary
        # ====================================================

        print()

        print(f"Train Loss      : {train_loss:.4f}")

        print(f"Validation Loss : {validation_loss:.4f}")

        print(f"Accuracy        : {accuracy * 100:.2f}%")

        # ====================================================
        # Save Best Model
        # ====================================================

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_model = copy.deepcopy(

                model.state_dict()

            )

            torch.save(

                best_model,

                save_path

            )

            print()

            print("Best Model Saved")

            print(f"Location : {save_path}")

    print()

    print("=" * 70)

    print(f"{model_name.upper()} TRAINING COMPLETED")

    print(f"Best Accuracy : {best_accuracy * 100:.2f}%")

    print(f"Model Saved   : {save_path}")

    print("=" * 70)


# ============================================================
# Train Both Models
# ============================================================

def train():

    print("=" * 70)

    print(PROJECT_NAME)

    print("=" * 70)

    print(f"Device           : {DEVICE}")

    print(f"Training Folder  : {TRAIN_DIR}")

    print(f"Testing Folder   : {TEST_DIR}")

    print(f"Epochs           : {EPOCHS}")

    print(f"Batch Size       : {BATCH_SIZE}")

    print(f"Image Size       : {IMAGE_SIZE}")

    print("=" * 70)

    # --------------------------------------------------------
    # Train EfficientNet-B0
    # --------------------------------------------------------

    train_model(

        model_name="effnet",

        save_path=EFFNET_MODEL_PATH

    )

    # --------------------------------------------------------
    # Train ResNet18
    # --------------------------------------------------------

    train_model(

        model_name="resnet",

        save_path=RESNET_MODEL_PATH

    )

    print()

    print("=" * 70)

    print("ALL MODELS TRAINED SUCCESSFULLY")

    print("=" * 70)

    print()

    print(f"EfficientNet Model : {EFFNET_MODEL_PATH}")

    print(f"ResNet18 Model     : {RESNET_MODEL_PATH}")

    print()

    print("Ready for Evaluation and Prediction")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train()