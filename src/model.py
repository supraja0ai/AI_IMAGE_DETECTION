"""
====================================================
AI IMAGE DETECTION V2
EfficientNet-B0 + ResNet18
====================================================
"""

import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    resnet18,
    ResNet18_Weights
)


# ====================================================
# EfficientNet-B0
# ====================================================

def get_effnet_model():

    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # ---------------------------------------------
    # Freeze entire backbone
    # ---------------------------------------------

    for param in model.parameters():
        param.requires_grad = False

    # ---------------------------------------------
    # Replace classifier
    # ---------------------------------------------

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            in_features,
            1
        )

    )

    # ---------------------------------------------
    # Train only classifier
    # ---------------------------------------------

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ====================================================
# ResNet18
# ====================================================

def get_resnet_model():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # ---------------------------------------------
    # Freeze backbone
    # ---------------------------------------------

    for param in model.parameters():
        param.requires_grad = False

    # ---------------------------------------------
    # Replace classifier
    # ---------------------------------------------

    in_features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            in_features,
            1
        )

    )

    # ---------------------------------------------
    # Train only classifier
    # ---------------------------------------------

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# ====================================================
# Model Factory
# ====================================================

def get_model(model_name):

    model_name = model_name.lower()

    if model_name == "effnet":
        return get_effnet_model()

    elif model_name == "resnet":
        return get_resnet_model()

    else:
        raise ValueError(
            f"Unknown model : {model_name}"
        )


# ====================================================
# Ensemble Models
# ====================================================

def get_ensemble_models():

    effnet = get_effnet_model()

    resnet = get_resnet_model()

    return effnet, resnet


# ====================================================
# Test
# ====================================================

if __name__ == "__main__":

    print("=" * 60)

    print("Loading Models...")

    print("=" * 60)

    effnet = get_model("effnet")

    resnet = get_model("resnet")

    print("✓ EfficientNet-B0 Loaded")

    print("✓ ResNet18 Loaded")

    print()

    print("Ensemble Ready")

    print("=" * 60)