"""
=========================================================
AI IMAGE DETECTION V2
Prediction Module
Sample Image Path: dataset\train\REAL\0000 (2).jpg
=========================================================
"""

import torch

from PIL import Image

from torchvision import transforms

from src.config import *
from src.model import get_model


# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(

        IMAGENET_MEAN,

        IMAGENET_STD

    )

])


# ============================================================
# Load Models
# ============================================================

def load_trained_models():

    # EfficientNet

    effnet = get_model("effnet").to(DEVICE)

    effnet.load_state_dict(

        torch.load(

            EFFNET_MODEL_PATH,

            map_location=DEVICE

        )

    )

    effnet.eval()

    # ResNet

    resnet = get_model("resnet").to(DEVICE)

    resnet.load_state_dict(

        torch.load(

            RESNET_MODEL_PATH,

            map_location=DEVICE

        )

    )

    resnet.eval()

    return effnet, resnet


# ============================================================
# Prediction
# ============================================================

def predict(effnet, resnet, image):

    if not isinstance(image, Image.Image):

        image = Image.open(image).convert("RGB")

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        # ---------------------------------------
        # EfficientNet Prediction
        # ---------------------------------------

        effnet_prob = torch.sigmoid(

            effnet(tensor)

        ).item()

        # ---------------------------------------
        # ResNet Prediction
        # ---------------------------------------

        resnet_prob = torch.sigmoid(

            resnet(tensor)

        ).item()

        # ---------------------------------------
        # Ensemble
        # ---------------------------------------

        ensemble_prob = (

            EFFNET_WEIGHT * effnet_prob +

            RESNET_WEIGHT * resnet_prob

        )

        prediction = (

            "FAKE"

            if ensemble_prob >= THRESHOLD

            else "REAL"

        )

        confidence = (

            ensemble_prob

            if prediction == "FAKE"

            else 1 - ensemble_prob

        ) * 100

        effnet_prediction = (

            "FAKE"

            if effnet_prob >= THRESHOLD

            else "REAL"

        )

        resnet_prediction = (

            "FAKE"

            if resnet_prob >= THRESHOLD

            else "REAL"

        )

    return {

        "prediction": prediction,

        "confidence": confidence,

        "ensemble_probability": ensemble_prob,

        "effnet_prediction": effnet_prediction,

        "effnet_probability": effnet_prob,

        "resnet_prediction": resnet_prediction,

        "resnet_probability": resnet_prob

    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    effnet, resnet = load_trained_models()

    image_path = input("Enter Image Path : ")

    image = Image.open(image_path)

    result = predict(

        effnet,

        resnet,

        image

    )

    print()

    print("=" * 60)

    print("Prediction Results")

    print("=" * 60)

    print(f"Final Prediction       : {result['prediction']}")

    print(f"Confidence            : {result['confidence']:.2f}%")

    print()

    print(f"EfficientNet          : {result['effnet_prediction']}")

    print(f"EfficientNet Prob     : {result['effnet_probability']:.4f}")

    print()

    print(f"ResNet18              : {result['resnet_prediction']}")

    print(f"ResNet18 Prob         : {result['resnet_probability']:.4f}")

    print()

    print(f"Ensemble Probability  : {result['ensemble_probability']:.4f}")

    print("=" * 60)

