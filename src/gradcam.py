"""
=========================================================
AI IMAGE DETECTION V2
Grad-CAM Visualization
Sample Image Path: dataset\train\REAL\0000 (2).jpg
=========================================================
"""

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import *
from src.model import get_model


# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# ============================================================
# Load EfficientNet Model
# ============================================================

def load_model():

    model = get_model("effnet").to(DEVICE)

    model.load_state_dict(
        torch.load(
            EFFNET_MODEL_PATH,
            map_location=DEVICE
        )
    )

    # IMPORTANT: Enable gradients for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True

    model.eval()

    return model


# ============================================================
# Generate Grad-CAM
# ============================================================

def generate_gradcam(image_path):

    model = load_model()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    rgb_image = np.array(image).astype(np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Target convolution layer
    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Binary classifier target
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    output_path = OUTPUT_DIR / "gradcam_output.png"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    print("=" * 60)
    print("Grad-CAM generated successfully")
    print(f"Saved at: {output_path}")
    print("=" * 60)

    return visualization


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    image_path = input("Enter Image Path: ")

    generate_gradcam(image_path)
