"""
=========================================================
AI IMAGE DETECTION V2
Model Evaluation
=========================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

from src.config import *
from src.dataset import get_dataloaders
from src.model import get_model


# ============================================================
# Load Trained Model
# ============================================================

def load_model(model_name, model_path):

    model = get_model(model_name).to(DEVICE)

    model.load_state_dict(

        torch.load(

            model_path,

            map_location=DEVICE

        )

    )

    model.eval()

    return model


# ============================================================
# Evaluate Single Model
# ============================================================

def evaluate_model(model_name, model_path):

    print("\n" + "=" * 60)
    print(f"Evaluating {model_name.upper()}")
    print("=" * 60)

    model = load_model(

        model_name,

        model_path

    )

    _, test_loader = get_dataloaders()

    y_true = []

    y_pred = []

    y_prob = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(

                outputs

            ).cpu().numpy().flatten()

            predictions = (

                probabilities >= THRESHOLD

            ).astype(int)

            y_true.extend(

                labels.numpy()

            )

            y_pred.extend(

                predictions

            )

            y_prob.extend(

                probabilities

            )

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    print(f"Accuracy : {accuracy*100:.2f}%")

    return (

        np.array(y_true),

        np.array(y_pred),

        np.array(y_prob),

        accuracy

    )


# ============================================================
# Evaluate Ensemble
# ============================================================

def evaluate_ensemble():

    print("\n" + "=" * 60)
    print("Evaluating Ensemble Model")
    print("=" * 60)

    effnet = load_model(

        "effnet",

        EFFNET_MODEL_PATH

    )

    resnet = load_model(

        "resnet",

        RESNET_MODEL_PATH

    )

    _, test_loader = get_dataloaders()

    y_true = []

    y_pred = []

    y_prob = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            labels = labels.numpy()

            effnet_prob = torch.sigmoid(

                effnet(images)

            )

            resnet_prob = torch.sigmoid(

                resnet(images)

            )

            ensemble_prob = (

                (EFFNET_WEIGHT * effnet_prob)

                +

                (RESNET_WEIGHT * resnet_prob)

            )

            ensemble_prob = ensemble_prob.cpu().numpy().flatten()

            predictions = (

                ensemble_prob >= THRESHOLD

            ).astype(int)

            y_true.extend(

                labels

            )

            y_pred.extend(

                predictions

            )

            y_prob.extend(

                ensemble_prob

            )

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    print(f"Ensemble Accuracy : {accuracy*100:.2f}%")

    return (

        np.array(y_true),

        np.array(y_pred),

        np.array(y_prob),

        accuracy

    )
# ============================================================
# Save Evaluation Results
# ============================================================

def save_results(model_name, y_true, y_pred, y_prob):

    print("\nSaving Evaluation Results...")

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(

        y_true,

        y_pred,

        target_names=["REAL", "FAKE"]

    )

    print(report)

    report_path = OUTPUT_DIR / f"{model_name}_classification_report.txt"

    with open(report_path, "w") as file:

        file.write(report)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=["REAL", "FAKE"]

    )

    disp.plot(

        cmap="Blues",

        colorbar=False

    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(

        OUTPUT_DIR / f"{model_name}_confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(

        y_true,

        y_prob

    )

    roc_auc = auc(

        fpr,

        tpr

    )

    plt.figure(figsize=(6,6))

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"AUC = {roc_auc:.4f}"

    )

    plt.plot(

        [0,1],

        [0,1],

        linestyle="--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"{model_name} ROC Curve")

    plt.legend(loc="lower right")

    plt.grid(True)

    plt.savefig(

        OUTPUT_DIR / f"{model_name}_roc_curve.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("Results Saved Successfully")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(PROJECT_NAME)

    print("=" * 70)

    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    eff_true, eff_pred, eff_prob, eff_acc = evaluate_model(

        "effnet",

        EFFNET_MODEL_PATH

    )

    save_results(

        "EfficientNet_B0",

        eff_true,

        eff_pred,

        eff_prob

    )

    # --------------------------------------------------------
    # ResNet18
    # --------------------------------------------------------

    res_true, res_pred, res_prob, res_acc = evaluate_model(

        "resnet",

        RESNET_MODEL_PATH

    )

    save_results(

        "ResNet18",

        res_true,

        res_pred,

        res_prob

    )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ens_true, ens_pred, ens_prob, ens_acc = evaluate_ensemble()

    save_results(

        "Ensemble",

        ens_true,

        ens_pred,

        ens_prob

    )

    # --------------------------------------------------------
    # Accuracy Comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL MODEL COMPARISON")

    print("=" * 70)

    print(f"EfficientNet-B0 Accuracy : {eff_acc * 100:.2f}%")

    print(f"ResNet18 Accuracy        : {res_acc * 100:.2f}%")

    print(f"Ensemble Accuracy        : {ens_acc * 100:.2f}%")

    print("=" * 70)

    # --------------------------------------------------------
    # Accuracy Comparison Graph
    # --------------------------------------------------------

    plt.figure(figsize=(7,5))

    models = [

        "EfficientNet-B0",

        "ResNet18",

        "Ensemble"

    ]

    accuracies = [

        eff_acc * 100,

        res_acc * 100,

        ens_acc * 100

    ]

    plt.bar(

        models,

        accuracies

    )

    plt.ylabel("Accuracy (%)")

    plt.title("Model Accuracy Comparison")

    plt.ylim(0, 100)

    plt.savefig(

        OUTPUT_DIR / "accuracy_comparison.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("\nEvaluation Completed Successfully!")

    print(f"\nResults saved to : {OUTPUT_DIR}")