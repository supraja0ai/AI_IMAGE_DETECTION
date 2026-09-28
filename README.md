# AI-Generated Image Detection System

An end-to-end deep learning project for classifying images as **Real**
or **AI-Generated (Fake)**. The project uses a ResNet18-based image
classifier, a Streamlit interface, and Grad-CAM visualizations to help
inspect model predictions.

## Project Overview

The system explores binary image classification using the CIFAKE
dataset. It includes data preparation, model training, evaluation,
prediction, and an interactive application.

### Key Features

-   **Binary image classification:** Predicts whether an image is Real
    or Fake.
-   **Deep learning:** Uses a ResNet18-based convolutional neural
    network implemented with PyTorch.
-   **Interactive interface:** Provides image upload and prediction
    through Streamlit.
-   **Explainability:** Uses Grad-CAM to visualize image regions that
    influence a prediction.
-   **Evaluation:** Supports analysis with classification metrics and
    visualizations.

## Technology Stack

-   Python
-   PyTorch and Torchvision
-   ResNet18
-   Streamlit
-   OpenCV
-   NumPy
-   Matplotlib
-   scikit-image
-   pytorch-grad-cam

## Dataset

This project uses the [CIFAKE dataset on
Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images),
which contains real images and AI-generated synthetic images.

The full dataset is not included in this repository. Download it
separately and arrange it in the following structure:

``` text
dataset/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

The commonly distributed CIFAKE dataset contains 50,000 training images
per class and 10,000 test images per class, at 32 × 32 pixels.

## Project Structure

``` text
AI_IMAGE_DETECTION/
├── app.py
├── config.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── models/
├── dataset/
├── outputs/
└── README.md
```

The tree above is a representative layout. Actual files and directories
may vary by project version. Dataset files, model checkpoints, and
temporary outputs may be excluded from Git to keep the repository
manageable.

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/supraja0ai/AI_IMAGE_DETECTION.git
cd AI_IMAGE_DETECTION
```

### 2. Create and activate a virtual environment

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download CIFAKE from Kaggle and place the extracted images in the
directory structure shown in the Dataset section. Ensure that the paths
in the project configuration match your local dataset location.

### 5. Prepare a trained model

If the application requires a trained checkpoint, place it at the model
path configured by the project (for example, `models/best_model.pth`).
If you do not have a checkpoint, train the model using the project's
training script.

## Training and Evaluation

From the project root, the typical commands are:

**Train**

``` bash
python src/train.py
```

**Evaluate**

``` bash
python src/evaluate.py
```

These commands assume the scripts use these entry points and their
default configuration. If your local version uses different paths or
arguments, follow the options defined in the scripts.

## Run the Streamlit Application

From the project root:

``` bash
streamlit run app.py
```

Open the local URL printed in the terminal. Upload an image through the
interface to view the model's prediction.

## Explainable AI with Grad-CAM

The project uses Gradient-weighted Class Activation Mapping (Grad-CAM)
to create heatmaps highlighting image regions associated with a model
prediction.

Grad-CAM can help inspect model behavior, but a heatmap is not proof
that a prediction is correct and does not establish that the highlighted
regions are causal.

## Model Evaluation

Use the project's evaluation pipeline to examine suitable metrics, such
as:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   ROC curve and ROC-AUC, where supported

Add the actual results from your held-out test set and include
evaluation plots here once they have been generated and verified. Avoid
reporting unverified or training-only metrics as test performance.

## Limitations

-   CIFAKE images are small (32 × 32 pixels); results may not generalize
    to higher-resolution images.
-   Performance on unseen image generators, image-editing pipelines, or
    real-world photographs may differ from performance on CIFAKE.
-   A model prediction is an estimate, not definitive proof of image
    authenticity.
-   Grad-CAM is an interpretation aid and should not be treated as a
    guarantee of model correctness.

## Future Improvements

-   Evaluate on diverse, higher-resolution datasets.
-   Test generalization to image generators not represented in training.
-   Explore frequency-domain and other image-forensics features.
-   Add confidence calibration and robustness testing.
-   Package and deploy the application.

## Author

**P Supraja**\
AI/ML Engineer \| Deep Learning \| Computer Vision \| Explainable AI

GitHub: [@supraja0ai](https://github.com/supraja0ai)

------------------------------------------------------------------------

This project is intended for educational and research purposes. Its
predictions should not be treated as conclusive evidence of image
authenticity.
