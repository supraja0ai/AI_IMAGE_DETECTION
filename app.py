import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

import streamlit as st
from PIL import Image

from predict import (
    load_trained_models,
    predict
)

# ====================================================
# PAGE CONFIG
# ====================================================

st.set_page_config(

    page_title="AI Generated Image Detection",

    page_icon="🛡️",

    layout="wide"

)

# ====================================================
# LOAD MODELS
# ====================================================

@st.cache_resource
def load_models():

    return load_trained_models()

effnet, resnet = load_models()

# ====================================================
# SIDEBAR
# ====================================================

with st.sidebar:

    st.image(

        "https://img.icons8.com/color/96/artificial-intelligence.png",

        width=80

    )

    st.title("Real vs AI Image Detector")

    st.markdown("---")

    st.success("Models Loaded Successfully")

    st.write("""

### 🔍 About the Model

This application uses an **Ensemble Deep Learning Architecture**
consisting of:

- 🧠 EfficientNet-B0
- 🧠 ResNet18

Both models are trained independently using Transfer Learning
on the CIFAKE dataset.

The final prediction is obtained using a
**Weighted Ensemble Strategy**, improving robustness
and generalization over a single model.

### 🚀 Key Features

- 📷 REAL vs AI Generated Detection
- 🧠 EfficientNet-B0
- 🧠 ResNet18
- ⚖️ Ensemble Prediction
- 🎯 Confidence Score
- ⚡ Fast CPU Inference
- 🌐 Streamlit Web Application

### 📌 Workflow

Upload Image

↓

Preprocessing

↓

EfficientNet-B0 + ResNet18

↓

Weighted Ensemble

↓

Prediction

""")

    st.markdown("---")

    st.markdown("### 📊 Model Details")

    st.markdown("**Primary Model**")

    st.info("EfficientNet-B0")

    st.markdown("**Secondary Model**")

    st.info("ResNet18")

    st.markdown("**Inference**")

    st.info("Weighted Ensemble")

    st.markdown("**Classes**")

    st.info("REAL / FAKE")

    st.markdown("---")

# ====================================================
# HEADER
# ====================================================

st.title("🛡️ AI Generated Image Detection System")

st.caption(

    "Deep Learning based Image Authenticity Verification using Ensemble Learning"

)

st.divider()

# ====================================================
# FILE UPLOAD
# ====================================================

uploaded_file = st.file_uploader(

    "📤 Upload an Image\n\nUpload an image to determine whether it is REAL or AI GENERATED.",

    type=[

        "jpg",

        "jpeg",

        "png"

    ]

)
# ====================================================
# PREDICTION
# ====================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.2, 1])

    # ------------------------------------------------
    # Uploaded Image
    # ------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(

            image,

            use_container_width=True

        )

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------

    start = time.time()

    result = predict(

        effnet,

        resnet,

        image

    )

    end = time.time()

    prediction = result["prediction"]

    confidence = result["confidence"]

    # ------------------------------------------------
    # Prediction Output
    # ------------------------------------------------

    with col2:

        st.subheader("Final Prediction")

        if prediction == "REAL":

            st.success("✅ REAL IMAGE")

        else:

            st.error("🚨 AI GENERATED IMAGE")

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

        st.progress(confidence / 100)

        st.metric(

            "Inference Time",

            f"{end-start:.3f} sec"

        )

    st.divider()

    # ====================================================
    # Individual Model Predictions
    # ====================================================

    st.subheader("Model Predictions")

    c1, c2 = st.columns(2)

    with c1:

        st.info("🧠 EfficientNet-B0")

        st.metric(

            "Prediction",

            result["effnet_prediction"]

        )

        st.metric(

            "Confidence",

            f"{result['effnet_probability']*100:.2f}%"

        )

    with c2:

        st.info("🧠 ResNet18")

        st.metric(

            "Prediction",

            result["resnet_prediction"]

        )

        st.metric(

            "Confidence",

            f"{result['resnet_probability']*100:.2f}%"

        )

    st.divider()

    # ====================================================
    # Ensemble Information
    # ====================================================

    st.subheader("Ensemble Decision")

    st.write("""

The final prediction is generated using a **Weighted Ensemble**
of both deep learning models.

**Final Probability**

= (0.60 × EfficientNet-B0)

+ (0.40 × ResNet18)

This improves prediction robustness and generalization compared
to relying on a single model.

""")

    st.info(

        f"Final Ensemble Probability : {result['ensemble_probability']*100:.2f}%"

    )

    st.divider()

    # ====================================================
    # Interpretation
    # ====================================================

    if prediction == "REAL":

        st.success(

            "The uploaded image is classified as an authentic (REAL) image based on the combined predictions of EfficientNet-B0 and ResNet18."

        )

    else:

        st.warning(

            "The uploaded image is classified as AI Generated (FAKE) based on the combined predictions of EfficientNet-B0 and ResNet18."

        )

    st.write("""

The confidence score represents how certain the ensemble model
is about its final prediction.

Higher confidence indicates stronger agreement between the
models and greater certainty in the classification.

""")

# ====================================================
# FOOTER
# ====================================================

st.divider()

st.caption(

    "Built using PyTorch | Streamlit | EfficientNet-B0 | ResNet18 | Ensemble Learning"

)