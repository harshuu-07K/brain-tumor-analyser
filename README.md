# 🧠 Brain Tumor MRI Image Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning system and interactive clinical decision-support web application for automated detection and 4-class classification of brain tumors from magnetic resonance imaging (MRI) scans.

---

## 📌 Project Overview & Clinical Significance

Brain tumors are among the most lethal and complex oncological conditions, requiring rapid and precise diagnostic triage. Conventional radiological evaluation of multi-sequence MRI scans is time-consuming and subject to inter-observer variability. 

**NeuroScan AI** provides automated, deep learning-assisted classification of brain MRI scans into four categories:
1. **Glioma:** Highly invasive tumors originating in glial tissue, demanding rapid neurosurgical assessment.
2. **Meningioma:** Typically benign extra-axial tumors arising from the meninges with clear margins.
3. **Pituitary Adenoma:** Tumors of the pituitary gland causing hormonal dysregulation and visual pathway compression.
4. **No Tumor (Healthy Control):** Anatomically normal brain parenchyma without mass lesions.

### Core Objectives:
- **Triage & Acceleration:** Flag high-risk intracranial lesions for expedited specialist review.
- **Second-Opinion Support:** Provide objective diagnostic support in under-resourced hospital settings.
- **Benchmarking:** Empirically compare a purpose-built **Custom Deep CNN** against a pretrained **Transfer Learning architecture (MobileNetV2)**.

---

## 📊 Dataset Overview

The dataset consists of **2,443 brain MRI scans** organized into training, validation, and testing partitions across 4 distinct categories:

| Class | Training Images | Validation Images | Testing Images | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 564 | 161 | 80 | 805 |
| **Meningioma** | 358 | 124 | 63 | 545 |
| **Pituitary** | 438 | 118 | 54 | 610 |
| **No Tumor** | 335 | 99 | 49 | 483 |
| **Total** | **1,695** | **502** | **246** | **2,443** |

### Data Augmentation & Preprocessing:
- Rescaling pixel intensities from `[0, 255]` to `[0.0, 1.0]`.
- Spatial transformations: Random rotations (±15°), horizontal flipping, zoom (10%), and width/height shifts (8%).
- Standardized image resolution: `224 × 224 × 3`.

---

## 🏗️ Architecture & Model Design

### 1. Custom Deep CNN
- **Input:** `(224, 224, 3)`
- **Feature Extractor:** 4 sequential Convolutional Blocks:
  - Conv2D (32, 64, 128, 256 filters) with `3×3` kernels and ReLU activation.
  - Batch Normalization after every Conv layer for training stability and vanishing gradient mitigation.
  - Max Pooling (`2×2`) for spatial downsampling.
  - Spatial Dropout (0.20 to 0.35) for regularization.
- **Classification Head:**
  - Global Average Pooling (GAP) layer to reduce parameters and prevent overfitting.
  - Fully connected Dense layer (256 units) + Batch Normalization + Dropout (0.50).
  - Dense output layer (4 units) with Softmax activation.

### 2. Transfer Learning (MobileNetV2)
- **Base:** Pretrained on ImageNet with frozen lower feature-extraction weights.
- **Top Head:** Custom GlobalAveragePooling2D → Dense(256) → Batch Normalization → Dropout(0.40) → Softmax(4).
- **Fine-Tuning:** Unfreezing the top 30 layers and retraining with a low learning rate (`1e-5`) for domain adaptation to medical MRI textures.

---

## 📈 Evaluation & Comparative Analysis

Both models are evaluated on the independent test set (246 images) using standard clinical classification metrics:

| Metric | Custom CNN | MobileNetV2 (Transfer Learning) | Performance Gain |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 69.92% | **90.65%** | **+20.73%** |
| **Weighted Precision** | 68.60% | **90.61%** | **+22.01%** |
| **Weighted Recall (Sensitivity)** | 69.92% | **90.65%** | **+20.73%** |
| **Weighted F1-Score** | 61.86% | **90.40%** | **+28.54%** |

The comparative performance and diagnostic curves are rendered dynamically within the Streamlit application and exported to `/plots`.

---

## 🖥️ Streamlit Web Application Features

website link : https://brain-tumor-analyser.streamlit.app/

The interactive dashboard (`app.py`) includes:
- **🩺 Diagnostic Assistant:**
  - Dynamic model selection (Custom CNN vs MobileNetV2).
  - Dual input options: Upload custom DICOM-converted scans (JPG/PNG) or select from validated test samples.
  - Real-time prediction card with confidence meter and color-coded risk alerts.
  - Interactive class probability distribution chart.
  - Detailed clinical profile, common symptoms, MRI radiological findings, and recommended clinical urgency.
- **📊 Model Comparison & Analytics:**
  - Side-by-side metric tables and benchmark comparison charts.
  - Confusion matrix heatmaps and training/validation learning curves.
- **🔬 Dataset Explorer:**
  - Split breakdown and class distribution histograms.
  - Multi-class MRI visual gallery.
- **ℹ️ Clinical Guide & Disclaimer:**
  - Comprehensive documentation and medical disclaimers.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshuu-07K/brain-tumor-analyser.git
cd brain-tumor-analyser
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
The dataset is not included in this repository due to size constraints (~75 MB).  
Download it from the [Roboflow Brain Tumor MRI Dataset](https://roboflow.com/) and place it in the following structure:
```
database/
├── train/     # glioma/, meningioma/, no_tumor/, pituitary/
├── valid/
└── test/
```

### 5. Train Models
To run the full end-to-end training and evaluation pipeline:
```bash
python train_models.py
```
*Trained models will be saved to `/models` and diagnostic plots to `/plots`.*

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 6. Run Standalone Evaluation
To evaluate any model checkpoint against the test set:
```bash
python evaluate.py --model models/mobilenet_v2.keras --test_dir database/test
```

---

## 📁 Repository Directory Structure

```
├── database/
│   ├── train/               # 1,695 training images across 4 classes
│   ├── valid/               # 502 validation images
│   └── test/                # 246 testing images
├── models/
│   ├── custom_cnn.keras     # Custom CNN trained weights
│   ├── custom_cnn.h5        # HDF5 model format
│   ├── mobilenet_v2.keras   # MobileNetV2 trained weights
│   └── mobilenet_v2.h5      # HDF5 model format
├── plots/
│   ├── Custom_CNN_history.png
│   ├── Custom_CNN_confusion_matrix.png
│   ├── MobileNetV2_history.png
│   ├── MobileNetV2_confusion_matrix.png
│   └── model_comparison.png
├── sample_images/           # Test MRI scans for demo testing
├── app.py                   # Streamlit web application
├── train_models.py          # End-to-end training pipeline
├── evaluate.py              # Standalone model evaluation script
├── Brain_Tumor_Classification.ipynb # Interactive Jupyter Notebook
├── requirements.txt         # Project dependencies
└── README.md                # Comprehensive documentation
```

---

## ⚕️ Clinical Disclaimer
This system is developed strictly for research, educational, and benchmarking purposes. It is **not** an FDA-cleared medical device. Decisions regarding patient care must always be made in consultation with board-certified radiologists, neuro-oncologists, and neurosurgeons.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more details.
