# Skin Cancer Detection

A comprehensive pipeline for the detection and classification of **skin lesions** using deep learning models, developed as part of a university project in **Data Mining and Machine Learning**.

This work investigates the potential of **CAD systems** (Computer-Aided Diagnosis) to assist dermatologists by leveraging the power of **image preprocessing**, **segmentation**, **feature extraction**, and **deep classification models** on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

---

## Project Objective

- Evaluate the advantages of deep learning-based CAD tools for skin lesion detection.
- Develop a structured pipeline for image-based classification.
- Tackle challenges such as **class imbalance** and **domain adaptation**.
- Experiment with segmented vs. non-segmented image pipelines.

---

## Pipeline Overview

```text
[RAW IMAGES]
    ↓
[PREPROCESSING]
    - Hair removal
    - CLAHE (Contrast enhancement)
    - Soft sharpening
    - Resize/rescale
    ↓
[SEGMENTATION]
    - U-Net with ResNet34 encoder (pre-trained)
    ↓
[FEATURE EXTRACTION & AUGMENTATION]
    ↓
[CLASSIFICATION]
    - EfficientNetB0 (fine-tuned & frozen versions)
    ↓
[ANALYSIS & EVALUATION]


