# Bone Fracture Detection

This repository contains a bone fracture detection project implemented using **CNN**, **ResNet**, and **YOLO** models. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Results](#results)
- [File Structure](#file-structure)
- [How to Use](#how-to-use)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

Bone fracture detection is an essential task in medical diagnostics. This project automates fracture detection using three models:
- **Convolutional Neural Network (CNN)**
- **ResNet (Residual Network)**
- **YOLO (You Only Look Once)**

The repository also includes a performance comparison graph (Epochs vs. Accuracy).

---

## Dataset

The dataset is sourced from Kaggle:  
[Bone Fracture Detection Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project).  
Download the dataset and place it in the `dataset/` directory.

---

## Features

- **Multi-model Implementation**:
    - CNN: Standard deep learning architecture for image classification.
    - ResNet: Pre-trained ResNet architecture fine-tuned for bone fracture detection.
    - YOLO: Real-time object detection model for detecting fractures.

- **Performance Metrics**:
    - Accuracy, Precision, Recall, and F1 Score for all models.

- **Visualization**:
    - Graph of Epochs vs. Accuracy for comparing model performance.

---

## Results

The following metrics were used for evaluation:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

The performance graph is saved in the `graphs/` directory.

---
