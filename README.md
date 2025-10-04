# Project Overview

This repository contains two main components:

- **training/**: Tools and scripts for training, evaluating, and exporting deep learning models.
- **serving/**: FastAPI-based web API for serving trained models and making predictions.

---

## Training

The `training` folder provides:

- Configurable pipelines for model training and evaluation.
- Support for multiple architectures (e.g., ResNet, MobileNet).
- Data loading and preprocessing utilities.
- Scripts for inference and exporting models to ONNX format.

## Serving

The `serving` folder provides:

- FastAPI app for model inference via HTTP endpoints.
- Utilities for loading ONNX models and preprocessing inputs.
- Example endpoint: `/predict` for image classification.

**Notes:**
- Export models from `training` and place them in `serving/model/`.
- Update model paths in `serving/api.py` as needed.

---

For more details, see the individual README files in each folder.