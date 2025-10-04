# Training Toolkit

This repository provides a modular toolkit for training, evaluating, and deploying deep learning models, with a focus on image classification tasks.

## Features

- **Configurable Training Pipeline:** Easily set up data, models, and training parameters using configuration classes.
- **Model Zoo:** Includes implementations for ResNet18, SmallResNet, and MobileNetV3.
- **Data Pipelines:** Utilities for loading, preprocessing image datasets.
- **Evaluation & Inference:** Scripts and modules for model evaluation and inference.
- **Export Utilities:** Export trained models to ONNX format for deployment.
- **Metadata Generation:** Tools for generating dataset metadata.

## Directory Structure

```
training/
├── artifacts/           # Model checkpoints, evaluation reports, predictions, metadata
├── build/               # Build artifacts
├── dev/                 # Development notebooks and ONNX models
├── scripts/             # Training, evaluation, and inference scripts
├── src/
│   └── training/
│       ├── config/      # Configuration classes
│       ├── data_pipelines/ # Data loading and preprocessing
│       ├── evaluation/  # Evaluation utilities
│       ├── inference/   # Inference utilities
│       ├── models/      # Model definitions
│       ├── trainer/     # Training loop and logic
│       ├── utils/       # Utility functions (export, metadata, etc.)
├── tests/               # Unit tests
├── pyproject.toml       # Project metadata and dependencies
└── README.md            # This file
```

## Getting Started

### Installation

Install dependencies using [uv]

```sh
uv  pip install .
```

### Training a Model

Run the training script:

```sh
python training/scripts/train.py --model_name smallresnet
```

### Inference

```sh
python training/scripts/inference.py --model_path artifacts/checkpoints/smallresnet/best.pt --input_path <image_path>
```

### Evaluating a Model

```sh
python training/scripts/evaluate.py --model_name smallresnet
```



## Configuration

- Model, data, and training configurations are managed via classes in [`src/training/config`](src/training/config).

## Exporting Models

Export trained models to ONNX format using [`src/training/utils/export_onnx.py`](src/training/utils/export_onnx.py).

## Metadata Generation

Generate dataset metadata using [`src/training/utils/generate_metadata.py`](src/training/utils/generate_metadata.py).
