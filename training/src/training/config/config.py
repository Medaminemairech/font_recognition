from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    data_dir: str = (
        "c:/Users/amine/Downloads/reduced_technical_test/dataset_word_color/"
    )
    img_size: int = 224
    train_split: float = 0.95
    subset_fraction: Optional[float] = None
    batch_size: int = 512
    num_classes: int = 23  # This will be updated based on the dataset


@dataclass
class ModelConfig:
    name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 23


@dataclass
class TrainingConfig:
    num_epochs: int = 20
    device: str = "cuda"  # or "cpu"
    lr: float = 1e-4
    batch_size: int = 32
    optimizer: str = "adam"
    lr_scheduler: str = "plateau"
    early_stopping_patience: int = 5
    checkpoint_dir: str = (
        "c:/Users/amine/Downloads/PUBLICIS/training/artifacts/checkpoints"
    )


@dataclass
class TestConfig:
    test_dir: str = "c:/Users/amine/Downloads/reduced_technical_test/dataset_image_test"
    batch_size: int = 32
    img_size: int = 224
    model_name: str = "resnet18"
    checkpoint_path: str = (
        "c:/Users/amine/Downloads/PUBLICIS/training/artifacts/checkpoints/best_model.pth"
    )
    artifacts_dir: str = "c:/Users/amine/Downloads/PUBLICIS/training/artifacts"


@dataclass
class EvalConfig:
    images_dir: str = (
        "c:/Users/amine/Downloads/reduced_technical_test/dataset_image_test"
    )
    predictions_csv_path: str = None
    report_path: str = (
        "c:/Users/amine/Downloads/PUBLICIS/training/artifacts/evaluation_reports"
    )
