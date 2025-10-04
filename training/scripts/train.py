import argparse
import logging

from training.config import DataConfig, ModelConfig, TrainingConfig
from training.data_pipelines import ImageDataModule
from training.models import models_dict
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model_name: str = "resnet18"):
    """
    Main function to set up data, model, and start training.
    """
    # Configuration
    logger.info("Setting up configurations...")
    data_config = DataConfig(batch_size=256, subset_fraction=None)
    model_config = ModelConfig(name=model_name)
    train_config = TrainingConfig(batch_size=data_config.batch_size)

    # Data Preparation
    logger.info("Preparing data...")
    data_preprocessor = ImageDataModule(data_config)
    train_loader, val_loader = data_preprocessor.setup_training()
    model_config.num_classes = data_preprocessor.num_classes
    logger.info(f"{data_preprocessor.summary}")

    # Model Initialization
    logger.info(f"Initializing model: {model_config.name}...")
    model = models_dict[model_config.name](model_config)

    # Training
    logger.info("Starting training...")
    trainer = Trainer(model, train_config, classes_dict=data_preprocessor.classes_dict)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_name = parser.add_argument(
        "--model_name",
        type=str,
        default="smallresnet",
        help="Name of the model to use.",
    )
    args = parser.parse_args()
    train(args.model_name)
