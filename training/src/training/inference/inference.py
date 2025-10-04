import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from training.data_pipelines import TestDataLoader
from training.config import ModelConfig, TestConfig
from training.models import models_dict


class TestRunner:
    """
    Runs inference on a test dataset and saves predictions.
    """

    def __init__(self, config: TestConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_loader()
        self._prepare_model()

    def _prepare_loader(self):
        self.test_loader = TestDataLoader(
            test_dir=self.config.test_dir,
            batch_size=self.config.batch_size,
            img_size=self.config.img_size,
        )

    def _prepare_model(self):
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.idx_to_class = {v: k for k, v in checkpoint["classes_dict"].items()}
        # Create model
        self.model = models_dict[self.config.model_name](
            ModelConfig(
                name=self.config.model_name, num_classes=checkpoint["number_of_classes"]
            )
        )
        # Load checkpoint

        self.model.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.model.eval()

    def run(self):
        all_predictions = []
        all_filenames = []

        with torch.no_grad():
            for images, filenames in self.test_loader:
                images = images.to(self.device)
                outputs = self.model.model(images)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().tolist()

                # Map back to class names if mapping is provided
                if self.idx_to_class:
                    preds = [self.idx_to_class[p] for p in preds]

                all_predictions.extend(preds)
                all_filenames.extend(filenames)

        # Ensure artifacts folder exists

        artifacts_dir = os.path.join(
            self.config.artifacts_dir, f"predictions/{self.config.model_name}"
        )
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        file_name = self.config.checkpoint_path.split("/")[-1].replace(
            ".pt", "_predictions"
        )
        output_path = artifacts_dir / f"{file_name}.csv"
        df = pd.DataFrame({"filename": all_filenames, "prediction": all_predictions})
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        return output_path
