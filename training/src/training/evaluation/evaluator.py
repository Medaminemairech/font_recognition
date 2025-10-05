import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from training.config import EvalConfig


class Evaluator:
    """Evaluator for font classification models.
    Computes confusion matrix and classification report.

    Args:
        config: EvalConfig object with evaluation settings.
    Returns: dict with accuracy, classification report, confusion matrix, and missing predictions.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model_name = config.predictions_csv_path.split("/")[-2]
        self.true_labels = []
        self.pred_labels = []

    def _extract_font(self, image_path: Path) -> str:
        """Extracts the 'font' field from the image metadata."""
        try:
            info = Image.open(image_path).info
            return info.get("font", None)
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return None

    def _load_true_labels(self):
        """Loads true fonts from image metadata."""
        self.true_labels = []
        image_files = sorted(Path(self.config.images_dir).glob("*.*"))
        for img_path in image_files:
            font = self._extract_font(img_path)
            if font:
                self.true_labels.append((img_path.name, font))

    def _load_pred_labels(self):
        """Loads predictions from CSV."""
        df = pd.read_csv(self.config.predictions_csv_path)
        self.pred_labels = [
            (row["filename"], row["prediction"]) for _, row in df.iterrows()
        ]

    def evaluate(self, save_report: bool = True):
        """Compute confusion matrix and generate evaluation report."""
        self._load_true_labels()
        self._load_pred_labels()

        # Align true and predicted labels by filename
        pred_dict = dict(self.pred_labels)
        y_true = []
        y_pred = []
        missing_preds = []
        for filename, true_font in self.true_labels:
            if filename in pred_dict:
                y_true.append(true_font)
                y_pred.append(pred_dict[filename])
            else:
                missing_preds.append(filename)

        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} files")

        labels = sorted(list(set(y_true + y_pred)))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.show()

        # Classification report
        report_dict = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )

        # Overall accuracy
        acc = accuracy_score(y_true, y_pred)
        # Save report
        if save_report:
            report_path = os.path.join(self.config.report_path, self.model_name)
            print(f"Saving report to {report_path}")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write(f"Overall Accuracy: {acc:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report_str)
                if missing_preds:
                    f.write(f"\nMissing predictions for {len(missing_preds)} files:\n")
                    f.write(", ".join(missing_preds))

        return {
            "accuracy": acc,
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "missing_predictions": missing_preds,
        }
