from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from training.config import TrainingConfig


class Trainer:
    def __init__(self, model, config: TrainingConfig, classes_dict=None):
        if not torch.cuda.is_available():
            config.device = "cpu"
        self.model_name = model.config.name
        self.model = model.to(config.device).model

        self.config = config
        self.classes_dict = classes_dict

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        elif config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=config.lr, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # LR Scheduler
        self.scheduler = None
        if config.lr_scheduler:
            if config.lr_scheduler.lower() == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=2
                )
            else:
                raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")

        # Early stopping
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # Checkpointing
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, f"{self.model_name}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def save_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, f"best.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "number_of_classes": self.model.fc.out_features,
                "val_loss": val_loss,
                "classes_dict": self.classes_dict,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc="Training")
        for imgs, labels in loop:
            imgs, labels = imgs.to(self.config.device), labels.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss / total, acc=100 * correct / total)

        return running_loss / total, 100 * correct / total

    def validate_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            loop = tqdm(val_loader, desc="Validation")
            for imgs, labels in loop:
                imgs, labels = imgs.to(self.config.device), labels.to(
                    self.config.device
                )
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                loop.set_postfix(
                    val_loss=val_loss / val_total, val_acc=100 * val_correct / val_total
                )

        return val_loss / val_total, 100 * val_correct / val_total

    def fit(self, train_loader, val_loader):
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
            )

            # Save metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping & checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

    def plot_history(self):
        """Plot training & validation loss and accuracy curves."""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        plt.figure(figsize=(14, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label="Train Acc")
        plt.plot(epochs, self.history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)

        plt.show()
