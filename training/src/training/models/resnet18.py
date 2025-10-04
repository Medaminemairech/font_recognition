import torch.nn as nn
from torchvision import models

from .base import BaseClassifier


class ResNetClassifier(BaseClassifier):
    def _build_model(self):

        weights = (
            models.ResNet18_Weights.IMAGENET1K_V1 if self.config.pretrained else None
        )
        model = models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False

        # Replace final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.config.num_classes)
        return model
