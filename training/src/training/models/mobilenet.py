import torch.nn as nn
from torchvision import models

from .base import BaseClassifier


class MobileNetClassifier(BaseClassifier):
    def _build_model(self):

        weights = (
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if self.config.pretrained
            else None
        )
        model = models.mobilenet_v3_small(weights=weights)
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier head
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, self.config.num_classes)
        return model
