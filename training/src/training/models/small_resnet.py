from .base import BaseClassifier
from .utils import SmallResNet


class SmallResNetClassifier(BaseClassifier):
    def _build_model(self):
        # Use SmallResNet
        return SmallResNet(num_classes=self.config.num_classes)
