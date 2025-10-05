from .base import BaseClassifier
from .utils import SmallResNet


class SmallResNetClassifier(BaseClassifier):
    """SmallResNet classifier model.
    Args:
        config: ModelConfig object with model settings.
    Returns: SmallResNet model with modified output layer.
    """

    def _build_model(self):
        # Use SmallResNet
        return SmallResNet(num_classes=self.config.num_classes)
