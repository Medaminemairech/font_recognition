from training.config import ModelConfig


class BaseClassifier:
    """Base class for classification models.
    Args:
        config: ModelConfig object with model settings.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _build_model method")

    def to(self, device):
        self.model = self.model.to(device)
        return self
