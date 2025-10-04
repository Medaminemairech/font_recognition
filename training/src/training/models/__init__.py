from .resnet18 import ResNetClassifier
from .mobilenet import MobileNetClassifier
from .small_resnet import SmallResNetClassifier

models_dict = {
    "resnet18": ResNetClassifier,
    "mobilenet": MobileNetClassifier,
    "smallresnet": SmallResNetClassifier,
}
