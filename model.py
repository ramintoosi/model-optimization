"""
This module defines the model.
"""

from torch import nn
from torchvision.models import resnet18


def get_resnet18(num_classes: int = 10) -> nn.Module:
    """
    Get the ResNet-18 model.

    :param num_classes: Number of classes for the output layer.
    :return: ResNet-18 model.
    """
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
