"""
This module defines the model.
"""

from torch import nn
from torchvision.models import resnet34, ResNet34_Weights
import torchsummary


def get_model(num_classes: int = 10) -> nn.Module:
    """
    Get the model.

    :param num_classes: Number of classes for the output layer.
    :return: Pytorch model.
    """
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(model)
    return model
