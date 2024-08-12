from typing import Optional, Any, Callable

import torch
from torch import nn, Tensor
from torchvision.models.resnet import ResNet34_Weights, ResNet, BasicBlock, _resnet
from torch.nn.quantized import FloatFunctional


class QBasicBlock(BasicBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff = FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.ff.add(out, identity)
        out = self.relu(out)

        return out


def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet34_Weights.verify(weights)

    return _resnet(QBasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def get_model(num_classes: int = 10) -> nn.Module:
    """
    Get the model.

    :param num_classes: Number of classes for the output layer.
    :return: Pytorch model.
    """
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
