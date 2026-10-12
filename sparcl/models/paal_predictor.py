import torch
import torch.nn as nn
from torchvision.models import resnet18


class AccuracyPredictor(nn.Module):
    """PAAL accuracy predictor: ResNet-18 regressing per-image Dice from [image ; foreground probability]."""

    def __init__(self, in_channels=4):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))
