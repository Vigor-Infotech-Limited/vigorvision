# vigorvision/nn/module/seblock.py

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    Applies global average pooling, bottleneck FC layers, and channel-wise scaling.

    Args:
        channels (int): Number of input/output channels
        reduction (int): Reduction ratio for bottleneck, default = 16
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale  # channel-wise scaling
