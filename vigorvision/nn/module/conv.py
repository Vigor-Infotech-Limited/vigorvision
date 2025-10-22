# vigorvision/nn/module/conv.py
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Flexible convolutional block with optional SE and residual.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        activation=True,
        residual=False,
        depthwise=False,
        use_se=False,
        dropout=0.0
    ):
        super().__init__()
        self.residual = residual and in_channels == out_channels and stride == 1
        self.use_se = use_se
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        if padding is None:
            padding = kernel_size // 2
        conv_groups = in_channels if depthwise else groups

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            groups=conv_groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU() if activation else nn.Identity()

        # Lazy import SEBlock to avoid circular import
        if self.use_se:
            
            self.se = SEBlock(out_channels)
            
    def forward(self, x):
        identity = x
        out = self.act(self.bn(self.conv(x)))
        if self.use_se:
            out = self.se(out)
        if self.dropout:
            out = self.dropout(out)
        if self.residual:
            out += identity
        return out
    
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

class SiLU(nn.Module):
    """SiLU Activation (Swish)."""
    def forward(self, x):
        return x * torch.sigmoid(x)

