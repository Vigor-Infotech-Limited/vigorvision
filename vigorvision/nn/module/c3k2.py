# vigorvision/nn/module/c3k2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import ConvBlock
from .seblock import SEBlock


class C3K2Block(nn.Module):
    """
    Enhanced Dual Kernel Convolution Block with:
    - Dual Path Conv
    - Cross Mixing + SE Attention
    - Smart Residual with shape matching
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        residual=True,
        use_se=True,
        depthwise=False,
        dropout=0.0
    ):
        super(C3K2Block, self).__init__()
        hidden_channels = hidden_channels or out_channels // 2

        self.residual_flag = residual
        self.use_se = use_se

        # Residual Pathway (adjust if needed)
        self.residual_conv = ConvBlock(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Dual Branch Convs
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)
        self.cv2 = ConvBlock(in_channels, hidden_channels, kernel_size=1, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)

        # Cross Mixing Convs
        self.cross_conv = nn.Sequential(
            ConvBlock(2 * hidden_channels, 2 * hidden_channels, kernel_size=1, residual=False, use_se=use_se, depthwise=False, dropout=dropout),
            nn.GELU()
        )

        # Final Output Conv
        self.out_conv = ConvBlock(2 * hidden_channels, out_channels, kernel_size=1, residual=False, use_se=False, depthwise=False, dropout=dropout)

        # Final SE Attention
        self.final_se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = x

        x1 = self.cv1(x)
        x2 = self.cv2(x)

        # Spatial align if mismatch
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')

        merged = torch.cat([x1, x2], dim=1)
        mixed = self.cross_conv(merged)
        out = self.out_conv(mixed)
        out = self.final_se(out)

        # Residual with shape check
        res = self.residual_conv(identity)
        if out.shape[2:] != res.shape[2:]:
            res = F.interpolate(res, size=out.shape[2:], mode='nearest')

        return out + res if self.residual_flag else out


def test_c3k2_block():
    x = torch.randn(1, 64, 128, 128)
    model = C3K2Block(64, 64, residual=True, use_se=True, depthwise=True, dropout=0.1)
    y = model(x)
    print(f"Input shape: {x.shape} → Output shape: {y.shape}")


if __name__ == "__main__":
    test_c3k2_block()
