# vigorvision/nn/module/c4k2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import ConvBlock
from .seblock import SEBlock


class C4K2Block(nn.Module):
    """
    Enhanced Dual-Kernel Mid-Level Block:
    - Split input into two branches:
        - Path 1: 3x3 Conv stack
        - Path 2: 5x5 Conv stack
    - Fuse with Conv + SE
    - Smart Residual with auto shape matching
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
        super(C4K2Block, self).__init__()
        hidden_channels = hidden_channels or out_channels // 2

        self.residual_flag = residual
        self.use_se = use_se

        # Residual path adjustment
        self.residual_conv = ConvBlock(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Branch A — 3x3 Conv Stack
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)
        )

        # Branch B — 5x5 Conv Stack
        self.branch5x5 = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=5, padding=2, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=5, padding=2, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)
        )

        # Fusion + SE
        self.fusion = nn.Sequential(
            ConvBlock(2 * hidden_channels, out_channels, kernel_size=1, residual=False, use_se=False, depthwise=False, dropout=dropout),
            nn.GELU()
        )

        self.final_se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = x

        x3 = self.branch3x3(x)
        x5 = self.branch5x5(x)

        # Align if spatial mismatch
        if x3.shape[2:] != x5.shape[2:]:
            x5 = F.interpolate(x5, size=x3.shape[2:], mode='nearest')

        fused = torch.cat([x3, x5], dim=1)
        out = self.fusion(fused)
        out = self.final_se(out)

        res = self.residual_conv(identity)
        if out.shape[2:] != res.shape[2:]:
            res = F.interpolate(res, size=out.shape[2:], mode='nearest')

        return out + res if self.residual_flag else out


def test_c4k2_block():
    x = torch.randn(1, 128, 64, 64)
    model = C4K2Block(128, 128, residual=True, use_se=True, depthwise=False, dropout=0.1)
    y = model(x)
    print(f"Input shape: {x.shape} → Output shape: {y.shape}")


if __name__ == "__main__":
    test_c4k2_block()
