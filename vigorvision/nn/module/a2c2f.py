# vigorvision/nn/module/a2c2f.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import ConvBlock
from .seblock import SEBlock


class A2C2FBlock(nn.Module):
    """
    Hybrid Attention-Augmented Cascaded Convolution Fusion Block:
    - Multi-Branch Conv Mixer with increasing receptive fields
    - Sequential Depthwise Conv Stack after fusion
    - Final SE Attention + Smart Residual
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        num_deep_layers=2,
        residual=True,
        use_se=True,
        depthwise=False,
        dropout=0.0
    ):
        super(A2C2FBlock, self).__init__()
        hidden_channels = hidden_channels or out_channels // 2

        self.residual_flag = residual
        self.use_se = use_se

        self.residual_conv = ConvBlock(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # --- Multi-Branch ---
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)
        )

        self.branch5x5 = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=5, padding=2, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=5, padding=2, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)
        )

        self.branch7x7 = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=7, padding=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=7, padding=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout)
        )

        # --- Fusion ---
        self.fusion = nn.Sequential(
            ConvBlock(3 * hidden_channels, out_channels, kernel_size=1, residual=False, use_se=False, depthwise=False, dropout=dropout),
            nn.GELU()
        )

        # --- Deep Sequential Conv Stack ---
        self.deep_conv_stack = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(out_channels, hidden_channels, kernel_size=1, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
                ConvBlock(hidden_channels, hidden_channels, kernel_size=3, residual=False, use_se=use_se, depthwise=depthwise, dropout=dropout),
                SEBlock(hidden_channels) if use_se else nn.Identity(),
                ConvBlock(hidden_channels, out_channels, kernel_size=1, residual=False, use_se=False, depthwise=False, dropout=dropout)
            )
            for _ in range(num_deep_layers)
        ])

        self.final_se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = x

        x3 = self.branch3x3(x)
        x5 = self.branch5x5(x)
        x7 = self.branch7x7(x)

        # Align spatial if needed
        size = x3.shape[2:]
        if x5.shape[2:] != size:
            x5 = F.interpolate(x5, size=size, mode='nearest')
        if x7.shape[2:] != size:
            x7 = F.interpolate(x7, size=size, mode='nearest')

        fused = torch.cat([x3, x5, x7], dim=1)
        out = self.fusion(fused)
        out = self.deep_conv_stack(out)
        out = self.final_se(out)

        res = self.residual_conv(identity)
        if out.shape[2:] != res.shape[2:]:
            res = F.interpolate(res, size=out.shape[2:], mode='nearest')

        return out + res if self.residual_flag else out


def test_a2c2f_block():
    x = torch.randn(1, 256, 32, 32)
    model = A2C2FBlock(256, 256, num_deep_layers=2, residual=True, use_se=True, depthwise=False, dropout=0.1)
    y = model(x)
    print(f"Input shape: {x.shape} → Output shape: {y.shape}")


if __name__ == "__main__":
    test_a2c2f_block()
