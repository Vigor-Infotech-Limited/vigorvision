# vigorvision/nn/module/vigorneck.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .a2c2f import A2C2FBlock
from .c4k2 import C4K2Block
from .c3k2 import C3K2Block
from .seblock import SEBlock
from .custom_upsample import CustomUpsample  # Assuming you have this defined
from .conv import ConvBlock


class VigorNeck(nn.Module):
    """
    VigorNeck:
    - PANet style deep multi-scale feature aggregation
    - A2C2F + C4K2 + C3K2 modules for feature refinement
    - Smart upsample/downsample with SE
    """

    def __init__(self, channels_list, use_se=True, depthwise=False, dropout=0.0):
        """
        channels_list: List of [C3, C4, C5] channel sizes from backbone
        """
        super(VigorNeck, self).__init__()

        assert len(channels_list) == 3, "channels_list must be [C3, C4, C5]"

        C3, C4, C5 = channels_list

        self.use_se = use_se

        # Reduce & refine top C5
        self.reduce_c5 = ConvBlock(C5, C4, kernel_size=1)
        self.a2c2f_c5 = A2C2FBlock(C4, C4, residual=True, use_se=use_se, depthwise=depthwise, dropout=dropout)

        # Upsample & merge with C4
        # Upsample & merge with C4
        self.upsample_c5 = CustomUpsample(C4, C4, scale=2, mode='nearest')
        self.c4_merge = C3K2Block(C4 + C4, C4, residual=True, use_se=use_se, depthwise=depthwise, dropout=dropout)

        # Upsample & merge with C3
        self.upsample_c4 = CustomUpsample(C4, C3, scale=2, mode='nearest')
        self.c3_merge = C4K2Block(C4 + C3, C3, residual=True, use_se=use_se, depthwise=depthwise, dropout=dropout)

        # Downsample fusion path for detection heads
        self.down_c3 = ConvBlock(C3, C4, kernel_size=3, stride=2)
        self.c4_out = C3K2Block(C4 + C4, C4, residual=True, use_se=use_se, depthwise=depthwise, dropout=dropout)

        self.down_c4 = ConvBlock(C4, C5, kernel_size=3, stride=2)
        self.c5_out = C4K2Block(C5 + C5, C5, residual=True, use_se=use_se, depthwise=depthwise, dropout=dropout)

        # Final SE Attention on all outputs
        self.se_c3 = SEBlock(C3) if use_se else nn.Identity()
        self.se_c4 = SEBlock(C4) if use_se else nn.Identity()
        self.se_c5 = SEBlock(C5) if use_se else nn.Identity()

    def forward(self, c3, c4, c5):
        # Top-Down Path
        p5 = self.reduce_c5(c5)
        p5 = self.a2c2f_c5(p5)
        p5_up = self.upsample_c5(p5)

        p4 = torch.cat([p5_up, c4], dim=1)
        p4 = self.c4_merge(p4)
        p4_up = self.upsample_c4(p4)

        p3 = torch.cat([p4_up, c3], dim=1)
        p3 = self.c3_merge(p3)

        # Bottom-Up Path
        p3_down = self.down_c3(p3)
        p4_out = torch.cat([p3_down, p4], dim=1)
        p4_out = self.c4_out(p4_out)

        p4_down = self.down_c4(p4_out)
        p5_out = torch.cat([p4_down, p5], dim=1)
        p5_out = self.c5_out(p5_out)

        # Final SE Attention
        return self.se_c3(p3), self.se_c4(p4_out), self.se_c5(p5_out)


def test_vigorneck():
    c3 = torch.randn(1, 256, 80, 80)
    c4 = torch.randn(1, 512, 40, 40)
    c5 = torch.randn(1, 1024, 20, 20)
    model = VigorNeck([256, 512, 1024], use_se=True, depthwise=False, dropout=0.1)
    p3, p4, p5 = model(c3, c4, c5)
    print(f"P3 shape: {p3.shape} | P4 shape: {p4.shape} | P5 shape: {p5.shape}")


if __name__ == "__main__":
    test_vigorneck()
