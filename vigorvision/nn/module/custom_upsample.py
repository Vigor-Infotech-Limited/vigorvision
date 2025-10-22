import torch
import torch.nn as nn
import torch.nn.functional as F
from vigorvision.nn.module.seblock import SEBlock


class CustomUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, mode='nearest', use_learnable=False):
        """
        Deep custom upsample block.
        Args:
            in_channels (int): Input feature channels.
            out_channels (int): Output channels after upsampling and fusion.
            scale (int): Upsampling scale factor.
            mode (str): Interpolation mode: 'nearest' or 'bilinear'.
            use_learnable (bool): If True, use transposed convolution.
        """
        super(CustomUpsample, self).__init__()
        self.scale = scale
        self.mode = mode
        self.use_learnable = use_learnable

        if use_learnable:
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=scale * 2, stride=scale,
                padding=scale // 2, output_padding=scale % 2
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode=mode, align_corners=False if mode == 'bilinear' else None),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )

        # Refinement path
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            SEBlock(out_channels)
        )

        # Residual attention
        self.residual_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x_up = self.upsample(x)
        x_refined = self.refine(x_up)
        fused = self.residual_fuse(x_refined + x_up)
        return fused
