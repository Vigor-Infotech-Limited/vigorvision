# vigorvision/models/vision/visionmodel.py

import torch
import torch.nn as nn

from vigorvision.nn.module.conv import ConvBlock
from vigorvision.nn.module.c3k2 import C3K2Block
import torch.nn.functional as F
from vigorvision.nn.module.c4k2 import C4K2Block
from vigorvision.nn.module.a2c2f import A2C2FBlock
from vigorvision.nn.module.vigorneck import VigorNeck
from vigorvision.nn.module.detectionhead import DetectionHead
from vigorvision.models.model_config import get_scaled_config as model_configs, model_variants


class VisionModel(nn.Module):
    def __init__(self, dataset, num_classes: int, variant: str = "vision-s"):
        super(VisionModel, self).__init__()

        assert variant in model_variants, f"Unknown model variant: {variant}"
        self.config = model_configs(variant, dataset)
        self.num_classes = num_classes
        


        self.backbone = self._build_backbone(self.config["backbone"])
        self.neck = VigorNeck([256, 512, 1024])
        self.head = DetectionHead(
            ch=[256, 512, 1024],
            nc=self.num_classes,
            anchors=self.config["head"]["anchors"],
            stride=self.config["head"]["strides"]
        )

        self.anchors = self.head.anchors
        self.stride = self.head.stride
        
        self.proj3 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1, groups=256),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        
        self.proj4 = nn.LazyConv2d(512, kernel_size=1)
        self.proj5 = nn.LazyConv2d(1024, kernel_size=1)


    def _build_backbone(self, cfg):
        layers = []
        for block_type, in_ch, out_ch, repeat, kwargs in cfg:
            for _ in range(repeat):
                if block_type == "conv":
                    layers.append(ConvBlock(in_ch, out_ch, **kwargs))
                elif block_type == "c3k2":
                    layers.append(C3K2Block(in_ch, out_ch, **kwargs))
                elif block_type == "c4k2":
                    layers.append(C4K2Block(in_ch, out_ch, **kwargs))
                elif block_type == "a2c2f":
                    layers.append(A2C2FBlock(in_ch, out_ch, **kwargs))
                else:
                    raise ValueError(f"Unsupported block: {block_type}")
                in_ch = out_ch
        return nn.Sequential(*layers)




    def forward(self, x, targets=None):
        feature_maps = []

        # collect all feature maps
        for layer in self.backbone:
            x = layer(x)
            feature_maps.append(x)

        # sort by spatial size (large to small)
        feature_maps = sorted(feature_maps, key=lambda f: f.shape[2], reverse=True)

        # pick top 3 distinct scales
        unique_maps = []
        seen = set()
        for f in feature_maps:
            if f.shape[2] not in seen:
                unique_maps.append(f)
                seen.add(f.shape[2])
            if len(unique_maps) == 3:
                break

        # pad with interpolations if fewer than 3
        while len(unique_maps) < 3:
            last = unique_maps[-1]
            unique_maps.append(F.adaptive_avg_pool2d(last, (last.shape[2] // 2, last.shape[3] // 2)))

        c3, c4, c5 = unique_maps[0], unique_maps[1], unique_maps[2]

        # normalize channels via pre-defined projections
        if c3.shape[1] != 256:
            c3 = self.proj3(c3)
        if c4.shape[1] != 512:
            c4 = self.proj4(c4)
        if c5.shape[1] != 1024:
            c5 = self.proj5(c5)

        # enforce proper downscale ratios
        if c4.shape[2] != c3.shape[2] // 2:
            c4 = F.interpolate(c4, size=(c3.shape[2] // 2, c3.shape[3] // 2), mode='nearest')
        if c5.shape[2] != c4.shape[2] // 2:
            c5 = F.interpolate(c5, size=(c4.shape[2] // 2, c4.shape[3] // 2), mode='nearest')

        # pass through neck and head
        p3, p4, p5 = self.neck(c3, c4, c5)

        if self.training:
            preds, loss = self.head([p3, p4, p5], targets)
            return preds, loss
        else:
            preds = self.head([p3, p4, p5])
            return preds