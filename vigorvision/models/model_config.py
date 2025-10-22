# vigorvision/models/model_config.py
from vigorvision.utils import AutoAnchor as aa
"""
Unified model configuration for scalable Vision models.

Supports variants:
  - vision-n
  - vision-s
  - vision-m
  - vision-l
  - vision-x
"""



model_variants = {
    "vision-n": {"depth_mult": 0.33, "width_mult": 0.50, "base_channels": 32},
    "vision-s": {"depth_mult": 0.50, "width_mult": 0.75, "base_channels": 48},
    "vision-m": {"depth_mult": 0.75, "width_mult": 1.00, "base_channels": 64},
    "vision-l": {"depth_mult": 1.00, "width_mult": 1.00, "base_channels": 64},
    "vision-x": {"depth_mult": 1.33, "width_mult": 1.25, "base_channels": 80},
}


def make_divisible(x, divisor=8):
    return int((x + divisor / 2) // divisor * divisor)


def get_scaled_config(variant: str, dataset):
    assert variant in model_variants, f"Invalid model variant: {variant}"
    #Anchor_set = aa(dataset)
    #Anchor = Anchor_set.run()
    #print(Anchor)
    depth_mult = model_variants[variant]["depth_mult"]
    width_mult = model_variants[variant]["width_mult"]
    base = model_variants[variant]["base_channels"]

    def c(ch):  # scale and round channels
        return make_divisible(ch * width_mult)

    def d(repeats):  # scale and ceil depth
        return max(round(repeats * depth_mult), 1)

    return {
        "backbone": [
            # (block_type, in_ch, out_ch, repeat, kwargs)
            ("conv", 3, c(base), 1, {"kernel_size": 3, "stride": 1}),
            ("c3k2", c(base), c(base * 2), d(1), {"residual": True}),
            ("c4k2", c(base * 2), c(base * 4), d(1), {"residual": True}),
            ("a2c2f", c(base * 4), c(base * 8), d(1), {"residual": True}),
            ("a2c2f", c(base * 8), c(base * 16), d(1), {"residual": True}),
            ("a2c2f", c(base * 16), c(base * 32), d(1), {"residual": True}),
        ],
        "head": {
            "anchors": [[10,13], [16,30], [33,23],
                                    [30,61], [62,45], [59,119],
                                    [116,90], [156,198], [373,326]],
            "strides": [8, 16, 32]
        },
        "neck_channels": [c(base * 8), c(base * 16), c(base * 32)],  # P3, P4, P5
    }
