# vigorvision/models/build.py

import torch.nn as nn
from vigorvision.models.vision.visionmodel import VisionModel


def build_model(dataset, num_classes: int = 11, variant: str = "vision-s") -> nn.Module:
    """
    Build the Vision model based on the given variant.

    Args:
        num_classes (int): Number of output classes.
        variant (str): Model variant, one of ['vision-n', 'vision-s', 'vision-m', 'vision-l', 'vision-x'].

    Returns:
        nn.Module: Instantiated VisionModel.
    """
    model = VisionModel(dataset, num_classes=num_classes, variant=variant)
    return model


def test_build_model():
    """
    Test function for verifying the Vision model builder.
    """
    import torch

    for variant in ["vision-n", "vision-s", "vision-m", "vision-l", "vision-x"]:
        print(f"\n[TEST] Building {variant} model:")
        model = build_model(num_classes=11, variant=variant)
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        if isinstance(output, tuple):
            output = output[0]
        print(f" - Output shape: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")


if __name__ == "__main__":
    test_build_model()
