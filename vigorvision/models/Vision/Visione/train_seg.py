# vigorvision  AGPL-3.0 License - https://vigorvision.com/license

from copy import copy, deepcopy

from vigorvision.models.Vision.segment import SegmentationTrainer
from vigorvision.nn.tasks import VisionESegModel
from vigorvision.utils import RANK

from .train import VisionETrainer, VisionETrainerFromScratch, VisionEVPTrainer
from .val import VisionESegValidator


class VisionESegTrainer(VisionETrainer, SegmentationTrainer):
    """
    Trainer class for VisionE segmentation models.

    This class combines VisionETrainer and SegmentationTrainer to provide training functionality
    specifically for VisionE segmentation models.

    Attributes:
        cfg (dict): Configuration dictionary with training parameters.
        overrides (dict): Dictionary with parameter overrides.
        _callbacks (list): List of callback functions for training events.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return VisionESegModel initialized with specified config and weights.

        Args:
            cfg (dict | str): Model configuration dictionary or YAML file path.
            weights (str, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (VisionESegModel): Initialized VisionE segmentation model.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = VisionESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """
        Create and return a validator for VisionE segmentation model evaluation.

        Returns:
            (VisionESegValidator): Validator for VisionE segmentation models.
        """
        self.loss_names = "box", "seg", "cls", "dfl"
        return VisionESegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class VisionEPESegTrainer(SegmentationTrainer):
    """
    Fine-tune VisionESeg model in linear probing way.

    This trainer specializes in fine-tuning VisionESeg models using a linear probing approach, which involves freezing
    most of the model and only training specific layers.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return VisionESegModel initialized with specified config and weights for linear probing.

        Args:
            cfg (dict | str): Model configuration dictionary or YAML file path.
            weights (str, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (VisionESegModel): Initialized VisionE segmentation model configured for linear probing.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = VisionESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # NOTE: `get_text_pe` related to text model and VisionEDetect.reprta,
        # it'd get correct results as long as loading proper pretrained weights.
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model


class VisionESegTrainerFromScratch(VisionETrainerFromScratch, VisionESegTrainer):
    """Trainer for VisionE segmentation from scratch."""

    pass


class VisionESegVPTrainer(VisionEVPTrainer, VisionESegTrainerFromScratch):
    """Trainer for VisionE segmentation with VP."""

    pass
