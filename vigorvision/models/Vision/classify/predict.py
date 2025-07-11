# vigorvision  AGPL-3.0 License - https://vigorvision.com/license

import cv2
import torch
from PIL import Image

from vigorvision.data.augment import classify_transforms
from vigorvision.engine.predictor import BasePredictor
from vigorvision.engine.results import Results
from vigorvision.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.

    This predictor handles the specific requirements of classification models, including preprocessing images
    and postprocessing predictions to generate classification results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        _legacy_transform_name (str): Name of the legacy transform class for backward compatibility.

    Methods:
        preprocess: Convert input images to model-compatible format.
        postprocess: Process model predictions into Results objects.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Examples:
        >>> from vigorvision.utils import ASSETS
        >>> from vigorvision.models.Vision.classify import ClassificationPredictor
        >>> args = dict(model="Vision11n-cls.pt", source=ASSETS)
        >>> predictor = ClassificationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the ClassificationPredictor with the specified configuration and set task to 'classify'.

        This constructor initializes a ClassificationPredictor instance, which extends BasePredictor for classification
        tasks. It ensures the task is set to 'classify' regardless of input configuration.

        Args:
            cfg (dict): Default configuration dictionary containing prediction settings. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be executed during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self._legacy_transform_name = "vigorvision.Vision.data.augment.ToTensor"

    def setup_source(self, source):
        """Sets up source and inference mode and classify transforms."""
        super().setup_source(source)
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms")
            else True
        )
        self.transforms = self.model.model.transforms if not updated else classify_transforms(self.imgsz)

    def preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """
        Process predictions to return Results objects with classification probabilities.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Input images after preprocessing.
            orig_imgs (List[np.ndarray] | torch.Tensor): Original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing classification results for each image.
        """
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
