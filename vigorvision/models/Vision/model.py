# vigorvision  AGPL-3.0 License - https://vigorvision.com/license

from pathlib import Path

from vigorvision.data.build import load_inference_source
from vigorvision.engine.model import Model
from vigorvision.models import Vision
from vigorvision.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    VisionEModel,
    VisionESegModel,
)
from vigorvision.utils import ROOT, YAML


class Vision(Model):
    """Vision (You Only Look Once) object detection model."""

    def __init__(self, model="Vision11n.pt", task=None, verbose=False):
        """
        Initialize a Vision model.

        This constructor initializes a Vision model, automatically switching to specialized model types
        (VisionWorld or VisionE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'Vision11n.pt', 'Vision11n.yaml'.
            task (str | None): Vision task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'.
                Defaults to auto-detection based on model.
            verbose (bool): Display model info on load.

        Examples:
            >>> from vigorvision import Vision
            >>> model = Vision("Vision11n.pt")  # load a pretrained Visionv11n detection model
            >>> model = Vision("Vision11n-seg.pt")  # load a pretrained Vision11n segmentation model
        """
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if VisionWorld PyTorch model
            new_instance = VisionWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "Visione" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if VisionE PyTorch model
            new_instance = VisionE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default Vision initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": Vision.classify.ClassificationTrainer,
                "validator": Vision.classify.ClassificationValidator,
                "predictor": Vision.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": Vision.detect.DetectionTrainer,
                "validator": Vision.detect.DetectionValidator,
                "predictor": Vision.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": Vision.segment.SegmentationTrainer,
                "validator": Vision.segment.SegmentationValidator,
                "predictor": Vision.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": Vision.pose.PoseTrainer,
                "validator": Vision.pose.PoseValidator,
                "predictor": Vision.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": Vision.obb.OBBTrainer,
                "validator": Vision.obb.OBBValidator,
                "predictor": Vision.obb.OBBPredictor,
            },
        }


class VisionWorld(Model):
    """Vision-World object detection model."""

    def __init__(self, model="Visionv8s-world.pt", verbose=False) -> None:
        """
        Initialize Visionv8-World model with a pre-trained model file.

        Loads a Visionv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": Vision.detect.DetectionValidator,
                "predictor": Vision.detect.DetectionPredictor,
                "trainer": Vision.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class VisionE(Model):
    """VisionE object detection and segmentation model."""

    def __init__(self, model="Visione-11s-seg.pt", task=None, verbose=False) -> None:
        """
        Initialize VisionE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": VisionEModel,
                "validator": Vision.Visione.VisionEDetectValidator,
                "predictor": Vision.detect.DetectionPredictor,
                "trainer": Vision.Visione.VisionETrainer,
            },
            "segment": {
                "model": VisionESegModel,
                "validator": Vision.Visione.VisionESegValidator,
                "predictor": Vision.segment.SegmentationPredictor,
                "trainer": Vision.Visione.VisionESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, VisionEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires
        that the model is an instance of VisionEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = VisionE("Visione-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = model.model.backbone(img)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, VisionEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary and class names for the VisionE model.

        This method configures the vocabulary and class names used by the model for text processing and
        classification tasks. The model must be an instance of VisionEModel.

        Args:
            vocab (list): Vocabulary list containing tokens or words used by the model for text processing.
            names (list): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of VisionEModel.

        Examples:
            >>> model = VisionE("Visione-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, VisionEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, VisionEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        """
        Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
        """
        assert isinstance(self.model, VisionEModel)
        self.model.set_classes(classes, embeddings)
        # Verify no background class is present
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp=False,
        refer_data=None,
        **kwargs,
    ):
        """
        Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths,
                directory paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a
                generator as they are computed.
            visual_prompts (dict): Dictionary containing visual prompts for the model. Must include 'bboxes' and
                'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically
                loaded based on the task.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = VisionE("Visione-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
        self.predictor = (predictor or self._smart_load("predictor"))(
            overrides={
                "task": self.model.task,
                "mode": "predict",
                "save": False,
                "verbose": refer_image is None,
                "batch": 1,
            },
            _callbacks=self.callbacks,
        )

        if len(visual_prompts):
            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())

        self.predictor.setup_model(model=self.model)

        if refer_image is None and source is not None:
            dataset = load_inference_source(source)
            if dataset.mode in {"video", "stream"}:
                # NOTE: set the first frame as refer image for videos/streams inference
                refer_image = next(iter(dataset))[1][0]
        if refer_image is not None and len(visual_prompts):
            vpe = self.predictor.get_vpe(refer_image)
            self.model.set_classes(self.model.names, vpe)
            self.task = "segment" if isinstance(self.predictor, Vision.segment.SegmentationPredictor) else "detect"
            self.predictor = None  # reset predictor

        return super().predict(source, stream, **kwargs)
