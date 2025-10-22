import torch
from tqdm import tqdm
from vigorvision.utils.decoder import PredictionDecoder
from vigorvision.utils.metrics import evaluate_predictions
from vigorvision.utils.general import xywh2xyxy


class Evaluator:
    """
    Evaluator class for model inference and evaluation on test set.
    Fully compatible with the upgraded beast-level PredictionDecoder.
    """

    def __init__(self, model, dataloader, device, iou_threshold=0.5, conf_threshold=0.25, class_names=None):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.class_names = class_names

        # Initialize decoder with beast-level settings
        self.decoder = PredictionDecoder(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=300,
            multi_label=False,      # Change to True if you want multi-label eval
            agnostic=False,         # Change to True for class-agnostic NMS
            top_k_per_class=100
        )

    @torch.no_grad()
    def evaluate(self):
        """
        Run evaluation on the test dataloader and return metrics.
        """
        all_preds = []
        all_targets = []

        pbar = tqdm(self.dataloader, desc="[Evaluator] Running Evaluation", leave=False)

        for batch in pbar:
            imgs, targets = batch[0].to(self.device), batch[1]

            # Model forward pass
            outputs = self.model(imgs)

            # Ensure model outputs predictions, anchors, strides
            # If your model returns tuple: (predictions, anchors, strides)
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                predictions, anchors, strides = outputs[0], outputs[1], outputs[2]
            else:
                raise ValueError(
                    "Model must return (predictions, anchors, strides[, ...]) for the decoder to work."
                )

            num_classes = getattr(self.model, "num_classes", None)
            if num_classes is None:
                raise AttributeError("Model must have a 'num_classes' attribute for evaluation.")

            # Decode raw predictions
            decoded_batch = self.decoder(predictions, anchors, strides, num_classes)

            # Format predictions & targets for metrics
            for i, pred in enumerate(decoded_batch):
                if pred.shape[0] > 0:
                    all_preds.append({
                        "boxes": pred[:, :4].cpu(),
                        "scores": pred[:, 4].cpu(),
                        "labels": pred[:, 5].int().cpu()
                    })
                else:
                    all_preds.append({
                        "boxes": torch.empty((0, 4)),
                        "scores": torch.tensor([]),
                        "labels": torch.tensor([])
                    })

                if targets is not None:
                    t = targets[i]
                    all_targets.append({
                        "boxes": xywh2xyxy(t[:, :4]).cpu(),
                        "labels": t[:, 4].int().cpu()
                    })

        # Compute evaluation metrics
        metrics = evaluate_predictions(all_preds, all_targets, class_names=self.class_names)
        return metrics
