import torch
from tqdm import tqdm
from vigorvision.nn.loss import ComputeLoss
from vigorvision.utils.metrics import DetectionMetrics
from vigorvision.utils.general import increment_path, colorstr
from vigorvision.utils.decoder import PredictionDecoder
from vigorvision.utils.iou import nms as non_max_suppression


class Validator:
    def __init__(
        self,
        model,
        dataloader,
        device,
        anchors,
        num_classes,
        use_amp=True,
        conf_thres=0.25,
        iou_thres=0.6,
        max_det=100,
        save_dir=None,
        verbose=False,
    ):
        """
        Validator for evaluating model on val/test set.
        """
        self.model = model.to(device).eval()
        self.device = device
        self.dataloader = dataloader
        self.use_amp = use_amp
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.anchors = anchors
        self.num_classes = num_classes
        self.loss_fn = ComputeLoss(anchors=anchors, num_classes=num_classes, device=device)
        self.metrics = DetectionMetrics(num_classes=num_classes)
        self.save_dir = save_dir
        self.verbose = verbose
        self.pred_decoder = PredictionDecoder(
            conf_threshold=conf_thres,
            iou_threshold=iou_thres,
            max_detections=max_det,
            agnostic=False  # or True if you want class-agnostic NMS
        )
        self.strides = model.DetectionHead.strides
    @torch.no_grad()
    def evaluate(self):
        """
        Run validation for one full epoch.
        Returns:
            Dictionary of validation statistics
        """
        self.model.eval()
        self.metrics.reset()

        stats = []
        total_loss = torch.zeros(4, device=self.device)

        pbar = tqdm(self.dataloader, desc="Validating", leave=False)

        for batch in pbar:
            imgs, targets = batch
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                preds = self.model(imgs)
                loss_items = self.loss_fn(preds, targets)
                loss, box_loss, cls_loss, obj_loss = loss_items

            total_loss[0] += loss.item()
            total_loss[1] += box_loss.item()
            total_loss[2] += cls_loss.item()
            total_loss[3] += obj_loss.item()

            # Decode & NMS
            decoded_preds = self.pred_decoder(preds, anchors=self.anchors, strides=self.strides, num_classes=self.num_classes)

            for i in range(imgs.size(0)):
                pred = decoded_preds[i]  # already [x1, y1, x2, y2, score, cls]


                gt = targets[targets[:, 0] == i][:, 1:]  # [cls, x, y, w, h]
                self.metrics.update(preds=pred, targets=gt)

        # Aggregate metrics
        results = self.metrics.compute()
        mean_loss = total_loss / len(self.dataloader)

        val_stats = {
            "val/total_loss": mean_loss[0].item(),
            "val/box_loss": mean_loss[1].item(),
            "val/cls_loss": mean_loss[2].item(),
            "val/obj_loss": mean_loss[3].item(),
            "metrics/precision": results["precision"],
            "metrics/recall": results["recall"],
            "metrics/mAP_0.5": results["map_50"],
            "metrics/mAP_0.5:0.95": results["map_50_95"],
        }

        if self.verbose:
            print(colorstr("Validator Results:"))
            for k, v in val_stats.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return val_stats
