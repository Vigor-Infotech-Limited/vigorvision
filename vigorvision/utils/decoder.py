import torch
import torch.nn.functional as F


class PredictionDecoder:
    """
    Decodes raw model outputs into usable predictions.
    Applies confidence thresholding, NMS, and box format conversion.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        multi_label: bool = False,
        agnostic: bool = False,
        top_k_per_class: int = 100,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.multi_label = multi_label
        self.agnostic = agnostic
        self.top_k_per_class = top_k_per_class

    
    def __call__(self, predictions, anchors, strides, num_classes):
        """
        predictions: List of tensors [batch, anchors * (num_classes + 5), H, W] from all scales
        anchors: List of anchor box tensors for each scale
        strides: List of strides for each scale
        Returns: List of detections per image [x1, y1, x2, y2, conf, cls]
        """
        with torch.no_grad():
            device = predictions[0].device
            decoded_outputs = []

            for i, pred in enumerate(predictions):
                bs, _, h, w = pred.shape
                na = anchors[i].shape[0]
                nc = num_classes
                pred = pred.view(bs, na, nc + 5, h, w).permute(0, 1, 3, 4, 2).contiguous()

                # Grid
                yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
                grid = torch.stack((xv, yv), 2).to(device).float()
                grid = grid.view(1, 1, h, w, 2)

                # Decode boxes
                pred_boxes = torch.zeros_like(pred[..., :4])
                pred_boxes[..., 0:2] = (pred[..., 0:2].sigmoid() + grid) * strides[i]  # cx, cy
                pred_boxes[..., 2:4] = (pred[..., 2:4].sigmoid() * 2) ** 2 * anchors[i].view(1, na, 1, 1, 2)  # w, h

                # Objectness + class confidence
                objectness = pred[..., 4:5].sigmoid()
                class_scores = pred[..., 5:].sigmoid()
                conf = objectness * class_scores  # [B, na, H, W, C]

                decoded_outputs.append((pred_boxes.view(bs, -1, 4), conf.view(bs, -1, nc)))

            return self._postprocess(decoded_outputs, num_classes)

    def _postprocess(self, decoded, num_classes):
        """
        Apply confidence thresholding and NMS.
        """
        from vigorvision.utils.box_ops import nms
        results = []
        for boxes, conf in zip(*decoded):
            scores, labels = conf.max(dim=-1)
            mask = scores > self.conf_threshold
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

            if boxes.numel() == 0:
                results.append(torch.zeros((0, 6), device=boxes.device))
                continue

            # Apply class-wise NMS if not class-agnostic
            if not self.agnostic:
                keep = []
                for cls in range(num_classes):
                    cls_mask = labels == cls
                    if cls_mask.sum() == 0:
                        continue
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    cls_keep = nms(cls_boxes, cls_scores, self.iou_threshold)
                    keep.append(cls_mask.nonzero()[cls_keep].squeeze(1))
                keep = torch.cat(keep) if keep else torch.tensor([], dtype=torch.long, device=boxes.device)
            else:
                keep = nms(boxes, scores, self.iou_threshold)

            keep = keep[:self.max_detections]
            results.append(torch.cat([boxes[keep], scores[keep].unsqueeze(1), labels[keep].unsqueeze(1).float()], dim=1))

        return results  # List of [N, 6]: x1, y1, x2, y2, score, cls
