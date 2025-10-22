# vigorvision/utils/metrics.py

import torch
import numpy as np
from collections import defaultdict
from .iou import box_iou
from vigorvision.utils.iou import box_iou

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


class AverageMeter:
    """
    Tracks and updates the average, sum, and count for metrics.
    Can be used for loss, accuracy, precision, recall, etc.
    """

    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets all counters."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Updates with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Computes Average Precision (AP) given recall and precision curves.
    Implements the VOC2007 11-point metric or integration-based AP.

    Args:
        recall (np.ndarray): recall curve
        precision (np.ndarray): precision curve
    Returns:
        float: AP value
    """
    # Append sentinel values at the ends
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def compute_map(
    preds: List[Dict],
    targets: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = None
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Computes mAP over multiple IoU thresholds like COCO and per-class mAP.
    
    Args:
        preds (List[Dict]): list of prediction dicts 
            {'boxes': Tensor[N,4], 'scores': Tensor[N], 'labels': Tensor[N]}
        targets (List[Dict]): list of target dicts 
            {'boxes': Tensor[M,4], 'labels': Tensor[M]}
        num_classes (int): number of classes
        iou_thresholds (List[float], optional): IoU thresholds to evaluate. Defaults to 0.5:0.95 step 0.05.
    
    Returns:
        Tuple[Dict[str, float], Dict[str, float], float]: 
            (mAP per IoU threshold, mAP per class, mean mAP)
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)

    ap_per_threshold = {}
    ap_per_class = np.zeros(num_classes, dtype=np.float32)

    # Track per-class AP over all thresholds
    per_class_ap_lists = [[] for _ in range(num_classes)]

    for thr in iou_thresholds:
        tp = [[] for _ in range(num_classes)]
        fp = [[] for _ in range(num_classes)]
        scores = [[] for _ in range(num_classes)]
        num_gts = [0] * num_classes

        for pred, target in zip(preds, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']

            gt_boxes = target['boxes']
            gt_labels = target['labels']

            for c in range(num_classes):
                pred_mask = pred_labels == c
                gt_mask = gt_labels == c

                pred_boxes_c = pred_boxes[pred_mask]
                pred_scores_c = pred_scores[pred_mask]
                gt_boxes_c = gt_boxes[gt_mask]

                num_gts[c] += gt_boxes_c.size(0)

                if pred_boxes_c.size(0) == 0:
                    continue

                order = torch.argsort(pred_scores_c, descending=True)
                pred_boxes_c = pred_boxes_c[order]
                pred_scores_c = pred_scores_c[order]

                matched = torch.zeros(gt_boxes_c.size(0), dtype=torch.bool)

                for pb, score in zip(pred_boxes_c, pred_scores_c):
                    if gt_boxes_c.size(0) == 0:
                        fp[c].append(1)
                        tp[c].append(0)
                        scores[c].append(score.item())
                        continue

                    ious = box_iou(pb.unsqueeze(0), gt_boxes_c).squeeze(0)
                    best_iou, best_idx = ious.max(0)

                    if best_iou >= thr and not matched[best_idx]:
                        tp[c].append(1)
                        fp[c].append(0)
                        matched[best_idx] = True
                    else:
                        tp[c].append(0)
                        fp[c].append(1)

                    scores[c].append(score.item())

        ap_list = []
        for c in range(num_classes):
            if num_gts[c] == 0:
                ap_list.append(0.0)
                continue

            scores_c = np.array(scores[c])
            tp_c = np.array(tp[c])
            fp_c = np.array(fp[c])

            if scores_c.shape[0] == 0:
                ap_list.append(0.0)
                continue

            indices = np.argsort(-scores_c)
            tp_c = np.cumsum(tp_c[indices])
            fp_c = np.cumsum(fp_c[indices])

            recall = tp_c / (num_gts[c] + 1e-6)
            precision = tp_c / (tp_c + fp_c + 1e-6)

            ap = compute_ap(recall, precision)
            ap_list.append(ap)

            # Append for per-class averaging across thresholds
            per_class_ap_lists[c].append(ap)

        ap_per_threshold[f"mAP@{thr:.2f}"] = np.mean(ap_list)

    # Compute mean AP per class across thresholds
    for c in range(num_classes):
        if len(per_class_ap_lists[c]) > 0:
            ap_per_class[c] = np.mean(per_class_ap_lists[c])

    mean_map = np.mean(list(ap_per_threshold.values()))

    # Make class mAP dict
    ap_per_class_dict = {f"class_{c}": ap_per_class[c] for c in range(num_classes)}

    return mean_map, ap_per_class_dict, ap_per_threshold

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Computes IoU between two sets of boxes."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2 - inter_area
    return inter_area / (union_area + 1e-6)

def evaluate_predictions(preds, targets, class_names=None, iou_thresholds=None):
    """
    Compute detailed detection metrics (precision, recall, mAP) per class.
    Compatible with Evaluator output.
    
    Args:
        preds (list of dict): [{"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}, ...]
        targets (list of dict): [{"boxes": Tensor[M,4], "labels": Tensor[M]}, ...]
        class_names (list[str]): Optional list of class names.
        iou_thresholds (list[float]): List of IoU thresholds for mAP computation. 
                                      Default is np.arange(0.5, 0.96, 0.05) like COCO.
    Returns:
        dict: Metrics summary
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)

    device = preds[0]["boxes"].device if preds and isinstance(preds[0]["boxes"], torch.Tensor) else torch.device("cpu")

    num_classes = int(max(
        [p["labels"].max().item() if p["labels"].numel() else 0 for p in preds] +
        [t["labels"].max().item() if t["labels"].numel() else 0 for t in targets]
    )) + 1

    # Storage for stats
    tp_per_class = {t: np.zeros(num_classes) for t in iou_thresholds}
    fp_per_class = {t: np.zeros(num_classes) for t in iou_thresholds}
    fn_per_class = {t: np.zeros(num_classes) for t in iou_thresholds}
    scores_per_class = defaultdict(list)

    for pred, tgt in zip(preds, targets):
        pred_boxes = pred["boxes"].to(device)
        pred_scores = pred["scores"].to(device)
        pred_labels = pred["labels"].to(device)
        tgt_boxes = tgt["boxes"].to(device)
        tgt_labels = tgt["labels"].to(device)

        if tgt_boxes.numel() == 0 and pred_boxes.numel() == 0:
            continue

        for t in iou_thresholds:
            matched_gt = set()

            for i, (pbox, plabel, pscore) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                cls = int(plabel)
                scores_per_class[cls].append(pscore.item())

                # Find gt of same class
                gt_mask = tgt_labels == cls
                if not gt_mask.any():
                    fp_per_class[t][cls] += 1
                    continue

                gt_boxes_cls = tgt_boxes[gt_mask]
                ious = box_iou(pbox.unsqueeze(0), gt_boxes_cls)[0]
                max_iou, max_idx = ious.max(0)

                if max_iou >= t and max_idx.item() not in matched_gt:
                    tp_per_class[t][cls] += 1
                    matched_gt.add(max_idx.item())
                else:
                    fp_per_class[t][cls] += 1

            # FN = GT not matched
            for cls in tgt_labels.unique():
                cls = int(cls)
                total_gt_cls = (tgt_labels == cls).sum().item()
                matched_gt_cls = tp_per_class[t][cls]
                fn_per_class[t][cls] += max(total_gt_cls - matched_gt_cls, 0)

    # Compute per-class metrics
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    map_per_iou = []

    for cls in range(num_classes):
        if class_names:
            cname = class_names[cls]
        else:
            cname = f"class_{cls}"

        precisions, recalls = [], []

        for t in iou_thresholds:
            tp = tp_per_class[t][cls]
            fp = fp_per_class[t][cls]
            fn = fn_per_class[t][cls]

            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            precisions.append(precision)
            recalls.append(recall)

        # Store last IoU threshold metrics (0.5)
        precision_per_class[cname] = precisions[0]
        recall_per_class[cname] = recalls[0]
        f1_per_class[cname] = 2 * precision_per_class[cname] * recall_per_class[cname] / (
            precision_per_class[cname] + recall_per_class[cname] + 1e-16
        )

        # mAP per class = average precision over IoU thresholds
        ap_cls = np.mean(precisions)
        map_per_iou.append(ap_cls)

    # Macro averages
    mAP_50 = np.mean([precision_per_class[c] for c in precision_per_class])
    mAP_50_95 = np.mean(map_per_iou)

    metrics = {
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "mAP@0.5": mAP_50,
        "mAP@[0.5:0.95]": mAP_50_95,
        "macro_precision": np.mean(list(precision_per_class.values())),
        "macro_recall": np.mean(list(recall_per_class.values())),
        "macro_f1": np.mean(list(f1_per_class.values()))
    }

    return metrics

def compute_ap(recall, precision):
    """Compute Average Precision (AP) given precision and recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under PR curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])


def ap_per_class(tp, conf, pred_cls, target_cls, iou_threshold=0.5, class_names=None):
    """Compute precision, recall, AP, F1 for each class."""
    if len(tp) == 0:
        return [], [], [], [], [], []

    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    ap, precision, recall, f1, p, r = [], [], [], [], [], []

    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_gt == 0 and n_p == 0:
            continue
        elif n_gt == 0 or n_p == 0:
            ap.append(0.0)
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)
            p.append(0.0)
            r.append(0.0)
            continue

        fpc = (1 - tp[i]).astype(np.float32)
        tpc = tp[i].astype(np.float32)

        fpc = np.cumsum(fpc)
        tpc = np.cumsum(tpc)

        recall_curve = tpc / (n_gt + 1e-16)
        precision_curve = tpc / (tpc + fpc)

        ap_class = compute_ap(recall_curve, precision_curve)

        p_c = precision_curve[-1] if precision_curve.size else 0.0
        r_c = recall_curve[-1] if recall_curve.size else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c + 1e-16)

        ap.append(ap_class)
        precision.append(p_c)
        recall.append(r_c)
        f1.append(f1_c)
        p.append(p_c)
        r.append(r_c)

    return np.array(precision), np.array(recall), np.array(ap), np.array(f1), unique_classes, np.stack((p, r, ap), axis=1)


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
        self.num_classes = num_classes

    def process_batch(self, detections, labels):
        """
        Update confusion matrix using detections and ground truths.
        detections: (nx6) [x1, y1, x2, y2, conf, cls]
        labels: (mx5) [cls, x1, y1, x2, y2]
        """
        gt_classes = labels[:, 0].int()
        pred_classes = detections[:, 5].int()
        ious = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(ious > 0.5)

        if x[0].numel():
            matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]].unsqueeze(1)), 1)
            if x[0].numel() > 1:
                matches = matches[matches[:, 2].argsort(descending=True)]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = torch.zeros((0, 3), device=labels.device)

        for i, label in enumerate(gt_classes):
            j = matches[:, 0] == i
            if j.any():
                self.matrix[label, pred_classes[matches[j, 1].long()[0]]] += 1
            else:
                self.matrix[label, self.num_classes] += 1  # background FN

        for i, pred in enumerate(pred_classes):
            if not (matches[:, 1] == i).any():
                self.matrix[self.num_classes, pred] += 1  # background FP

    def print(self):
        print("Confusion Matrix:")
        print(self.matrix)

    def summary(self):
        TP = np.diag(self.matrix)
        FP = self.matrix[:-1, :-1].sum(0) - TP
        FN = self.matrix[:-1, :-1].sum(1) - TP
        precision = TP / (TP + FP + 1e-16)
        recall = TP / (TP + FN + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        return precision, recall, f1

    def mean_results(self):
        precision, recall, f1 = self.summary()
        return precision.mean(), recall.mean(), f1.mean()


class DetectionMetrics:
    """
    Beast-level detection metrics computation:
    - mAP@0.5:0.95 (COCO style)
    - mAP@0.5, mAP@0.75
    - Precision, Recall, F1-score
    - Per-class and overall statistics
    - Supports multiple IoU thresholds
    """
    def __init__(self, num_classes: int, iou_thresholds=None, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.iou_thresholds = (
            torch.arange(0.5, 0.96, 0.05, device=device)
            if iou_thresholds is None else torch.tensor(iou_thresholds, device=device)
        )

        # Accumulated state
        self.stats = []  # list of (tp, conf, pred_cls, target_cls)
        self.seen_images = set()

    @staticmethod
    def box_iou(box1, box2):
        """
        Compute IoU between two sets of boxes.
        box1: [N, 4], box2: [M, 4] in xyxy format
        """
        area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
        area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

        union = area1[:, None] + area2 - inter
        return inter / union.clamp(min=1e-6)

    def process_batch(self, detections, labels):
        """
        detections: [N, 6] (x1, y1, x2, y2, conf, cls)
        labels:     [M, 5] (cls, x1, y1, x2, y2)
        """
        gt_classes = labels[:, 0]
        gt_boxes = labels[:, 1:]

        if detections is None or detections.shape[0] == 0:
            if len(gt_classes):
                self.stats.append((torch.zeros(0, len(self.iou_thresholds)), 
                                   torch.Tensor(), torch.Tensor(), gt_classes))
            return

        pred_boxes = detections[:, :4]
        pred_scores = detections[:, 4]
        pred_classes = detections[:, 5]

        correct = torch.zeros(pred_boxes.size(0), len(self.iou_thresholds), dtype=torch.bool, device=self.device)

        if gt_boxes.numel():
            ious = self.box_iou(pred_boxes, gt_boxes)
            for i, iou_thr in enumerate(self.iou_thresholds):
                matches = torch.nonzero((ious >= iou_thr) & 
                                        (pred_classes[:, None] == gt_classes), as_tuple=False)
                if matches.shape[0]:
                    # Sort by confidence and IoU
                    matches = matches[ious[matches[:, 0], matches[:, 1]].argsort(descending=True)]
                    seen_gt = set()
                    for pred_idx, gt_idx in matches:
                        if gt_idx.item() not in seen_gt:
                            correct[pred_idx, i] = True
                            seen_gt.add(gt_idx.item())

        self.stats.append((correct.cpu(), pred_scores.cpu(), pred_classes.cpu(), gt_classes.cpu()))

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated stats"""
        if len(self.stats) == 0:
            return {k: 0.0 for k in ["mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1"]}

        correct, conf, pred_cls, target_cls = [torch.cat(x, 0) for x in zip(*self.stats)]
        precision, recall, ap = self.ap_per_class(correct, conf, pred_cls, target_cls)

        return {
            "mAP50": ap[:, 0].mean().item(),
            "mAP75": ap[:, 5].mean().item(),
            "mAP50_95": ap.mean(1).mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": (2 * precision * recall / (precision + recall + 1e-16)).mean().item()
        }

    def ap_per_class(self, tp, conf, pred_cls, target_cls):
        """COCO-style AP per class"""
        # Sort by confidence
        i = torch.argsort(conf, descending=True)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        unique_classes = torch.unique(target_cls)
        ap, p, r = [], [], []

        for c in unique_classes:
            i = pred_cls == c
            n_gt = (target_cls == c).sum().item()
            n_p = i.sum().item()

            if n_p == 0 or n_gt == 0:
                ap.append(torch.zeros(len(self.iou_thresholds)))
                p.append(0)
                r.append(0)
                continue

            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall_curve = tpc / (n_gt + 1e-16)
            precision_curve = tpc / (tpc + fpc)

            r.append(recall_curve[-1])
            p.append(precision_curve.max())

            ap_c = torch.zeros(len(self.iou_thresholds))
            for j in range(len(self.iou_thresholds)):
                ap_c[j] = self.compute_ap(recall_curve[:, j], precision_curve[:, j])
            ap.append(ap_c)

        return torch.tensor(p), torch.tensor(r), torch.stack(ap)

    @staticmethod
    def compute_ap(recall, precision):
        """Compute Average Precision (area under PR curve)"""
        mrec = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
        mpre = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))

        mpre = torch.flip(torch.cummax(torch.flip(mpre, [0]), dim=0)[0], [0])
        return torch.trapz(mpre, mrec)

    def reset(self):
        self.stats.clear()

def compute_segmentation_metrics(preds, targets, num_classes, ignore_index=None, eps=1e-7):
    """
    Compute segmentation metrics: mIoU, Dice, Precision, Recall, F1 for each class and overall.

    Args:
        preds (Tensor | List[Tensor]): Predicted segmentation masks (B,H,W) or (B,C,H,W) logits.
        targets (Tensor | List[Tensor]): Ground truth masks (B,H,W).
        num_classes (int): Total number of classes.
        ignore_index (int | None): Class index to ignore in metrics computation (e.g., background).
        eps (float): Epsilon to avoid division by zero.

    Returns:
        dict: {
            'overall': {...}, 
            'per_class': {class_id: {...}, ...}
        }
    """

    # Convert lists to tensors
    if isinstance(preds, list):
        preds = torch.stack(preds)
    if isinstance(targets, list):
        targets = torch.stack(targets)

    # If preds are logits/probabilities, take argmax to get class labels
    if preds.ndim == 4:
        preds = preds.argmax(dim=1)

    assert preds.shape == targets.shape, \
        f"Predictions and targets must have same shape, got {preds.shape} vs {targets.shape}"

    # Flatten for per-pixel analysis
    preds_flat = preds.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()

    # Mask out ignore_index if provided
    if ignore_index is not None:
        mask = targets_flat != ignore_index
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]

    # Initialize confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds_flat, targets_flat):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    # Metric containers
    per_class_metrics = {}
    ious, dices, precisions, recalls, f1s = [], [], [], [], []

    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        tn = confusion.sum() - (tp + fp + fn)

        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class_metrics[cls] = {
            'IoU': iou,
            'Dice': dice,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': confusion[cls, :].sum()
        }

    overall_metrics = {
        'mIoU': np.mean(ious),
        'mDice': np.mean(dices),
        'mPrecision': np.mean(precisions),
        'mRecall': np.mean(recalls),
        'mF1': np.mean(f1s),
        'Total Pixels': confusion.sum()
    }

    return {
        'overall': overall_metrics,
        'per_class': per_class_metrics,
        'confusion_matrix': confusion
    }