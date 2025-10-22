# vigorvision/utils/box_ops.py

import torch

def _load_torchvision_ops():
    import torchvision.ops as ops
    return ops

def nms(boxes, scores, iou_threshold=0.6):
    """Performs Non-Maximum Suppression"""
    ops = _load_torchvision_ops()
    return ops.nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, labels, iou_threshold=0.6):
    """NMS across multiple classes (label-aware)"""
    ops = _load_torchvision_ops()
    return ops.batched_nms(boxes, scores, labels, iou_threshold)


def xyxy_to_xywh(boxes):
    """Convert [x1, y1, x2, y2] → [x, y, w, h]"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    w, h = x2 - x1, y2 - y1
    x, y = x1 + w / 2, y1 + h / 2
    return torch.stack([x, y, w, h], dim=-1)


def xywh_to_xyxy(boxes):
    """Convert [x, y, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = boxes.unbind(-1)
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes):
    """Compute area of [x1, y1, x2, y2] boxes"""
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)


def box_iou(box1, box2):
    """IoU between two sets of boxes (N,4), (M,4)"""
    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / union.clamp(min=1e-6)


def box_giou(box1, box2):
    """Generalized IoU"""
    iou = box_iou(box1, box2)

    lt = torch.min(box1[:, None, :2], box2[:, :2])
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - (iou * (box_area(box1)[:, None] + box_area(box2) - iou * (box_area(box1)[:, None] + box_area(box2)))) / area.clamp(min=1e-6))


def box_diou(box1, box2):
    """Distance IoU"""
    iou = box_iou(box1, box2)

    center1 = (box1[:, None, :2] + box1[:, None, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    center_dist = ((center1 - center2) ** 2).sum(dim=-1)

    lt = torch.min(box1[:, None, :2], box2[:, :2])
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])
    diag_dist = ((rb - lt) ** 2).sum(dim=-1)

    return iou - center_dist / diag_dist.clamp(min=1e-6)


def box_ciou(box1, box2):
    """Complete IoU (adds aspect ratio penalty)"""
    iou = box_iou(box1, box2)

    center1 = (box1[:, None, :2] + box1[:, None, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    center_dist = ((center1 - center2) ** 2).sum(dim=-1)

    lt = torch.min(box1[:, None, :2], box2[:, :2])
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])
    diag_dist = ((rb - lt) ** 2).sum(dim=-1)

    w1 = (box1[:, None, 2] - box1[:, None, 0]).clamp(min=1e-6)
    h1 = (box1[:, None, 3] - box1[:, None, 1]).clamp(min=1e-6)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=1e-6)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=1e-6)

    v = (4 / (torch.pi ** 2)) * ((torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=1e-6)

    return iou - (center_dist / diag_dist + alpha * v)


def clip_boxes(boxes, img_shape):
    """Clip boxes to image boundaries"""
    h, w = img_shape
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return boxes
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300
):
    import torchvision.ops as ops
    
    """
    Performs Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction (torch.Tensor): [batch, num_boxes, 5 + num_classes] -> [x, y, w, h, conf, cls_scores...]
        conf_thres (float): Confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        classes (list[int] or None): Filter by class IDs.
        agnostic (bool): Class-agnostic NMS.
        multi_label (bool): Allow multi-label per box.
        max_det (int): Maximum number of detections per image.

    Returns:
        list[torch.Tensor]: Detections per image: [x1, y1, x2, y2, conf, cls]
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # just in case

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates mask

    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):  # per image
        x = x[xc[xi]]  # filter by objectness confidence

        if not x.shape[0]:
            continue

        # Compute conf = obj_conf * class_conf
        x[:, 5:] *= x[:, 4:5]

        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
        box = xywh_to_xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If no boxes remain, skip
        n = x.shape[0]
        if not n:
            continue

        # Offset boxes by class for class-wise NMS
        c = x[:, 5:6] * (0 if agnostic else 4096)
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS
        keep = ops.nms(boxes, scores, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[xi] = x[keep]

    return output

import torch

def _xywh_to_xyxy(boxes):
    """Convert [x, y, w, h] → [x1, y1, x2, y2] format."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)

def _box_format(boxes):
    """Ensure boxes are in xyxy format."""
    if boxes.size(-1) != 4:
        raise ValueError(f"Boxes must have shape (..., 4), got {boxes.shape}")
    if (boxes[..., 2:] < boxes[..., :2]).any():
        # Assume they are xywh
        return _xywh_to_xyxy(boxes)
    return boxes

def bbox_iou(box1, box2, xywh=False, eps=1e-7):
    """
    IoU between two sets of boxes.
    Args:
        box1: (N, 4)
        box2: (M, 4)
        xywh: True if boxes are in center-x, center-y, width, height format.
        eps: epsilon to avoid division by zero.
    Returns:
        iou: (N, M)
    """
    if xywh:
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)

    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter = inter_w * inter_h

    union = area1[:, None] + area2 - inter + eps
    return inter / union


def bbox_giou(box1, box2, xywh=False, eps=1e-7):
    """Generalized IoU."""
    if xywh:
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)

    iou = bbox_iou(box1, box2, xywh=False, eps=eps)

    x1 = torch.min(box1[:, None, 0], box2[:, 0])
    y1 = torch.min(box1[:, None, 1], box2[:, 1])
    x2 = torch.max(box1[:, None, 2], box2[:, 2])
    y2 = torch.max(box1[:, None, 3], box2[:, 3])

    area_c = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    return iou - (area_c - (iou * (area_c - iou * area_c))) / area_c.clamp(min=eps)


def bbox_diou(box1, box2, xywh=False, eps=1e-7):
    """Distance IoU."""
    if xywh:
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)

    iou = bbox_iou(box1, box2, xywh=False, eps=eps)

    center1 = (box1[:, None, 0:2] + box1[:, None, 2:4]) / 2
    center2 = (box2[:, 0:2] + box2[:, 2:4]) / 2
    center_dist_sq = ((center1 - center2) ** 2).sum(dim=-1)

    x1 = torch.min(box1[:, None, 0], box2[:, 0])
    y1 = torch.min(box1[:, None, 1], box2[:, 1])
    x2 = torch.max(box1[:, None, 2], box2[:, 2])
    y2 = torch.max(box1[:, None, 3], box2[:, 3])
    diag_c = ((x2 - x1) ** 2 + (y2 - y1) ** 2).clamp(min=eps)

    return iou - center_dist_sq / diag_c


def bbox_ciou(box1, box2, xywh=False, eps=1e-7):
    """Complete IoU."""
    if xywh:
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)

    iou = bbox_iou(box1, box2, xywh=False, eps=eps)

    center1 = (box1[:, None, 0:2] + box1[:, None, 2:4]) / 2
    center2 = (box2[:, 0:2] + box2[:, 2:4]) / 2
    center_dist_sq = ((center1 - center2) ** 2).sum(dim=-1)

    x1 = torch.min(box1[:, None, 0], box2[:, 0])
    y1 = torch.min(box1[:, None, 1], box2[:, 1])
    x2 = torch.max(box1[:, None, 2], box2[:, 2])
    y2 = torch.max(box1[:, None, 3], box2[:, 3])
    diag_c = ((x2 - x1) ** 2 + (y2 - y1) ** 2).clamp(min=eps)

    w1 = (box1[:, None, 2] - box1[:, None, 0]).clamp(min=eps)
    h1 = (box1[:, None, 3] - box1[:, None, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)

    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    alpha = v / (1 - iou + v + eps)

    return iou - (center_dist_sq / diag_c + alpha * v)

