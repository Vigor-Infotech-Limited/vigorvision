import torch
import torch.nn.functional as F

def _load_torchvision_ops():
    import torchvision.ops as ops
    return ops

def bbox_iou(box1, box2, xywh=True, eps=1e-9):
    """
    Calculates IoU between two sets of boxes.
    Args:
        box1: (N, 4)
        box2: (M, 4)
        xywh: If True, expects (x, y, w, h) else (x1, y1, x2, y2)
    Returns:
        IoU: (N, M)
    """
    if xywh:
        # Convert from center xywh -> corners
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection
    inter_x1 = torch.max(b1_x1[:, None], b2_x1)
    inter_y1 = torch.max(b1_y1[:, None], b2_y1)
    inter_x2 = torch.min(b1_x2[:, None], b2_x2)
    inter_y2 = torch.min(b1_y2[:, None], b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    area1 = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    area2 = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
    union = area1[:, None] + area2 - inter_area + eps

    return inter_area / union


def bbox_giou(box1, box2, xywh=True, eps=1e-9):
    """
    Generalized IoU (GIoU)
    """
    iou = bbox_iou(box1, box2, xywh, eps)

    if xywh:
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Convex hull
    c_x1 = torch.min(b1_x1[:, None], b2_x1)
    c_y1 = torch.min(b1_y1[:, None], b2_y1)
    c_x2 = torch.max(b1_x2[:, None], b2_x2)
    c_y2 = torch.max(b1_y2[:, None], b2_y2)
    convex_area = (c_x2 - c_x1).clamp(0) * (c_y2 - c_y1).clamp(0)

    return iou - (convex_area - (iou * (convex_area + eps))) / convex_area.clamp(min=eps)


def bbox_diou(box1, box2, xywh=True, eps=1e-9):
    """
    Distance IoU (DIoU)
    """
    iou = bbox_iou(box1, box2, xywh, eps)

    if xywh:
        b1_cx = box1[:, 0]
        b1_cy = box1[:, 1]
        b2_cx = box2[:, 0]
        b2_cy = box2[:, 1]
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        b1_cx = (box1[:, 0] + box1[:, 2]) / 2
        b1_cy = (box1[:, 1] + box1[:, 3]) / 2
        b2_cx = (box2[:, 0] + box2[:, 2]) / 2
        b2_cy = (box2[:, 1] + box2[:, 3]) / 2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Center distance
    center_dist_sq = ((b1_cx[:, None] - b2_cx) ** 2 + (b1_cy[:, None] - b2_cy) ** 2)

    # Enclosing box diagonal
    c_x1 = torch.min(b1_x1[:, None], b2_x1)
    c_y1 = torch.min(b1_y1[:, None], b2_y1)
    c_x2 = torch.max(b1_x2[:, None], b2_x2)
    c_y2 = torch.max(b1_y2[:, None], b2_y2)
    diag_sq = ((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2).clamp(min=eps)

    return iou - center_dist_sq / diag_sq


def bbox_ciou(box1, box2, xywh=True, eps=1e-9):
    """
    Complete IoU (CIoU) — IoU + distance penalty + aspect ratio penalty
    """
    iou = bbox_iou(box1, box2, xywh, eps)

    if xywh:
        b1_cx = box1[:, 0]
        b1_cy = box1[:, 1]
        b2_cx = box2[:, 0]
        b2_cy = box2[:, 1]
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
        w1, h1 = box1[:, 2], box1[:, 3]
        w2, h2 = box2[:, 2], box2[:, 3]
    else:
        b1_cx = (box1[:, 0] + box1[:, 2]) / 2
        b1_cy = (box1[:, 1] + box1[:, 3]) / 2
        b2_cx = (box2[:, 0] + box2[:, 2]) / 2
        b2_cy = (box2[:, 1] + box2[:, 3]) / 2
        w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
        w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Center distance penalty
    center_dist_sq = ((b1_cx[:, None] - b2_cx) ** 2 + (b1_cy[:, None] - b2_cy) ** 2)
    c_x1 = torch.min(b1_x1[:, None], b2_x1)
    c_y1 = torch.min(b1_y1[:, None], b2_y1)
    c_x2 = torch.max(b1_x2[:, None], b2_x2)
    c_y2 = torch.max(b1_y2[:, None], b2_y2)
    diag_sq = ((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2).clamp(min=eps)

    # Aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1[:, None] / h1[:, None]), 2)
    alpha = v / (1 - iou + v + eps)

    return iou - (center_dist_sq / diag_sq + alpha * v)

def box_area(boxes):
    """
    Compute the area of a set of boxes in (x1, y1, x2, y2) format.
    Args:
        boxes (Tensor[N, 4]): boxes with corners
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)

def nms(boxes, scores, iou_threshold=0.6):
    """Performs Non-Maximum Suppression"""
    ops = _load_torchvision_ops()
    return ops.nms(boxes, scores, iou_threshold)

def box_iou(box1, box2, eps=1e-7):
    """
    Standard IoU between box1 and box2.
    Args:
        box1: (N, 4) [x1, y1, x2, y2]
        box2: (M, 4) [x1, y1, x2, y2]
    Returns:
        iou (Tensor[N, M])
        union (Tensor[N, M])
    """
    area1 = box_area(box1)
    area2 = box_area(box2)

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter + eps
    iou = inter / union
    return iou, union


def box_giou(box1, box2, eps=1e-7):
    """
    Generalized IoU from https://giou.stanford.edu/
    Args:
        box1: (N, 4)
        box2: (M, 4)
    Returns:
        giou (Tensor[N, M])
    """
    iou, _ = box_iou(box1, box2, eps)

    x1 = torch.min(box1[:, None, 0], box2[:, 0])
    y1 = torch.min(box1[:, None, 1], box2[:, 1])
    x2 = torch.max(box1[:, None, 2], box2[:, 2])
    y2 = torch.max(box1[:, None, 3], box2[:, 3])

    area_c = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    _, union = box_iou(box1, box2, eps)
    inter = iou * union

    return iou - (area_c - union) / area_c.clamp(min=eps)


def box_diou(box1, box2, eps=1e-7):
    """
    Distance IoU from https://arxiv.org/pdf/1911.08287.pdf
    """
    iou, _ = box_iou(box1, box2, eps)

    center1 = (box1[:, None, 0:2] + box1[:, None, 2:4]) / 2
    center2 = (box2[:, 0:2] + box2[:, 2:4]) / 2

    center_dist_sq = ((center1 - center2) ** 2).sum(dim=2)

    x1 = torch.min(box1[:, None, 0], box2[:, 0])
    y1 = torch.min(box1[:, None, 1], box2[:, 1])
    x2 = torch.max(box1[:, None, 2], box2[:, 2])
    y2 = torch.max(box1[:, None, 3], box2[:, 3])
    diag_c = ((x2 - x1) ** 2 + (y2 - y1) ** 2).clamp(min=eps)

    return iou - center_dist_sq / diag_c


def box_ciou(box1, box2, eps=1e-7):
    """
    Complete IoU from https://arxiv.org/pdf/1911.08287.pdf
    """
    iou, _ = box_iou(box1, box2, eps)

    center1 = (box1[:, None, 0:2] + box1[:, None, 2:4]) / 2
    center2 = (box2[:, 0:2] + box2[:, 2:4]) / 2
    center_dist_sq = ((center1 - center2) ** 2).sum(dim=2)

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


def iou_types():
    """
    Returns list of supported IoU types
    """
    return ['iou', 'giou', 'diou', 'ciou']


def compute_iou(box1, box2, iou_type='iou'):
    """
    Unified IoU computation interface
    Args:
        box1: Tensor[N, 4]
        box2: Tensor[M, 4]
        iou_type: str in ['iou', 'giou', 'diou', 'ciou']
    Returns:
        iou (Tensor[N, M])
    """
    assert iou_type in iou_types(), f"Invalid IoU type: {iou_type}"

    if iou_type == 'iou':
        return box_iou(box1, box2)[0]
    elif iou_type == 'giou':
        return box_giou(box1, box2)
    elif iou_type == 'diou':
        return box_diou(box1, box2)
    elif iou_type == 'ciou':
        return box_ciou(box1, box2)


def test_iou():
    box1 = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[1, 1, 2, 2], [2, 2, 4, 4]], dtype=torch.float32)

    for iou_type in iou_types():
        result = compute_iou(box1, box2, iou_type)
        print(f"{iou_type.upper()}:\n{result}")


if __name__ == "__main__":
    test_iou()
