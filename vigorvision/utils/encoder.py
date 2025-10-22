import torch
import torch.nn.functional as F

from vigorvision.utils.general import xywh2xyxy

class TargetEncoder:
    """
    Encodes ground truth boxes and labels into training targets
    for anchor-based detection models.
    """

    def __init__(self, anchors, strides, num_classes, ignore_thresh=0.5):
        """
        Args:
            anchors (List[List[Tuple]]): List of anchor boxes per detection layer.
            strides (List[int]): List of strides per layer.
            num_classes (int): Number of target classes.
            ignore_thresh (float): IoU threshold for negative matches.
        """
        from vigorvision.utils.iou import bbox_iou
        self.anchors = anchors
        self.strides = strides
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

    def __call__(self, targets, image_size):
        """
        Args:
            targets (Tensor): [num_targets, 6] => (image_idx, class, x, y, w, h) normalized
            image_size (int): Input image size (e.g., 640)
        
        Returns:
            List of tuples: (tcls, tbox, indices, anchors) per layer
        """
        na = [len(a) for a in self.anchors]  # anchors per layer
        nl = len(self.anchors)  # num detection layers
        targets_per_layer = []

        gain = torch.ones(7, device=targets.device)  # [img_idx, cls, x, y, w, h, anchor_id]
        for i in range(nl):
            anchors_layer = self.anchors[i]
            stride = self.strides[i]
            grid_size = image_size // stride
            anchors_tensor = torch.tensor(anchors_layer, device=targets.device)

            t = targets.clone()
            gain[2:6] = grid_size  # scale x,y,w,h

            t = t.repeat(len(anchors_layer), 1, 1)  # [na, nt, 6]
            ai = torch.arange(len(anchors_layer), device=targets.device).float().view(-1, 1).repeat(1, t.shape[1])
            t = torch.cat((t, ai[:, :, None]), dim=2)  # append anchor index
            t = t.view(-1, 7)

            # match targets to anchors
            gwh = t[:, 4:6] * grid_size
            ratios = gwh[:, None] / anchors_tensor[None]
            matches = torch.max(ratios, 1. / ratios).max(2)[0] < 4.0  # shape (num_targets * na, na)

            t = t[matches.view(-1)]  # keep only matched targets
            if t.shape[0] == 0:
                targets_per_layer.append((torch.zeros((0,), dtype=torch.long),  # tcls
                                          torch.zeros((0, 4)),                # tbox
                                          (torch.zeros((0,), dtype=torch.long),  # image idx
                                           torch.zeros((0,), dtype=torch.long),  # anchor idx
                                           torch.zeros((0,), dtype=torch.long),  # grid_y
                                           torch.zeros((0,), dtype=torch.long)),  # grid_x
                                          torch.zeros((0, 2))))  # anchors
                continue

            gxy = t[:, 2:4] * grid_size
            gwh = t[:, 4:6] * grid_size
            gij = gxy.long()
            gi, gj = gij.T

            indices = (t[:, 0].long(), t[:, 6].long(), gj.clamp_(0, grid_size - 1), gi.clamp_(0, grid_size - 1))
            tbox = torch.cat((gxy - gij, gwh), 1)  # delta x/y, width, height
            anchors_used = anchors_tensor[t[:, 6].long()]
            tcls = t[:, 1].long()

            targets_per_layer.append((tcls, tbox, indices, anchors_used))

        return targets_per_layer
