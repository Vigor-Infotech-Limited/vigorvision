# -------------------------vigorvision/loss.py---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from vigorvision.utils.iou import bbox_ciou
from vigorvision.utils.general import one_hot

class ComputeLoss:
    def __init__(self, model, hyp, autobalance=False, focal_loss=True, device="cpu"):
        """
        ComputeLoss initializes loss components for training.
        Args:
            model: Detection model with anchor information.
            hyp: Hyperparameters dictionary with keys: box, cls, obj, fl_gamma, label_smoothing.
            autobalance: If True, enables balancing of objectness loss.
            focal_loss: If True, applies focal loss to objectness and classification loss.
            device: Torch device.
        """
        self.device = device
        self.hyp = hyp
        self.autobalance = autobalance
        self.focal_loss = focal_loss
        self.box_gain = hyp['box']
        self.cls_gain = hyp['cls']
        self.obj_gain = hyp['obj']
        self.fl_gamma = hyp.get("fl_gamma", 1.5)
        self.label_smoothing = hyp.get("label_smoothing", 0.0)

        # Loss functions
        BCEcls = nn.BCEWithLogitsLoss(reduction='none')
        BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        if focal_loss:
            self.cls_loss_fn = self.focal(BCEcls)
            self.obj_loss_fn = self.focal(BCEobj)
        else:
            self.cls_loss_fn = BCEcls
            self.obj_loss_fn = BCEobj

        self.model = model
        self.anchors = model.anchors.to(device)
        self.strides = model.stride
        self.na = self.anchors.shape[1]  # num anchors per scale
        self.nl = self.anchors.shape[0]  # num detection layers

    def focal(self, loss_fn):
        def func(pred, true):
            loss = loss_fn(pred, true)
            pred_prob = torch.sigmoid(pred)
            p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
            alpha_factor = true * 0.25 + (1 - true) * 0.75
            modulating_factor = (1.0 - p_t) ** self.fl_gamma
            return (alpha_factor * modulating_factor * loss).mean()
        return func

    def __call__(self, preds, targets):
        """
        Args:
            preds: list of (bs, na, ny, nx, no) - raw predictions from each scale.
            targets: (n, 6) [image_id, class, x, y, w, h]
        Returns:
            total_loss, dict with loss components
        """
        device = targets.device
        bs = preds[0].shape[0]
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        targets_list, indices, anchors = self.build_targets(preds, targets)

        for i, pred in enumerate(preds):
            b, a, gj, gi = indices[i]  # image, anchor, grid_y, grid_x
            tobj = torch.zeros_like(pred[..., 0], device=device)

            if b.shape[0]:
                ps = pred[b, a, gj, gi]
                # Regression
                pxy = ps[:, 0:2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_ciou(pbox, targets_list[i][:, 2:6])
                lbox += (1.0 - iou).mean()

                # Objectness
                tobj[b, a, gj, gi] = iou.detach().clamp(0)

                # Classification
                t = one_hot(targets_list[i][:, 1].long(), pred.shape[-1] - 5, self.label_smoothing).to(device)
                lcls += self.cls_loss_fn(ps[:, 5:], t)

            lobj += self.obj_loss_fn(pred[..., 4], tobj)

        lbox *= self.box_gain
        lcls *= self.cls_gain
        lobj *= self.obj_gain

        total_loss = lbox + lcls + lobj
        return total_loss, {
            'box_loss': lbox.detach(),
            'cls_loss': lcls.detach(),
            'obj_loss': lobj.detach(),
            'total_loss': total_loss.detach()
        }

    def build_targets(self, preds, targets):
        """
        Match ground truth targets to anchors and prediction grids.
        Returns:
            targets, matched indices, and anchors for each layer.
        """
        na, nt = self.na, targets.shape[0]
        targets_list, indices, anchors_out = [], [], []

        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2).view(-1, 7)

        for i, pred in enumerate(preds):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(pred.shape)[[3, 2, 3, 2]]  # xywh gain

            t = targets.clone()
            t[:, 2:6] *= gain[2:6]

            r = t[:, 4:6] / anchors[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < 4.0

            t = t[j]
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy.long()
            gi, gj = gij.T
            a = t[:, 6].long()
            indices.append((b, a, gj, gi))
            targets_list.append(torch.cat((gxy - gij, gwh, t[:, 1:2]), 1))
            anchors_out.append(anchors[a])

        return targets_list, indices, anchors_out
