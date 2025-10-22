import torch
import torch.nn as nn
import torch.nn.functional as F
from vigorvision.utils.general import make_anchors
from vigorvision.utils.box_ops import bbox_iou
from vigorvision.utils.autoanchor import check_anchor_order
from vigorvision.nn.loss import ComputeLoss
from vigorvision.nn.module.conv import ConvBlock


class DetectionHead(nn.Module):
    def __init__(self, ch, nc, anchors, stride):
        """
        Args:
            ch (list[int]): Input channels from neck per detection scale.
            nc (int): Number of object classes.
            anchors (list[list[tuple]]): Anchor box sizes per scale.
            stride (list[int]): Stride per scale.
        """
        super().__init__()
        self.nc = nc
        self.no = nc + 5 
        
        self.stride = stride

        num_layers = len(stride) if 'stride' in locals() else 3
        num_anchors = len(anchors) // num_layers

        self.nl = num_layers  # number of detection layers
        self.na = num_anchors  # number of anchors per layer

        self.anchors = torch.tensor(anchors).float().view(self.nl, self.na, 2)  # [nl, na, 2]


        self.detect_layers = nn.ModuleList([
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        ])

        hyp = {
            "box": 0.05,          
            "cls": 0.5,           
            "obj": 1.0,           
            "fl_gamma": 1.5,
            "label_smoothing": 0.0
        }
        self.loss_fn = ComputeLoss(model=self, hyp=hyp, autobalance=False, focal_loss=True, device="cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, x, targets=None):
        z = []  # inference output
        losses = {}  # training loss output

        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = self.detect_layers[i](x[i])
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                # Apply sigmoid to xy, obj, cls predictions
                y = x[i].sigmoid()
                anchors_grid = self.anchors[i].to(x[i].device) / self.stride[i]
                grid = make_anchors((ny, nx), anchors_grid, stride=self.stride[i])
                pred = self.decode_preds(y, grid, self.stride[i])
                z.append(pred.view(bs, -1, self.no))

        if self.training:
            loss, loss_items = self.loss_fn(x, targets)
            losses = {
                'box_loss': loss_items[0],
                'obj_loss': loss_items[1],
                'cls_loss': loss_items[2],
                'focal_loss': loss_items[3],
                'total_loss': loss
            }
            return x, losses

        return torch.cat(z, 1)  # inference output: [bs, total_anchors, no]

    def decode_preds(self, pred, grid, stride):
        """
        Decode predictions: convert network output to xywh.
        """
        xy = (pred[..., 0:2] * 2 - 0.5 + grid) * stride  # x, y
        wh = (pred[..., 2:4] * 2) ** 2 * self.anchors[0].to(pred.device)  # width, height
        conf = pred[..., 4:]
        return torch.cat((xy, wh, conf), -1)


def test_detection_head():
    from vigorvision.utils.autoanchor import generate_anchors
    ch = [256, 512, 1024]
    nc = 11
    stride = [8, 16, 32]
    anchors = generate_anchors(base_sizes=[(10,13), (16,30), (33,23)], scales=3)

    x = [torch.randn(2, ch[i], 32 // (2**i), 32 // (2**i)) for i in range(3)]
    model = DetectionHead(ch=ch, nc=nc, anchors=anchors, stride=stride)
    model.eval()
    y = model(x)
    print(f"Inference shape: {y.shape}")


if __name__ == "__main__":
    test_detection_head()
