# vigorvision/utils/autoanchor.py

import torch
import numpy as np
from sklearn.cluster import KMeans
import random
import time
from tqdm import tqdm
import logging
import math
import torch
import numpy as np

import logging

LOGGER = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")




def check_anchors(model, dataloader, thr=4.0, imgsz=640):
    LOGGER.info(f"Checking anchors for optimal fit with dataset (Threshold: {thr})...")

    # Use model's top-level anchors & strides
    try:
        anchors = model.anchors.clone().detach()
        strides = model.stride
    except AttributeError:
        raise AttributeError("VisionModel must have 'anchors' and 'strides' attributes.")

    try:
        anchors = model.anchors.clone().detach()
        # Convert stride to tensor if it is a list
        if isinstance(model.stride, list):
            strides = torch.tensor(model.stride, dtype=torch.float32)
        else:
            strides = model.stride
    except AttributeError:
        raise AttributeError("VisionModel must have 'anchors' and 'stride' attributes.")

    # Now scale anchors
    anchors = anchors * strides.view(-1, 1, 1)

    nl, na = anchors.shape[:2]
    LOGGER.info(f"Anchors (pixels): {anchors.view(-1, 2).cpu().numpy()}")

    # Collect all GT boxes from dataloader
    all_labels = []

    for i, (_, targets) in enumerate(dataloader):
        if isinstance(targets, list) and len(targets) > 0:
            targets_cat = torch.cat([t for t in targets if t.numel() > 0], 0)
        elif isinstance(targets, dict) and 'boxes' in targets:
            targets_cat = targets['boxes']
        else:
            print(f"Batch {i} has empty or unexpected targets: {targets}")
            continue

        print(f"Batch {i} targets: {targets_cat.shape}")
        all_labels.append(targets_cat)

    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels, 0)
        print("All labels collected:", all_labels.shape)
    else:
        print("No labels found in the dataset!")


    # Normalize to pixel space
    gwh = all_labels[:, 2:4] * imgsz
    gwh = gwh[gwh.min(1)[0] > 2.0]

    if gwh.numel() == 0:
        LOGGER.warning("All boxes are too small to evaluate anchors.")
        return

    # Calculate ratios between anchors and GT boxes
    ratios = []
    for anchor in anchors.view(-1, 2):
        r = gwh[:, None] / anchor[None]
        r = torch.min(r, 1 / r).max(2)[0]
        ratios.append(r)
    ratios = torch.stack(ratios, 1)

    best = ratios.min(1)[0]
    bpr = (best < (1 / thr)).float().mean().item()

    LOGGER.info(f"Best possible recall (BPR): {bpr:.4f}")
    if bpr < 0.98:
        LOGGER.warning(f"Low BPR ({bpr:.4f}) — Anchors may be suboptimal. Run autoanchor generation.")
    else:
        LOGGER.info("Anchors look good — no need for regeneration.")

    return bpr




def wh_iou(wh1, wh2, eps=1e-9):
    """Compute IoU between two sets of boxes (width-height format)."""
    wh1 = wh1[:, None]  # [n, 1, 2]
    wh2 = wh2[None]     # [1, m, 2]
    inter = torch.min(wh1, wh2).prod(2)
    union = (wh1.prod(2) + wh2.prod(2) - inter).clamp(min=eps)
    return inter / union


class AutoAnchor:
    """
    Automatic Anchor Box Generator using:
    - KMeans++ clustering
    - Genetic Evolution for refinement
    """

    def __init__(self, dataset, n_anchors=9, img_size=640, iou_threshold=0.25, generations=1000):
        """
        Args:
            dataset: iterable of dicts {'boxes': Tensor[N, 4] (x1,y1,x2,y2)}
            n_anchors: number of anchors to generate
            img_size: model input size (for normalization)
            iou_threshold: IoU threshold for fitness computation
            generations: number of genetic mutation generations
        """
        self.dataset = dataset
        self.n_anchors = n_anchors
        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.generations = generations
        
    def _collect_wh(self):
        logger.info("Collecting bounding boxes for anchor computation...")
        wh = []
        for sample in tqdm(self.dataset):
            # Case 1: dict
            if isinstance(sample, dict) and "boxes" in sample:
                boxes = sample["boxes"]

            # Case 2: tuple
            elif isinstance(sample, (tuple, list)) and len(sample) > 1:
                labels = sample[1]
                if isinstance(labels, (tuple, list)) and len(labels) > 0:
                    boxes = labels[0]  # first element = boxes
                else:
                    continue
            else:
                raise ValueError("Dataset format not recognized. Expected dict with 'boxes' or tuple (img, labels).")

            # Convert to numpy
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            if boxes.size == 0:
                continue

            # Force [N,2] shape
            wh_i = boxes[:, 2:] - boxes[:, :2]   # (N,2)
            if wh_i.ndim == 1:  # handle single box
                wh_i = wh_i.reshape(1, -1)

            wh.extend(wh_i)

        wh = np.array(wh).reshape(-1, 2)   # ✅ ensure always (N,2)
        wh = wh[(wh > 2).all(1)]           # filter tiny boxes
        return wh / self.img_size

    def _fitness(self, anchors, wh):
        iou = wh_iou(torch.tensor(wh, dtype=torch.float32),
                     torch.tensor(anchors, dtype=torch.float32))
        return (iou.max(1)[0] > self.iou_threshold).float().mean().item()

    def run(self):
        """Main entry point to generate anchors."""
        wh = self._collect_wh()

        logger.info(f"Running KMeans++ to find {self.n_anchors} anchors...")
        best_anchors = KMeans(n_clusters=self.n_anchors, n_init=10, random_state=0).fit(wh).cluster_centers_
        best_anchors = best_anchors[np.argsort(best_anchors.prod(1))]  # sort by area

        best_fitness = self._fitness(best_anchors, wh)
        logger.info(f"Initial anchor fitness: {best_fitness:.4f}")

        logger.info("Starting Genetic Evolution...")
        mutate_scale = 0.9
        start_time = time.time()

        for gen in range(self.generations):
            anchors_new = best_anchors * (np.random.normal(1, mutate_scale, best_anchors.shape))
            anchors_new = anchors_new.clip(min=1e-3)
            anchors_new = anchors_new[np.argsort(anchors_new.prod(1))]

            fitness = self._fitness(anchors_new, wh)
            if fitness > best_fitness:
                best_fitness = fitness
                best_anchors = anchors_new
                logger.info(f"Gen {gen+1}/{self.generations} - Improved fitness to {fitness:.4f}")

        best_anchors = best_anchors * self.img_size  # de-normalize
        logger.info(f"Best anchors after evolution:\n{np.round(best_anchors, 2)}")
        logger.info(f"Total evolution time: {time.time() - start_time:.2f}s")
        
        return np.round(best_anchors, 2)


def test_autoanchor():
    """Standalone test with synthetic data."""
    dataset = []
    for _ in range(1000):
        boxes = torch.rand((random.randint(1, 5), 4)) * 640
        boxes[:, 2:] += boxes[:, :2]
        dataset.append({'boxes': boxes})

    auto_anchor = AutoAnchor(dataset, n_anchors=9, img_size=640)
    anchors = auto_anchor.run()
    print("Generated Anchors:\n", anchors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_autoanchor()


def check_anchor_order(anchors: torch.Tensor, stride: torch.Tensor):
    """
    Ensure anchors are ordered from small to large according to stride.
    
    Args:
        anchors (Tensor[nl, na, 2]): Anchors in pixels per detection layer.
        stride (Tensor[nl]): Stride per detection layer.
    
    Returns:
        anchors (Tensor[nl, na, 2]): Possibly reordered anchors.
        stride (Tensor[nl]): Possibly reordered stride.
    """
    # Ensure input types
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, dtype=torch.float32)
    if not isinstance(stride, torch.Tensor):
        stride = torch.tensor(stride, dtype=torch.float32)

    # Mean anchor area for each detection layer
    anchor_areas = anchors.prod(dim=2).mean(dim=1)  # shape: (nl,)
    stride_order = stride.argsort()                 # expected order
    anchor_order = anchor_areas.argsort()           # actual order

    if not torch.equal(stride_order, anchor_order):
        logging.warning(
            f"⚠️ Anchor order {anchor_order.tolist()} "
            f"does not match stride order {stride_order.tolist()} — fixing automatically."
        )
        anchors = anchors[anchor_order]
        stride = stride[anchor_order]
    else:
        logging.info("✅ Anchor order matches stride order.")

    return anchors, stride


if __name__ == "__main__":
    # Quick test
    anchors = torch.tensor([
        [[116,90],[156,198],[373,326]],  # large scale
        [[30,61],[62,45],[59,119]],      # medium scale
        [[10,13],[16,30],[33,23]],       # small scale
    ], dtype=torch.float32)

    stride = torch.tensor([32, 16, 8], dtype=torch.float32)

    new_anchors, new_stride = check_anchor_order(anchors, stride)
    print("New Anchors:", new_anchors)
    print("New Strides:", new_stride)
