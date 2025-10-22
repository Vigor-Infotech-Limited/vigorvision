import os
import glob
import random
import warnings
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
import albumentations as A
import cv2

# ---------------------------- Albumentations Transform ----------------------------
def default_albumentations_transform(img_size):
    """Same name, improved internal handling."""
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.CLAHE(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Resize(*img_size)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ---------------------------- Mosaic Augmentation ----------------------------
def load_mosaic_image(dataset, index, input_size):
    """Rewritten for consistency and correctness."""
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    indices = [index] + [random.randint(0, len(dataset) - 1) for _ in range(3)]
    mosaic_img = Image.new('RGB', (input_size[0] * 2, input_size[1] * 2), (114, 114, 114))
    mosaic_labels = []

    xc = int(random.uniform(input_size[0] * 0.5, input_size[0] * 1.5))
    yc = int(random.uniform(input_size[1] * 0.5, input_size[1] * 1.5))

    for i, idx in enumerate(indices):
        img, labels = dataset.load_image_and_labels(idx)
        w, h = img.size
        img = img.resize(input_size)
        scale_x, scale_y = input_size[0] / w, input_size[1] / h
        labels = labels.clone()
        if labels.shape[0]:
            labels[:, 1:] *= torch.tensor([scale_x, scale_y, scale_x, scale_y])

        if i == 0:
            x1a, y1a, x2a, y2a = max(xc - input_size[0], 0), max(yc - input_size[1], 0), xc, yc
            x1b, y1b = input_size[0] - (x2a - x1a), input_size[1] - (y2a - y1a)
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - input_size[1], 0), min(xc + input_size[0], input_size[0] * 2), yc
            x1b, y1b = 0, input_size[1] - (y2a - y1a)
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - input_size[0], 0), yc, xc, min(input_size[1] * 2, yc + input_size[1])
            x1b, y1b = input_size[0] - (x2a - x1a), 0
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + input_size[0], input_size[0] * 2), min(input_size[1] * 2, yc + input_size[1])
            x1b, y1b = 0, 0

        mosaic_img.paste(img, (x1a, y1a))
        if labels.shape[0]:
            labels[:, 1] = labels[:, 1] * input_size[0] + x1a
            labels[:, 2] = labels[:, 2] * input_size[1] + y1a
            mosaic_labels.append(labels)

    if mosaic_labels:
        mosaic_labels = torch.cat(mosaic_labels, dim=0)
        # Normalize
        mosaic_labels[:, 1::2] /= (input_size[0] * 2)
        mosaic_labels[:, 2::2] /= (input_size[1] * 2)
    else:
        mosaic_labels = torch.zeros((0, 5), dtype=torch.float32)

    if mosaic_labels.numel() > 0:
        mosaic_labels[:, 1::2].clamp_(0.0, 1.0)
        mosaic_labels[:, 2::2].clamp_(0.0, 1.0)

    return mosaic_img.resize(input_size), mosaic_labels

# ---------------------------- Detection Dataset ----------------------------
class DetectionDataset(Dataset):
    """Fully aligned with your earlier interface, now path-safe and validated."""
    def __init__(self, img_dir, transform=None, img_size=(640, 640), classes=11,
                 mosaic_prob=0.25, use_lmdb=False, lmdb_path=None):
        
        # Force correct structure recognition
        img_subdir = img_dir + "/images"
        label_subdir = img_dir + "/labels"
        
        self.img_dir = img_subdir
        self.label_dir = label_subdir
        
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) +
            glob.glob(os.path.join(self.img_dir, "*.png")) +
            glob.glob(os.path.join(self.img_dir, "*.jpeg"))
        )

        if len(self.img_paths) == 0:
            raise ValueError(f"[ERROR] No images found in {self.img_dir}. Verify dataset structure.")

        self.transform = transform or default_albumentations_transform(img_size)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.classes = classes
        self.mosaic_prob = mosaic_prob
        self.use_lmdb = use_lmdb
        self.lmdb_path = lmdb_path

        print(f"[INFO] Loaded {len(self.img_paths)} images from {self.img_dir}")

    def _parse_labels(self, label_path):
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id, x, y, w, h = map(float, parts[:5])
                            if 0 <= cls_id < self.classes:
                                boxes.append([cls_id, x, y, w, h])
                        except Exception:
                            continue
        return boxes

    def load_image_and_labels(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        boxes = self._parse_labels(label_path)
        labels = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return img, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if random.random() < self.mosaic_prob:
            image, targets = load_mosaic_image(self, index, self.img_size)
        else:
            image, targets = self.load_image_and_labels(index)

        # Albumentations expects numpy + list labels
        image_np = np.array(image)
        bboxes = targets[:, 1:].tolist()
        class_labels = targets[:, 0].tolist()

        try:
            transformed = self.transform(image=image_np, bboxes=bboxes, class_labels=class_labels)
            image = T.ToTensor()(transformed["image"])
            boxes = torch.tensor(
                [[cls] + list(bbox) for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"])],
                dtype=torch.float32
            )
        except Exception as e:
            warnings.warn(f"Albumentations failed: {e}")
            image = T.ToTensor()(image)
            boxes = targets

        return image, boxes

# ---------------------------- Collate Function ----------------------------
def collate_fn(batch):
    images, targets = list(zip(*batch))
    
    # ✅ Force consistent size
    target_size = (images[0].shape[1], images[0].shape[2])
    images = [
        F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        if img.shape[1:] != target_size else img
        for img in images
    ]
    
    images = torch.stack(images, dim=0)
    return images, targets


# ---------------------------- Dataloader ----------------------------
def get_dataloader(
    img_dir,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    img_size=(640, 640),
    classes=11,
    mosaic_prob=0.25,
    use_lmdb=False,
    lmdb_path=None,
    transform=None
):
    dataset = DetectionDataset(
        img_dir=img_dir,
        transform=transform,
        img_size=img_size,
        classes=classes,
        mosaic_prob=mosaic_prob,
        use_lmdb=use_lmdb,
        lmdb_path=lmdb_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
