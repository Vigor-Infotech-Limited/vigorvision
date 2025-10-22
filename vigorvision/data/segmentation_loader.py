import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
import warnings


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic/instance segmentation.
    Expects image-mask pairs with identical file names and appropriate directories.
    Supports optional resizing, class remapping, normalization, and custom transforms.
    """

    def __init__(self, image_dir, mask_dir, transform=None, img_size=(512, 512), num_classes=None, remap_classes=None):
        """
        Args:
            image_dir (str): Directory with input images.
            mask_dir (str): Directory with ground truth masks.
            transform (callable, optional): Optional transform applied on image and mask.
            img_size (tuple): Desired output size (width, height).
            num_classes (int, optional): Total number of classes (for sanity check).
            remap_classes (dict, optional): Dictionary for remapping class values in mask.
        """
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)
        )
        self.mask_paths = sorted(
            glob.glob(os.path.join(mask_dir, "**", "*.*"), recursive=True)
        )
        self.transform = transform
        self.img_size = img_size
        self.num_classes = num_classes
        self.remap_classes = remap_classes

        if len(self.image_paths) != len(self.mask_paths):
            warnings.warn("Mismatch between number of images and masks!")

    def __len__(self):
        return len(self.image_paths)

    def load_image_mask(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")  # single channel class mask

        image = image.resize(self.img_size, resample=Image.BILINEAR)
        mask = mask.resize(self.img_size, resample=Image.NEAREST)

        mask_np = np.array(mask, dtype=np.uint8)

        # Remap classes if necessary
        if self.remap_classes:
            remapped_mask = np.zeros_like(mask_np, dtype=np.uint8)
            for original_class, new_class in self.remap_classes.items():
                remapped_mask[mask_np == original_class] = new_class
            mask_np = remapped_mask

        if self.num_classes and mask_np.max() >= self.num_classes:
            warnings.warn(f"Mask contains unknown class: {mask_np.max()} ≥ {self.num_classes}")

        return image, mask_np

    def __getitem__(self, index):
        image, mask_np = self.load_image_mask(index)

        if self.transform:
            image, mask_np = self.transform(image, mask_np)

        image = T.ToTensor()(image)
        mask = torch.from_numpy(mask_np).long()  # For segmentation loss (e.g., CrossEntropy)

        return image, mask


def collate_segmentation(batch):
    """
    Collate function for segmentation datasets.
    Supports variable-size batching if needed.
    """
    images, masks = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks


def get_segmentation_loader(
    image_dir,
    mask_dir,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    transform=None,
    img_size=(512, 512),
    num_classes=None,
    remap_classes=None
):
    """
    Get a DataLoader for semantic segmentation tasks.
    """
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        img_size=img_size,
        num_classes=num_classes,
        remap_classes=remap_classes
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_segmentation,
    )
