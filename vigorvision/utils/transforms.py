# vigorvision/utils/transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import numpy as np
import random
import torch
import cv2

import torch
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

def get_detection_transforms(img_size=(640, 640), is_train=True):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size[1], width=img_size[0], scale=(0.7, 1.2), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            A.MotionBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            A.CLAHE(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(height=img_size[1], width=img_size[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_segmentation_transforms(img_size=(640, 640), is_train=True):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size[1], width=img_size[0], scale=(0.7, 1.2), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size[1], width=img_size[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


# For additional tensor-level postprocessing or debugging
def post_transform(tensor_img):
    """
    Optional tensor-level transform to reverse normalization for visualization.
    """
    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor_img * inv_std + inv_mean


def visualize_transforms(image, bboxes=None, masks=None, class_labels=None):
    """
    Utility for visualizing augmented outputs.
    """
    import matplotlib.pyplot as plt

    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    if bboxes and class_labels:
        for bbox, label in zip(bboxes, class_labels):
            x_center, y_center, w, h = bbox
            x1 = int((x_center - w / 2) * image.shape[1])
            y1 = int((y_center - h / 2) * image.shape[0])
            x2 = int((x_center + w / 2) * image.shape[1])
            y2 = int((y_center + h / 2) * image.shape[0])
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor='red', facecolor='none', linewidth=2))
            ax.text(x1, y1 - 5, f'{label}', color='white', fontsize=10, backgroundcolor='red')

    if masks is not None:
        for mask in masks:
            ax.imshow(mask.squeeze(), alpha=0.3)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# vigorvision/utils/transforms.py

import cv2
import torch
import numpy as np

def val_preprocess(
    img_path,
    img_size=640,
    stride=32,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    device="cpu"
):
    """
    Preprocess an image for validation/inference in VigorVision.

    Args:
        img_path (str or np.ndarray): Path to image file or already loaded BGR image.
        img_size (int): Target image size (square).
        stride (int): Model stride to ensure dimensions are multiple of stride.
        mean (tuple): Normalization mean (ImageNet default).
        std (tuple): Normalization std (ImageNet default).
        device (str): Torch device.

    Returns:
        img_tensor (torch.Tensor): Preprocessed tensor [1, 3, H, W].
        img_resized (np.ndarray): Resized image for visualization (BGR).
        ratio_pad (tuple): (ratio, pad) used for reversing letterbox.
    """
    # Load image
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        assert img is not None, f"Image Not Found: {img_path}"
    else:
        img = img_path.copy()

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox resize (keeps aspect ratio, pads to multiple of stride)
    shape = img_rgb.shape[:2]  # (h, w)
    r = min(img_size / shape[0], img_size / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]  # padding
    dw /= 2
    dh /= 2

    # Resize
    img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad evenly
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Convert to float32 & normalize
    img_padded = img_padded.astype(np.float32) / 255.0
    img_padded = (img_padded - mean) / std

    # HWC -> CHW
    img_padded = img_padded.transpose(2, 0, 1)

    # To tensor
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).to(device).float()

    return img_tensor, img_resized, ((r, r), (dw, dh))
def val_postprocess(predictions, ratio_pad, img_size=640):
    """
    Postprocess model predictions after validation/inference.
    Args:
        predictions (torch.Tensor): Model output tensor [N, num_classes + 5, H, W].
        ratio_pad (tuple): (ratio, pad) used for letterbox resizing.
        img_size (int): Original image size.
    Returns:
        results (list): List of processed predictions for each image.
    """
    ratio, pad = ratio_pad
    results = []

    for pred in predictions:
        # Remove padding
        pred[:, 0] = (pred[:, 0] - pad[0]) / ratio[0]
        pred[:, 1] = (pred[:, 1] - pad[1]) / ratio[1]
        pred[:, 2] = pred[:, 2] / ratio[0]
        pred[:, 3] = pred[:, 3] / ratio[1]

        # Filter out low confidence detections
        pred = pred[pred[:, 4] > 0.25]

        # Convert to numpy and append to results
        results.append(pred.cpu().numpy())

    return results
def visualize_predictions(image, predictions, class_names=None):
    """
    Visualize model predictions on an image.
    Args:
        image (np.ndarray): Original image in BGR format.
        predictions (list): List of predictions for each image.
        class_names (list): List of class names for labels.
    """
    import matplotlib.pyplot as plt

    # Convert BGR to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    for pred in predictions:
        for box in pred:
            x1, y1, x2, y2, conf, cls = box[:6]
            label = f"{class_names[int(cls)]} {conf:.2f}" if class_names else f"Class {int(cls)} {conf:.2f}"

            # Draw bounding box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               edgecolor='red', facecolor='none', linewidth=2))
            plt.text(x1, y1 - 5, label, color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)  # only scale down, do not scale up

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width/height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def load_image(source):
    """
    Loads an image from path, URL, NumPy, PIL, or torch.Tensor.
    """
    if isinstance(source, (str, Path)):
        source = str(source)
        if source.startswith(('http://', 'https://')):
            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
        else:
            img = cv2.imread(source)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(source, Image.Image):  # PIL
        img = np.array(source.convert("RGB"))
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:  # grayscale
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        img = source
    elif torch.is_tensor(source):
        source = source.detach().cpu().numpy()
        if source.ndim == 3 and source.shape[0] in [1, 3]:  # CHW to HWC
            img = np.transpose(source, (1, 2, 0))
        else:
            img = source
    else:
        raise TypeError(f"Unsupported image type: {type(source)}")
    return img


def preprocess_image(source, img_size=640, stride=32, device=None, half=False):
    """
    Beast-level image preprocessing for inference.

    Args:
        source: Image source (path, URL, PIL, NumPy, torch.Tensor)
        img_size: Target size for model input
        stride: Model stride for letterbox alignment
        device: torch.device for final tensor
        half: Convert to float16 for faster inference (only if supported)
    """
    # Load image
    img = load_image(source)

    # Letterbox resize
    img, ratio, pad = letterbox(img, new_shape=img_size, stride=stride)

    # Convert HWC to CHW, BGR to RGB if needed
    img = img.transpose((2, 0, 1))  # to CHW
    img = np.ascontiguousarray(img)

    # Convert to torch
    img_tensor = torch.from_numpy(img).to(device if device else "cpu")
    img_tensor = img_tensor.float()  # convert to float32
    img_tensor /= 255.0  # normalize to 0-1
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # add batch dim

    # Half precision if requested
    if half:
        img_tensor = img_tensor.half()

    return img_tensor, img, ratio, pad
