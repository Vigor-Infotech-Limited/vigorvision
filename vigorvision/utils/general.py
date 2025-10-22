import os
import random
import shutil
import logging
import time
import math
import yaml
import torch
import numpy as np
import cv2
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from typing import Union, Tuple
import torchvision.transforms.functional as F
import platform

def seed_everything(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    """
    Set seeds for full reproducibility across:
        - Python random
        - NumPy
        - PyTorch (CPU & CUDA)
        - cuDNN settings
        - Environment variables
    Args:
        seed (int): Random seed value.
        deterministic (bool): Whether to set cuDNN to deterministic mode (slower but reproducible).
        benchmark (bool): Whether to enable cuDNN benchmark mode (faster but non-deterministic).
    
    Notes:
        - Setting `deterministic=True` disables `benchmark` internally.
        - Reproducibility may be affected by non-deterministic operations in certain layers.
        - On multi-GPU systems, all devices will be seeded.
    """
    logging.info(f"[SeedEverything] Seeding all libraries with seed={seed}, deterministic={deterministic}, benchmark={benchmark}")

    # 1. Python RNG
    random.seed(seed)

    # 2. NumPy RNG
    np.random.seed(seed)

    # 3. PyTorch RNG (CPU)
    torch.manual_seed(seed)

    # 4. PyTorch RNG (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # 5. cuDNN reproducibility settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark

    # 6. Extra env variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA determinism

    # 7. Prevent non-deterministic algorithms (PyTorch >= 1.8)
    try:
        torch.use_deterministic_algorithms(deterministic)
    except AttributeError:
        pass

    # 8. Platform-specific logs
    logging.info(f"[SeedEverything] Torch version: {torch.__version__}, CUDA: {torch.version.cuda}, Device: {platform.processor()}")

    # 9. Validate
    logging.info("[SeedEverything] Random int sample:", random.randint(0, 10000))
    logging.info("[SeedEverything] NumPy random sample:", np.random.randint(0, 10000))
    logging.info("[SeedEverything] Torch random sample:", torch.randint(0, 10000, (1,)).item())


def one_hot(labels, num_classes, smoothing=0.0, device=None, dtype=torch.float32):
    """
    Creates a one-hot encoded tensor with optional label smoothing.

    Args:
        labels (Tensor or list): Class indices, shape (N,) or scalar.
        num_classes (int): Number of classes.
        smoothing (float): Label smoothing factor in [0, 1).
                          0.1 means 10% of the label is distributed to other classes.
        device (torch.device or str, optional): Device for the output tensor.
        dtype (torch.dtype, optional): Data type for the output tensor.

    Returns:
        Tensor: One-hot encoded tensor of shape (N, num_classes) or (1, num_classes).
    """
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, device=device if device else None)

    if labels.ndim == 0:
        labels = labels.unsqueeze(0)  # scalar → batch of size 1

    device = device or labels.device
    N = labels.size(0)

    # Base one-hot
    one_hot_tensor = torch.zeros(N, num_classes, device=device, dtype=dtype)
    one_hot_tensor.scatter_(1, labels.view(-1, 1).long(), 1.0)

    if smoothing > 0.0:
        smooth_value = smoothing / num_classes
        one_hot_tensor = one_hot_tensor * (1.0 - smoothing) + smooth_value

    return one_hot_tensor

def set_seed(seed=42):
    """
    Ensures reproducibility by setting seeds for random, numpy, torch (cpu & cuda).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name=__name__, save_path=None):
    """
    Returns a configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if save_path:
        file_handler = logging.FileHandler(save_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def yaml_load(path):
    """
    Loads a YAML configuration file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def yaml_save(data, path):
    """
    Saves a dictionary to YAML file.
    """
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path, empty=False):
    """
    Creates directory if not exists. Optionally empties it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    elif empty:
        shutil.rmtree(path)
        os.makedirs(path)


def time_sync():
    """
    Returns a synchronized time stamp for CUDA or CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def increment_path(path, exist_ok=False, sep=''):
    """
    Automatically increments path like runs/exp -> runs/exp1 -> runs/exp2.
    """
    path = os.path.expanduser(path)
    if exist_ok or not os.path.exists(path):
        return path
    else:
        base, name = os.path.split(path)
        dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d.startswith(name)]
        matches = [int(d.replace(name, '').replace(sep, '')) for d in dirs if d.replace(name, '').replace(sep, '').isdigit()]
        n = max(matches) + 1 if matches else 1
        return os.path.join(base, f"{name}{sep}{n}")


def select_device(device=''):
    """
    Returns the appropriate torch.device (CUDA if available, else CPU).
    """
    if device.lower() == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device(device if device else 'cuda:0')



def check_dir(path):
    """
    Creates directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def xywh2xyxy(x):
    """
    Convert [x, y, w, h] to [x1, y1, x2, y2]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def make_anchors(
    grid_sizes,
    strides,
    anchor_grids,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generates anchors for all detection layers.

    Args:
        grid_sizes (List[Tuple[int, int]]): List of (height, width) for each feature map.
        strides (List[int]): Strides corresponding to each detection layer.
        anchor_grids (List[Tensor]): Predefined anchor shapes for each layer [na, 2].
        device (str): Device to place the anchors on.

    Returns:
        List[Tensor]: Anchors shaped as [1, na, h, w, 2] for each layer.
    """
    all_anchors = []

    for (h, w), stride, anchors in zip(grid_sizes, strides, anchor_grids):
        na = anchors.shape[0]  # number of anchors
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack((xv, yv), 2).to(device).float()  # [h, w, 2]

        grid = grid.view(1, 1, h, w, 2)  # [1, 1, h, w, 2]
        anchor_tensor = anchors.view(na, 1, 1, 2).to(device)  # [na, 1, 1, 2]

        anchor = grid + 0.5  # center offset
        anchor = anchor * stride
        anchor = anchor_tensor + 0 * anchor  # [na, 1, 1, 2]

        anchor = anchor.expand(-1, h, w, -1).permute(0, 2, 1, 3).unsqueeze(0)  # [1, na, h, w, 2]
        all_anchors.append(anchor)

    return all_anchors

def compute_num_params(model):
    """
    Computes the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_annealing_lr(base_lr, epoch, total_epochs, eta_min=1e-6):
    """
    Cosine Annealing Learning Rate Scheduler.
    """
    return eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def copy_files(src_folder, dst_folder, ext=(".py", ".yaml")):
    """
    Copies config/code files for reproducibility.
    """
    ensure_dir(dst_folder)
    for root, _, files in os.walk(src_folder):
        for f in files:
            if f.endswith(ext):
                rel_dir = os.path.relpath(root, src_folder)
                dst_dir = os.path.join(dst_folder, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(os.path.join(root, f), os.path.join(dst_dir, f))


import os
import torch

def colorstr(*input):
    """
    Returns a string with ANSI color formatting.
    Use it to enhance terminal logging like: print(colorstr('green', 'Training started'))

    Example:
        print(colorstr('blue', 'INFO:'), 'Training initialized...')
    """
    *colors, string = input if len(input) > 1 else ('blue', input[0])
    
    color_map = {
        'black': '\033[30m', 'red': '\033[31m',
        'green': '\033[32m', 'yellow': '\033[33m',
        'blue': '\033[34m', 'magenta': '\033[35m',
        'cyan': '\033[36m', 'white': '\033[37m',
        'bright': '\033[1m', 'dim': '\033[2m',
        'normal': '\033[22m', 'reset': '\033[0m'
    }

    prefix = ''.join(color_map.get(c.lower(), '') for c in colors)
    return f'{prefix}{string}{color_map["reset"]}'

import os
import torch


def load_model(checkpoint_path, model, optimizer=None, device=None):
    """
    Load model, optimizer, and metadata from a checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file (.pth or .pt)
        model (torch.nn.Module): Model instance to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to restore state
        device (str or torch.device, optional): Device to map the checkpoint to.
                                                Defaults to 'cuda' if available.

    Returns:
        dict: {
            'model': model,
            'optimizer': optimizer or None,
            'epoch': int,
            'is_best': bool,
            'class_names': list
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[INFO] Model weights loaded successfully.")
    else:
        raise KeyError("Checkpoint missing 'model_state_dict'.")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("[INFO] Optimizer state loaded successfully.")
    elif optimizer is not None:
        print("[WARN] No optimizer state found in checkpoint.")

    # Metadata
    epoch = checkpoint.get("epoch", 0)
    is_best = checkpoint.get("is_best", False)
    class_names = checkpoint.get("class_names", None)

    print(f"[INFO] Loaded checkpoint at epoch {epoch}.")
    print(f"[INFO] Best model: {is_best}")
    if class_names:
        print(f"[INFO] Classes: {class_names}")

    return {
        "model": model.to(device),
        "optimizer": optimizer,
        "epoch": epoch,
        "is_best": is_best,
        "class_names": class_names
    }

def save_model(
    model,
    optimizer,
    epoch,
    save_dir,
    is_best=False,
    best_map=None,
    filename="last.pt",
    class_names=None,
):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Current epoch.
        save_dir (str): Directory to save the model.
        is_best (bool): Whether this is the best model (by mAP).
        best_map (float): Best mAP value.
        filename (str): Filename to use (e.g. last.pt, best.pt).
        class_names (list or None): Class names to save with model.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare state dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_map": best_map,
        "class_names": class_names,
    }

    # Save last.pt, interrupted.pt, etc.
    model_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Optionally save best.pt separately
    if is_best:
        best_path = os.path.join(save_dir, "best.pt")
        torch.save(checkpoint, best_path)
        print(f"[INFO] Best model (mAP={best_map:.4f}) saved to {best_path}")



def load_image(
    source: Union[str, Path, np.ndarray, torch.Tensor, Image.Image],
    img_size: Tuple[int, int] = None,
    keep_aspect: bool = True,
    to_tensor: bool = True,
    normalize: bool = False,
    backend: str = "cv2"
):
    """
    Load an image from various sources into a standardized format.

    Args:
        source: Path, ndarray, torch.Tensor, or PIL.Image
        img_size: (width, height) to resize, or None for original size
        keep_aspect: Keep aspect ratio when resizing
        to_tensor: Return torch.Tensor in CHW format if True, else np.ndarray in HWC
        normalize: Normalize to 0-1 range if True
        backend: 'cv2' or 'pil' for reading

    Returns:
        img: torch.Tensor [C,H,W] or np.ndarray [H,W,C] RGB
        original_shape: (H, W)
        ratio: resize ratio
        padding: (pad_x, pad_y)
    """
    # -------------------------
    # 1️⃣ Read Image
    # -------------------------
    if isinstance(source, (str, Path)):
        path = str(source)
        if backend.lower() == "cv2":
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif backend.lower() == "pil":
            img = Image.open(path)
            try:
                # Handle EXIF orientation
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation, None)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except Exception:
                pass
            img = img.convert("RGB")
            img = np.array(img)
        else:
            raise ValueError("backend must be 'cv2' or 'pil'")

    elif isinstance(source, Image.Image):
        img = np.array(source.convert("RGB"))
    elif isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:  # Gray
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif isinstance(source, torch.Tensor):
        img = source.cpu().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW to HWC
            img = np.transpose(img, (1, 2, 0))
    else:
        raise TypeError("Unsupported source type")

    original_shape = img.shape[:2]  # (H, W)

    # -------------------------
    # 2️⃣ Resize (optional)
    # -------------------------
    ratio, pad = (1.0, (0, 0))
    if img_size is not None:
        target_w, target_h = img_size
        if keep_aspect:
            h, w = original_shape
            ratio = min(target_w / w, target_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            img = cv2.copyMakeBorder(resized, pad_y, target_h - new_h - pad_y,
                                     pad_x, target_w - new_w - pad_x,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
            pad = (pad_x, pad_y)
        else:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # -------------------------
    # 3️⃣ Normalize + Convert
    # -------------------------
    if normalize:
        img = img.astype(np.float32) / 255.0

    if to_tensor:
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # HWC → CHW
        if normalize is False:
            img = img.float()

    return img, original_shape, ratio, pad


def test_load_image():
    img, orig_shape, ratio, pad = load_image("sample.jpg", img_size=(640, 640), keep_aspect=True, normalize=True)
    print(f"Original shape: {orig_shape}, ratio: {ratio}, pad: {pad}, final shape: {tuple(img.shape)}")

class ToTensor:
    """
    Converts PIL Image and mask to Tensor. Mask is long (int64) for CE loss.
    """

    def __call__(self, image, mask):
        return F.to_tensor(image), torch.from_numpy(np.array(mask, dtype=np.int64))



if __name__ == "__main__":
    test_load_image()
