# vigorvision/train_seg.py

import os
import torch
from vigorvision.engine.trainer_seg import train_segmentation
from vigorvision.models.vision.segment import build_segmentation_model
from vigorvision.data.segmentation_loader import get_segmentation_loaders
from vigorvision.utils.transform_segmentation import (
    SegCompose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomBrightnessContrast,
    CLAHEEqualization, ElasticTransform
)
from vigorvision.utils.general import seed_everything

from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_transform(train=True, image_size=512):
    if train:
        return SegCompose([
            Resize((image_size, image_size)),
            RandomHorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness=0.3, contrast=0.3),
            CLAHEEqualization(clip_limit=2.5),
            ElasticTransform(alpha=60, sigma=6),
            ToTensor(),
            Normalize()
        ])
    else:
        return SegCompose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize()
        ])


def train_segmentation_model(
    model_name: str,
    num_classes: int,
    train_path: str,
    val_path: str,
    image_size: int = 512,
    batch_size: int = 8,
    num_epochs: int = 80,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    optimizer: str = 'adam',
    scheduler: str = 'cosine',
    num_workers: int = 4,
    save_dir: str = 'runs/segmentation/exp1',
    seed: int = 42,
    early_stopping: bool = True,
    early_stop_patience: int = 10,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = 'VigorVision-Segmentation',
    wandb_run_name: str = None,
):
    """
    Main segmentation training function for VigorVision.
    """
    # Seed everything
    seed_everything(seed)

    os.makedirs(save_dir, exist_ok=True)

    # Logging setup
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard")) if use_tensorboard else None
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or os.path.basename(save_dir),
            config={
                "model": model_name,
                "num_classes": num_classes,
                "image_size": image_size,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "lr": lr,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "early_stopping": early_stopping,
                "seed": seed
            }
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("[WARN] wandb not installed. Skipping wandb logging.")

    # Dataloaders
    train_loader, val_loader = get_segmentation_loaders(
        train_path=train_path,
        val_path=val_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=get_transform(train=True, image_size=image_size),
        val_transform=get_transform(train=False, image_size=image_size)
    )

    # Model
    model = build_segmentation_model(model_name, num_classes).to(device)

    # Train
    train_segmentation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'num_epochs': num_epochs,
            'lr': lr,
            'device': device,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'early_stopping': early_stopping,
            'early_stop_patience': early_stop_patience,
            'save_dir': save_dir,
            'use_tensorboard': use_tensorboard,
            'use_wandb': use_wandb
        },
        tb_writer=tb_writer,
    )

    # Close loggers
    if use_tensorboard:
        tb_writer.close()
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
