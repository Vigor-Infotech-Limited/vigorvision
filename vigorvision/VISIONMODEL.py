import os

import yaml
import json
import torch

from vigorvision.engine.trainer import Trainer
from vigorvision.data.dataloader import get_dataloader as create_dataloader
from vigorvision.utils.general import set_seed, get_logger
from vigorvision.utils.general import colorstr

logger = get_logger()

def load_class_names(source):
    """
    Load class names from a .txt file, .json file, or list.
    """
    if isinstance(source, list):
        return source
    elif os.path.isfile(source):
        if source.endswith(".txt"):
            with open(source, "r") as f:
                return [line.strip() for line in f if line.strip()]
        elif source.endswith(".json"):
            with open(source, "r") as f:
                return json.load(f)
    raise ValueError("Unsupported class_names format. Use list, .txt, or .json file.")



def Train_Model(data : str,
                epochs : int,
                batch_size : int,
                workers : int,
                img_size : int,
                project_name : str,
                variant : str,
                project_dir = 'train/train',
                device = 'cuda',
                wandb = True,
                seed = 30,
                amp = True,
                tensorboard = True,
                run_name = 'default',
                resume = False):
    
    from vigorvision.models.build import build_model
    set_seed(seed)

    logger.info("Thanks for choosing Vigor Infotech Limited 🌍")
    logger.info("🚀 Training pipeline initiated...")
    # === Load dataset paths from data.yaml ===
    assert os.path.exists(data), f"{data} not found!"

    with open(data, 'r') as f:
        data_yaml = yaml.safe_load(f) 
    train_dir = data_yaml["train"]
    val_dir = data_yaml["val"]

    # === Load class names ===
    class_names = data_yaml['names']
    num_classes = len(class_names)

    # === Device ===
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(colorstr("Device:"), device)

    # === Create dataloaders ===
    logger.info("Loading Train Data")
    train_loader = create_dataloader(
        img_dir=train_dir,
        img_size=img_size,
        batch_size=batch_size,
        classes=num_classes,
        shuffle=True,
        num_workers=workers
    )

    logger.info("Loading Validator Data")

    val_loader = create_dataloader(
        img_dir=val_dir,
        img_size=img_size,
        batch_size=batch_size,
        classes=num_classes,
        shuffle=False,
        num_workers=workers
    )

    print("Total train images:", len(train_loader.dataset))
    print("Entering Loop")
    
    logger.info("🛠️BUILDING MODEL")
    # === Build model ===
    model = build_model(dataset=train_loader, variant=variant, num_classes=num_classes)

    logger.info("✅ Model built successfully!")
    
    # === Optimizer ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

    # === Trainer ===
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        optimizer=optimizer,
        epochs=epochs,
        amp=amp,
        log_wandb=wandb,
        log_tensorboard=tensorboard,
        project=project_name,
        run_name=run_name,
        resume=resume,
        classes=num_classes,
        class_names=class_names
    )

    logger.info("🎉 Training Started!")
    # === Train ===
    trainer.train()

    logger.info('✅ Training Finished!')

