import os
import time
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from vigorvision.utils.general import seed_everything
from vigorvision.utils.metrics import compute_segmentation_metrics


class SegmentationTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        num_epochs=50,
        save_dir="checkpoints/",
        project_name="VigorVision-Segmentation",
        use_wandb=False,
        log_dir="runs/",
        device=None,
        patience=10,
        seed=42,
    ):
        # Basic config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.project_name = project_name
        self.use_wandb = use_wandb
        self.patience = patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        if self.use_wandb:
            wandb.init(project=self.project_name, config={
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "optimizer": optimizer.__class__.__name__,
                "scheduler": scheduler.__class__.__name__ if scheduler else "None",
                "loss_fn": criterion.__class__.__name__,
            })

        seed_everything(seed)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for i, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if self.use_wandb:
                wandb.log({"train/loss_batch": loss.item(), "epoch": epoch})

        avg_loss = running_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        running_loss = 0.0
        all_preds, all_masks = [], []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).detach().cpu()
                all_preds.append(preds)
                all_masks.append(masks.detach().cpu())

        avg_loss = running_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Val", avg_loss, epoch)

        if self.use_wandb:
            wandb.log({"val/loss": avg_loss, "epoch": epoch})

        # Metrics
        metrics = compute_segmentation_metrics(torch.cat(all_preds), torch.cat(all_masks))
        for k, v in metrics.items():
            self.writer.add_scalar(f"Metrics/{k}", v, epoch)
            if self.use_wandb:
                wandb.log({f"val/{k}": v, "epoch": epoch})

        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        filename = os.path.join(
            self.save_dir, f"{'best' if is_best else 'epoch_' + str(epoch)}_model.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def fit(self):
        print(f"Starting training for {self.num_epochs} epochs on {self.device}")

        for epoch in range(1, self.num_epochs + 1):
            start = time.time()
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            end = time.time()

            print(
                f"[Epoch {epoch}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {end - start:.2f}s"
            )

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss if val_loss is not None else train_loss)

            # Checkpoint
            is_best = val_loss is not None and val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Early stopping
            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered.")
                break

        self.writer.close()
        if self.use_wandb:
            wandb.finish()
