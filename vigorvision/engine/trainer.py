import os
import time
import wandb
import torch
import shutil
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from vigorvision.utils.metrics import compute_map as mAP, AverageMeter
from vigorvision.utils.general import increment_path, save_model
from vigorvision.utils.autoanchor import check_anchors
from vigorvision.utils.iou import bbox_iou
from vigorvision.nn.loss import ComputeLoss
from vigorvision.engine.validator import Validator
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        val_dataloader,
        device,
        optimizer,
        epochs=100,
        lr_scheduler=None,
        warmup_scheduler=None,
        save_dir="runs/train",
        log_dir="logs",
        project="VigorVision",
        run_name="exp",
        classes=11,
        class_names=None,
        amp=True,
        log_wandb=True,
        log_tensorboard=True,
        gradient_accumulation_steps=1,
        gradient_clipping=5.0,
        early_stopping_patience=15,
        check_anchor_every=5,
        resume=False,
        loss_fn=None,
    ):
        self.device = device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.class_names = class_names if class_names else [str(i) for i in range(classes)]

        self.lr_scheduler = lr_scheduler
        self.warmup_scheduler = warmup_scheduler
        self.gradient_clipping = gradient_clipping
        self.save_dir = increment_path(save_dir, run_name)
        self.classes = classes
        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.check_anchor_every = check_anchor_every
        default_hyp = {
            "box": 0.05,
            "cls": 0.5,
            "obj": 1.0,
            "fl_gamma": 1.5,
            "label_smoothing": 0.0
        }
        self.loss_fn = loss_fn if loss_fn else ComputeLoss(model, hyp=default_hyp)


        os.makedirs(self.save_dir, exist_ok=True)
        self.scaler = GradScaler(enabled=amp)

        self.log_wandb = log_wandb
        self.log_tensorboard = log_tensorboard
        if self.log_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, log_dir))

        if self.log_wandb:
            wandb.init(project=project, name=run_name, config={"epochs": epochs})

        self.best_map = 0.0
        self.early_stopping_counter = 0
        self.start_epoch = 0

        if resume:
            self._auto_resume()

    def _auto_resume(self):
        last_ckpt = os.path.join(self.save_dir, "last.pt")
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_map = ckpt.get("best_map", 0.0)
            print(f"[INFO] Resumed from checkpoint: {last_ckpt} (epoch {self.start_epoch})")
        else:
            print("[INFO] No checkpoint found, starting from scratch.")

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter('train_loss')
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.epochs}")
        self.optimizer.zero_grad()

        for i, (images, targets) in pbar:
            images = images.to(self.device)
            targets = [t.to(self.device) for t in targets]

            with autocast(enabled=self.amp):
                preds = self.model(images)
                loss, loss_items = self.loss_fn(preds, targets)

            self.scaler.scale(loss / self.gradient_accumulation_steps).backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            losses.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=losses.avg)

            if self.warmup_scheduler:
                self.warmup_scheduler.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return losses.avg

    def validate(self):
        self.model.eval()
        preds_all, targets_all = [], []

        with torch.no_grad():
            for images, targets in tqdm(self.val_dataloader, desc="Validating"):
                images = images.to(self.device)
                targets = [t.to(self.device) for t in targets]

                with autocast(enabled=self.amp):
                    preds = self.model(images)

                preds_all.extend(preds)
                targets_all.extend(targets)

        map_overall, per_class_map = mAP(preds_all, targets_all, num_classes=self.classes, iou_thresholds=torch.arange(0.5, 0.96, 0.05))
        return map_overall, per_class_map

    

    def train(self):
        try:
            start_time = time.time()

            for epoch in range(self.start_epoch, self.epochs):
                if epoch % self.check_anchor_every == 0:
                    check_anchors(self.model, self.dataloader)

                train_loss = self.train_one_epoch(epoch)

                # Validator Integration
                validator = Validator(
                    model=self.model,
                    dataloader=self.val_dataloader,
                    device=self.device,
                    anchors=self.model.anchors if hasattr(self.model, "anchors") else None,
                    num_classes=self.classes,
                    use_amp=self.amp,
                    save_dir=self.save_dir,
                )
                val_stats = validator.evaluate()

                # Extract metrics
                val_loss = val_stats["val/total_loss"]
                box_loss = val_stats["val/box_loss"]
                cls_loss = val_stats["val/cls_loss"]
                obj_loss = val_stats["val/obj_loss"]
                map_50 = val_stats["metrics/mAP_0.5"]
                map_50_95 = val_stats["metrics/mAP_0.5:0.95"]

                # ======== Logging ========
                if self.log_tensorboard:
                    self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                    self.tb_writer.add_scalar("Loss/val_total", val_loss, epoch)
                    self.tb_writer.add_scalar("Loss/val_box", box_loss, epoch)
                    self.tb_writer.add_scalar("Loss/val_cls", cls_loss, epoch)
                    self.tb_writer.add_scalar("Loss/val_obj", obj_loss, epoch)
                    self.tb_writer.add_scalar("mAP@0.5", map_50, epoch)
                    self.tb_writer.add_scalar("mAP@0.5:0.95", map_50_95, epoch)

                if self.log_wandb:
                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss_total": val_loss,
                        "val_loss_box": box_loss,
                        "val_loss_cls": cls_loss,
                        "val_loss_obj": obj_loss,
                        "mAP@0.5": map_50,
                        "mAP@0.5:0.95": map_50_95,
                        "epoch": epoch,
                    })

                # ======== Save Model ========
                save_model(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.save_dir,
                    is_best=False,
                    best_map=self.best_map,
                    filename="last.pt",
                    class_names=self.class_names
                )

                if map_50 > self.best_map:
                    self.best_map = map_50
                    save_model(
                        self.model,
                        self.optimizer,
                        epoch,
                        self.save_dir,
                        is_best=True,
                        best_map=self.best_map,
                        class_names=self.class_names
                    )
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print("Early stopping triggered.")
                        break
                
            total_time = time.time() - start_time
            print(f"\nTraining completed in {(total_time / 60):.2f} minutes.")

        except KeyboardInterrupt:
            print("[INFO] Training interrupted manually. Saving current checkpoint...")
            save_model(self.model, self.optimizer, epoch, self.save_dir, is_best=False, best_map=self.best_map, filename="interrupted.pt", class_names=self.class_names)
        finally:
            if self.log_tensorboard:
                self.tb_writer.close()
            if self.log_wandb:
                wandb.finish()
