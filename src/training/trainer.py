"""Training loop and engine for vehicle damage detection models."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.faster_rcnn import FasterRCNNModel
from src.tracking.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainerConfig:
    """Trainer configuration settings."""

    epochs: int = 40
    batch_size: int = 2
    image_size: int = 512
    learning_rate: float = 0.005
    weight_decay: float = 0.0005
    momentum: float = 0.9
    optimizer: str = "SGD"
    backbone_lr_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Learning rate scheduler
    scheduler_type: str = "step"  # "step" hoặc "cosine"
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.1

    # Mixed Precision (AMP) & Gradient Accumulation
    use_amp: bool = True
    gradient_accumulation_steps: int = 2
    clip_grad_norm: float = 10.0

    # Model image sizing
    min_size: int = 512
    max_size: int = 512

    # Checkpoint & logging
    checkpoint_dir: str = "artifacts/checkpoints"
    save_best_only: bool = True
    max_train_batches: int | None = None  # Giới hạn số batch mỗi epoch (cho test/dry-run)
    max_val_batches: int | None = None
    log_interval: int = 10


class Trainer:
    """Trainer engine supporting PyTorch detection models, AMP, Gradient Accumulation, and MLflow."""

    def __init__(
        self,
        model: FasterRCNNModel,
        config: TrainerConfig | None = None,
        train_loader: DataLoader[Any] | None = None,
        val_loader: DataLoader[Any] | None = None,
        tracker: MLflowTracker | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tracker = tracker

        self.checkpoint_path = Path(self.config.checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 1. Optimizer với differential learning rates cho backbone vs head
        param_groups = self.model.get_parameter_groups(
            lr=self.config.learning_rate,
            backbone_lr_ratio=self.config.backbone_lr_ratio,
        )
        if self.config.optimizer.upper() == "SGD":
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.upper() == "ADAMW":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: '{self.config.optimizer}'. Choose 'SGD' or 'AdamW'.")

        # 2. Learning rate scheduler
        if self.config.scheduler_type.lower() == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler_type.lower() == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        else:
            raise ValueError(f"Unsupported scheduler_type: '{self.config.scheduler_type}'. Choose 'step' or 'cosine'.")

        # 3. Mixed Precision (AMP) & Gradient Accumulation
        is_cuda = self.device.type == "cuda"
        self.use_amp = bool(self.config.use_amp and is_cuda)
        self.scaler = torch.amp.GradScaler("cuda" if is_cuda else "cpu", enabled=self.use_amp)
        self.accumulation_steps = max(1, self.config.gradient_accumulation_steps)

        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train model for a single epoch with Mixed Precision (AMP) and Gradient Accumulation."""
        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training.")

        self.model.train()
        total_loss = 0.0
        loss_components: dict[str, float] = {}
        num_batches = 0
        start_time = time.time()
        self.optimizer.zero_grad()

        num_total_batches = len(self.train_loader)
        if self.config.max_train_batches:
            num_total_batches = min(num_total_batches, self.config.max_train_batches)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if self.config.max_train_batches and batch_idx >= self.config.max_train_batches:
                break

            # Chuyển dữ liệu sang device
            images = [img.to(self.device) for img in images]
            targets = [{k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            # Forward pass under AMP autocast
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Faster R-CNN tự động tính multi-task loss khi model.train()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # Scale loss theo gradient accumulation steps
                loss_scaled = losses / self.accumulation_steps

            # Backward pass với GradScaler
            self.scaler.scale(loss_scaled).backward()

            # Optimizer step mỗi accumulation_steps hoặc khi hết batch trong epoch
            is_step = ((batch_idx + 1) % self.accumulation_steps == 0) or ((batch_idx + 1) == num_total_batches)
            if is_step:
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += losses.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

            num_batches += 1

            if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == num_total_batches:
                avg_batch_loss = total_loss / num_batches
                logger.info(
                    "Epoch [%d/%d] Batch [%d/%d] - Loss: %.4f",
                    epoch,
                    self.config.epochs,
                    batch_idx + 1,
                    num_total_batches,
                    avg_batch_loss,
                )

        self.lr_scheduler.step()

        elapsed = time.time() - start_time
        avg_train_loss = total_loss / max(1, num_batches)
        current_lr = self.optimizer.param_groups[-1]["lr"]
        metrics = {
            "train_loss": avg_train_loss,
            "train_time_sec": elapsed,
            "lr": current_lr,
        }
        for k, v in loss_components.items():
            metrics[f"train_{k}"] = v / max(1, num_batches)

        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set."""
        if self.val_loader is None:
            return {}

        # Faster R-CNN trong torchvision khi model.eval() chỉ trả về detections.
        # Để tính validation loss, ta tạm thời để model.train() nhưng không tính grad và không cập nhật weights!
        self.model.train()
        total_val_loss = 0.0
        val_components: dict[str, float] = {}
        num_batches = 0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.val_loader):
            if self.config.max_val_batches and batch_idx >= self.config.max_val_batches:
                break

            images = [img.to(self.device) for img in images]
            targets = [{k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            total_val_loss += losses.item()
            for k, v in loss_dict.items():
                val_components[k] = val_components.get(k, 0.0) + v.item()

            num_batches += 1

        elapsed = time.time() - start_time
        avg_val_loss = total_val_loss / max(1, num_batches)

        metrics = {
            "val_loss": avg_val_loss,
            "val_time_sec": elapsed,
        }
        for k, v in val_components.items():
            metrics[f"val_{k}"] = v / max(1, num_batches)

        return metrics

    def train(self) -> dict[str, Any]:
        """Execute the full training loop across configured epochs."""
        logger.info(
            "Starting training on %s: %d epochs, batch_size=%d, lr=%.4f (AMP=%s, GradAccum=%d)",
            self.device,
            self.config.epochs,
            self.config.batch_size,
            self.config.learning_rate,
            self.use_amp,
            self.accumulation_steps,
        )

        if self.tracker:
            try:
                self.tracker.log_params({
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "image_size": self.config.image_size,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "momentum": self.config.momentum,
                    "optimizer": self.config.optimizer,
                    "scheduler_type": self.config.scheduler_type,
                    "scheduler_step_size": self.config.scheduler_step_size,
                    "scheduler_gamma": self.config.scheduler_gamma,
                    "use_amp": self.use_amp,
                    "gradient_accumulation_steps": self.accumulation_steps,
                    "device": str(self.device),
                })
            except Exception as e:
                logger.debug("Parameters may already be logged in MLflow run: %s", e)

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # 1. Train 1 epoch
            train_metrics = self.train_one_epoch(epoch)

            # 2. Evaluate
            val_metrics = self.evaluate() if self.val_loader else {}

            epoch_time = time.time() - epoch_start
            combined_metrics = {**train_metrics, **val_metrics, "epoch_duration_sec": epoch_time}
            self.history.append(combined_metrics)

            # 3. Log to MLflow
            if self.tracker:
                self.tracker.log_metrics(combined_metrics, step=epoch)

            val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
            logger.info(
                "Epoch [%d/%d] Complete | Train Loss: %.4f | Val Loss: %.4f | LR: %.6f | Time: %.1fs",
                epoch,
                self.config.epochs,
                train_metrics["train_loss"],
                val_loss,
                train_metrics.get("lr", self.config.learning_rate),
                epoch_time,
            )

            # 4. Lưu Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_path / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": self.best_val_loss,
                        "config": self.config,
                    },
                    best_path,
                )
                logger.info("Saved new best model checkpoint to: %s", best_path)

        latest_path = self.checkpoint_path / "latest_model.pth"
        torch.save(
            {
                "epoch": self.config.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            latest_path,
        )

        if self.tracker:
            self.tracker.log_artifact(str(latest_path), artifact_path="checkpoints")

        return {
            "epochs": float(self.config.epochs),
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.history[-1]["train_loss"] if self.history else 0.0,
            "history": self.history,
        }
