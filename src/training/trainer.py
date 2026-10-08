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

    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    backbone_lr_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "artifacts/checkpoints"
    save_best_only: bool = True
    max_train_batches: int | None = None  # Giới hạn số batch mỗi epoch (cho test/dry-run)
    max_val_batches: int | None = None
    log_interval: int = 10


class Trainer:
    """Trainer engine supporting PyTorch detection models and MLflow experiment logging."""

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
        self.optimizer = torch.optim.SGD(
            param_groups,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        # 2. Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )

        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train model for a single epoch."""
        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training.")

        self.model.train()
        total_loss = 0.0
        loss_components: dict[str, float] = {}
        num_batches = 0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if self.config.max_train_batches and batch_idx >= self.config.max_train_batches:
                break

            # Chuyển dữ liệu sang device
            images = [img.to(self.device) for img in images]
            targets = [{k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            # Forward pass: Faster R-CNN tự động trả về dict các loss khi ở mode train()
            loss_dict = self.model(images, targets)

            # Tính tổng loss
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()

            # Gradient clipping để đảm bảo ổn định
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            total_loss += losses.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

            num_batches += 1

            if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_batch_loss = total_loss / num_batches
                logger.info(
                    "Epoch [%d/%d] Batch [%d/%d] - Loss: %.4f",
                    epoch,
                    self.config.epochs,
                    batch_idx + 1,
                    len(self.train_loader),
                    avg_batch_loss,
                )

        self.lr_scheduler.step()

        elapsed = time.time() - start_time
        avg_train_loss = total_loss / max(1, num_batches)
        metrics = {
            "train_loss": avg_train_loss,
            "train_time_sec": elapsed,
            "lr": self.optimizer.param_groups[-1]["lr"],
        }
        for k, v in loss_components.items():
            metrics[f"train_{k}"] = v / max(1, num_batches)

        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set."""
        if self.val_loader is None:
            return {}

        # Chú ý: Faster R-CNN trong torchvision khi model.eval() chỉ trả về detections.
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
            "Starting training on %s: %d epochs, initial lr=%.4f",
            self.device,
            self.config.epochs,
            self.config.learning_rate,
        )

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
                "Epoch %d Complete | Train Loss: %.4f | Val Loss: %.4f | Time: %.1fs",
                epoch,
                train_metrics["train_loss"],
                val_loss,
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
