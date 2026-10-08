import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.model_factory import create_model
from src.training.trainer import Trainer, TrainerConfig


class DummyDetectionDataset(Dataset):
    def __init__(self, num_samples: int = 4):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        img = torch.rand(3, 256, 256)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        return img, target


def dummy_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def test_trainer_amp_gradient_accumulation() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = DummyDetectionDataset(num_samples=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dummy_collate)

        model = create_model(backbone="resnet50_fpn", pretrained=False, min_size=256, max_size=256)

        config = TrainerConfig(
            epochs=2,
            batch_size=2,
            image_size=256,
            learning_rate=0.005,
            momentum=0.9,
            weight_decay=0.0005,
            optimizer="SGD",
            device="cuda" if torch.cuda.is_available() else "cpu",
            scheduler_type="step",
            scheduler_step_size=5,
            scheduler_gamma=0.1,
            use_amp=True,
            gradient_accumulation_steps=2,
            checkpoint_dir=tmp_dir,
            log_interval=1,
        )

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=loader,
            val_loader=loader,
        )

        assert trainer.config.scheduler_type == "step"
        assert trainer.config.scheduler_step_size == 5
        assert trainer.config.scheduler_gamma == 0.1
        assert trainer.accumulation_steps == 2

        # Chạy 1 epoch huấn luyện
        metrics = trainer.train_one_epoch(epoch=1)
        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0
        assert not torch.isnan(torch.tensor(metrics["train_loss"]))

        # Kiểm tra evaluate
        val_metrics = trainer.evaluate()
        assert "val_loss" in val_metrics
        assert val_metrics["val_loss"] > 0
