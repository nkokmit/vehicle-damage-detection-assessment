from pathlib import Path
import pytest
import torch
from src.data.loader import build_cardd_dataset, build_dataloader, build_dataset
from src.data.dataset import collate_fn, VehicleDamageDataset

CARDD_TRAIN_JSON = Path("data/CarDD_COCO/annotations/instances_train2017.json")
CARDD_IMG_DIR = Path("data/CarDD_COCO/train2017")


def test_build_dataset_returns_dataset() -> None:
    dataset = build_dataset("train")
    assert dataset.annotation_file.name == "train.json"


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataset_loading() -> None:
    dataset = build_cardd_dataset("train", cardd_dir="data/CarDD_COCO", use_augmentation=False)
    assert len(dataset) == 2816
    assert dataset.num_classes == 7
    assert len(dataset.classes) == 6


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataset_sample_format() -> None:
    dataset = build_cardd_dataset("train", cardd_dir="data/CarDD_COCO", use_augmentation=False)
    img_tensor, target = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dim() == 3
    assert img_tensor.shape[0] == 3

    expected_keys = {"boxes", "labels", "image_id", "area", "iscrowd", "orig_size", "file_name"}
    assert expected_keys.issubset(set(target.keys()))

    boxes = target["boxes"]
    assert isinstance(boxes, torch.Tensor)
    assert boxes.dim() == 2
    assert boxes.shape[1] == 4
    assert boxes.dtype == torch.float32

    if len(boxes) > 0:
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        assert (boxes[:, 3] >= boxes[:, 1]).all()

    labels = target["labels"]
    assert labels.dtype == torch.int64


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataloader_collate() -> None:
    dataset = build_cardd_dataset("train", cardd_dir="data/CarDD_COCO", use_augmentation=False)
    loader = build_dataloader(dataset, batch_size=3, shuffle=False, num_workers=0)

    batch_imgs, batch_targets = next(iter(loader))
    assert len(batch_imgs) == 3
    assert len(batch_targets) == 3
    for img, target in zip(batch_imgs, batch_targets):
        assert isinstance(img, torch.Tensor)
        assert isinstance(target, dict)
        assert "boxes" in target


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataset_summary() -> None:
    dataset = build_cardd_dataset("train", cardd_dir="data/CarDD_COCO", use_augmentation=False)
    summary = dataset.summary()
    assert summary["total_images"] == 2816
    assert summary["total_annotations"] == 6211
    assert "scratch" in summary["class_distribution"]


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataset_advanced_augmentation() -> None:
    dataset = build_cardd_dataset("train", cardd_dir="data/CarDD_COCO", img_size=(800, 800), use_augmentation=True, use_advanced_aug=True)
    img_tensor, target = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 800, 800)
    assert "boxes" in target

