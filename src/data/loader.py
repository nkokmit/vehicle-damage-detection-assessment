"""Data loader helper utilities for vehicle damage detection."""

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.data.dataset import VehicleDamageDataset, collate_fn
from src.data.transforms import get_train_transforms, get_val_transforms


def build_dataset(
    split: str,
    data_root: str | Path = "data/processed",
    annotation_file: str | Path | None = None,
    transforms: Callable[..., Any] | None = None,
) -> VehicleDamageDataset:
    root_path = Path(data_root)

    if annotation_file is not None:
        ann_path = Path(annotation_file)
        img_dir = root_path
    elif "CarDD" in str(root_path) or root_path.name == "CarDD_COCO":
        ann_path = root_path / "annotations" / f"instances_{split}2017.json"
        img_dir = root_path / f"{split}2017"
    else:
        img_dir = root_path / split
        ann_path = Path("data/annotations") / f"{split}.json"

    return VehicleDamageDataset(
        root_dir=img_dir,
        annotation_file=ann_path,
        transforms=transforms,
    )


def build_cardd_dataset(
    split: str = "train",
    cardd_dir: str | Path = "data/CarDD_COCO",
    img_size: tuple[int, int] = (800, 800),
    use_augmentation: bool = True,
    use_advanced_aug: bool = True,
) -> VehicleDamageDataset:
    cardd_path = Path(cardd_dir)
    img_dir = cardd_path / f"{split}2017"
    ann_file = cardd_path / "annotations" / f"instances_{split}2017.json"

    if split == "train" and use_augmentation:
        transforms = get_train_transforms(img_size=img_size, use_advanced_aug=use_advanced_aug)
    else:
        transforms = get_val_transforms(img_size=img_size)

    return VehicleDamageDataset(
        root_dir=img_dir,
        annotation_file=ann_file,
        transforms=transforms,
    )


def build_dataloader(
    dataset: VehicleDamageDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate: Callable[..., Any] = collate_fn,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate,
    )
