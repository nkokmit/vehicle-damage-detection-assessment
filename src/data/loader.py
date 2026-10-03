"""Data loader helper utilities."""

from pathlib import Path

from src.data.dataset import VehicleDamageDataset


def build_dataset(split: str, data_root: str | Path = "data/processed") -> VehicleDamageDataset:
    """Build a dataset object for a split."""

    split_dir = Path(data_root) / split
    annotation = Path("data/annotations") / f"{split}.json"
    return VehicleDamageDataset(root_dir=split_dir, annotation_file=annotation)
