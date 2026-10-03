"""Dataset utilities for vehicle damage detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetSample:
    """A lightweight sample representation."""

    image_path: Path
    target: dict[str, Any]


class VehicleDamageDataset:
    """Minimal dataset container that can be extended later."""

    def __init__(self, root_dir: str | Path, annotation_file: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.annotation_file = Path(annotation_file)
        self.samples: list[DatasetSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]
