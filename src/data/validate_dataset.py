"""Validation helpers for dataset integrity checks."""

from pathlib import Path


def validate_paths(data_dir: str | Path, annotation_file: str | Path) -> bool:
    """Check whether input dataset paths exist."""

    return Path(data_dir).exists() and Path(annotation_file).exists()
