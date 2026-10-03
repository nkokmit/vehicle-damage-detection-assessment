from src.data.loader import build_dataset


def test_build_dataset_returns_dataset() -> None:
    dataset = build_dataset("train")
    assert dataset.annotation_file.name == "train.json"
