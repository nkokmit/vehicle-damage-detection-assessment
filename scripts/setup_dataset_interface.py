import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PY = '''"""Dataset utilities for vehicle damage detection in CarDD COCO format."""

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Danh mục 6 lớp tổn thương xe chuẩn của CarDD
DEFAULT_CLASSES: tuple[str, ...] = (
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
)


@dataclass(slots=True)
class DatasetSample:
    """A lightweight sample representation for compatibility and inspectability."""

    image_path: Path
    target: dict[str, Any]


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Custom collate function for object detection DataLoader.

    Since each image may contain a variable number of bounding boxes,
    standard stacking fails. This collates images and targets as parallel tuples/lists.
    """
    return tuple(zip(*batch))


class VehicleDamageDataset(Dataset):
    """PyTorch Dataset for CarDD Vehicle Damage Detection (COCO Annotation Format).

    Supports:
        - Dynamic category loading from COCO JSON or fallback to CarDD defaults.
        - Albumentations transforms (with bounding boxes) and PyTorch transforms.
        - Robust bounding box clipping and validation (converting COCO [x,y,w,h] to [x1,y1,x2,y2]).
        - Safe handling of images with zero or multiple damage annotations.
        - Fast inspection and visualization helpers.
    """

    def __init__(
        self,
        root_dir: str | Path,
        annotation_file: str | Path,
        transforms: Callable[..., Any] | None = None,
        return_sample_obj: bool = False,
        filter_empty: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Directory containing the image files.
            annotation_file: Path to COCO instances JSON file.
            transforms: Albumentations or Callable transform pipeline.
            return_sample_obj: If True, __getitem__ returns a DatasetSample dataclass
                               instead of (image, target) tuple.
            filter_empty: If True, exclude images with 0 annotations.
        """
        self.root_dir = Path(root_dir)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms
        self.return_sample_obj = return_sample_obj
        self.filter_empty = filter_empty

        self.samples: list[DatasetSample] = []
        self.images_info: list[dict[str, Any]] = []
        self.annotations_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.categories: dict[int, str] = {}
        self.class_to_id: dict[str, int] = {}

        if self.annotation_file.exists():
            self._load_coco_annotations()
        else:
            logger.warning(
                "Annotation file not found: %s. Initialized empty dataset.",
                self.annotation_file,
            )

    def _load_coco_annotations(self) -> None:
        """Parse COCO json annotations and construct index."""
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # 1. Trích xuất danh mục nhãn (Categories)
        if "categories" in coco_data and coco_data["categories"]:
            for cat in coco_data["categories"]:
                cid = cat["id"]
                cname = cat["name"]
                self.categories[cid] = cname
                self.class_to_id[cname] = cid
        else:
            # Fallback dùng 6 class mặc định của CarDD
            self.categories = {i + 1: name for i, name in enumerate(DEFAULT_CLASSES)}
            self.class_to_id = {name: i + 1 for i, name in enumerate(DEFAULT_CLASSES)}

        # 2. Gom nhóm annotations theo image_id
        for ann in coco_data.get("annotations", []):
            self.annotations_map[ann["image_id"]].append(ann)

        # 3. Duyệt danh sách ảnh và lập chỉ mục
        for img in coco_data.get("images", []):
            img_id = img["id"]
            anns = self.annotations_map.get(img_id, [])

            if self.filter_empty and len(anns) == 0:
                continue

            self.images_info.append(img)

            # Khởi tạo danh sách DatasetSample để tương thích ngược
            file_name = img["file_name"]
            img_path = self.root_dir / file_name
            self.samples.append(
                DatasetSample(
                    image_path=img_path,
                    target={
                        "image_id": img_id,
                        "file_name": file_name,
                        "width": img.get("width"),
                        "height": img.get("height"),
                        "num_boxes": len(anns),
                    },
                )
            )

    @property
    def classes(self) -> list[str]:
        """List of category names in the dataset."""
        return [self.categories[cid] for cid in sorted(self.categories.keys())]

    @property
    def num_classes(self) -> int:
        """Total number of classes including background (index 0)."""
        return len(self.categories) + 1

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images_info)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]] | DatasetSample:
        """Fetch image and target annotations at given index.

        Args:
            index: Image index.

        Returns:
            Tuple of (image_tensor, target_dict) where target contains:
                - boxes: FloatTensor[N, 4] in [x1, y1, x2, y2] format
                - labels: Int64Tensor[N] (1-indexed class IDs)
                - image_id: Int64Tensor[1]
                - area: FloatTensor[N]
                - iscrowd: Int64Tensor[N]
                - orig_size: Int64Tensor[2] (height, width)
                - file_name: str
        """
        img_info = self.images_info[index]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = self.root_dir / file_name

        # Đọc ảnh RGB
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        raw_anns = self.annotations_map.get(img_id, [])

        boxes_list: list[list[float]] = []
        labels_list: list[int] = []
        areas_list: list[float] = []
        iscrowd_list: list[int] = []

        for ann in raw_anns:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Chuyển đổi sang chuẩn [x1, y1, x2, y2] (PASCAL VOC / Faster R-CNN)
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(w_orig), float(x + w))
            y2 = min(float(h_orig), float(y + h))

            # Bỏ qua các box không hợp lệ (diện tích = 0 hoặc tọa độ lỗi)
            if x2 <= x1 or y2 <= y1:
                continue

            category_id = int(ann["category_id"])
            area = float(ann.get("area", (x2 - x1) * (y2 - y1)))
            iscrowd = int(ann.get("iscrowd", 0))

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(category_id)
            areas_list.append(area)
            iscrowd_list.append(iscrowd)

        # Áp dụng Data Transforms / Augmentations
        if self.transforms is not None:
            if hasattr(self.transforms, "processors") or hasattr(self.transforms, "__call__"):
                try:
                    transformed = self.transforms(
                        image=image,
                        bboxes=boxes_list,
                        category_ids=labels_list,
                    )
                    image = transformed["image"]
                    boxes_list = [list(b) for b in transformed["bboxes"]]
                    labels_list = [int(cid) for cid in transformed["category_ids"]]
                except TypeError:
                    image = self.transforms(image)

        # Đảm bảo image là PyTorch Tensor [C, H, W] chuẩn hóa
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()

        # Tạo Tensor cho Bounding Boxes và Labels
        if len(boxes_list) > 0:
            boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32).reshape(-1, 4)
            labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas_list, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd_list, dtype=torch.int64)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
            areas_tensor = torch.empty((0,), dtype=torch.float32)
            iscrowd_tensor = torch.empty((0,), dtype=torch.int64)

        target: dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
            "orig_size": torch.tensor([h_orig, w_orig], dtype=torch.int64),
            "file_name": file_name,
        }

        if self.return_sample_obj:
            return DatasetSample(image_path=img_path, target=target)

        return image_tensor, target

    def get_image_metadata(self, index: int) -> dict[str, Any]:
        """Get image metadata without loading the image tensor."""
        img_info = self.images_info[index]
        img_id = img_info["id"]
        raw_anns = self.annotations_map.get(img_id, [])

        return {
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "width": img_info.get("width"),
            "height": img_info.get("height"),
            "num_objects": len(raw_anns),
            "classes": [self.categories[a["category_id"]] for a in raw_anns],
            "bboxes": [a["bbox"] for a in raw_anns],
        }

    def summary(self) -> dict[str, Any]:
        """Compute and return summary statistics of the dataset."""
        total_images = len(self.images_info)
        total_annotations = sum(len(anns) for anns in self.annotations_map.values())
        class_counts: dict[str, int] = defaultdict(int)

        for anns in self.annotations_map.values():
            for ann in anns:
                cname = self.categories.get(ann["category_id"], "unknown")
                class_counts[cname] += 1

        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "avg_boxes_per_image": round(total_annotations / max(1, total_images), 2),
            "class_distribution": dict(class_counts),
        }

    def visualize_sample(
        self,
        index: int,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Visualize ground truth bounding boxes on the image at `index`."""
        metadata = self.get_image_metadata(index)
        img_path = self.root_dir / metadata["file_name"]

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")

        colors = [
            (255, 140, 0),
            (0, 140, 255),
            (0, 215, 255),
            (211, 0, 148),
            (34, 34, 255),
            (50, 205, 50),
        ]

        for bbox, cname in zip(metadata["bboxes"], metadata["classes"]):
            x, y, w, h = [int(round(v)) for v in bbox]
            cid = self.class_to_id.get(cname, 1)
            color = colors[(cid - 1) % len(colors)]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img,
                cname,
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        if save_path:
            cv2.imwrite(str(save_path), img)

        return img
'''

TRANSFORMS_PY = '''"""Transform helpers for vehicle damage preprocessing and augmentation."""

from typing import Any, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    img_size: tuple[int, int] = (800, 800),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.4,
            ),
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.0,
        ),
    )


def get_val_transforms(
    img_size: tuple[int, int] = (800, 800),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.0,
        ),
    )


def get_default_transform() -> Callable[[Any], Any]:
    return lambda item: item
'''

LOADER_PY = '''"""Data loader helper utilities for vehicle damage detection."""

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
) -> VehicleDamageDataset:
    cardd_path = Path(cardd_dir)
    img_dir = cardd_path / f"{split}2017"
    ann_file = cardd_path / "annotations" / f"instances_{split}2017.json"

    if split == "train" and use_augmentation:
        transforms = get_train_transforms(img_size=img_size)
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
'''

VALIDATE_PY = '''"""Validation helpers for dataset integrity checks."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_paths(data_dir: str | Path, annotation_file: str | Path) -> bool:
    return Path(data_dir).exists() and Path(annotation_file).exists()


def validate_coco_dataset(
    img_dir: str | Path,
    annotation_file: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    img_path = Path(img_dir)
    ann_path = Path(annotation_file)

    report: dict[str, Any] = {
        "valid": True,
        "img_dir_exists": img_path.exists(),
        "ann_file_exists": ann_path.exists(),
        "total_images_in_json": 0,
        "existing_images": 0,
        "missing_images": [],
        "total_annotations": 0,
        "invalid_bboxes": [],
        "categories_found": {},
    }

    if not report["img_dir_exists"] or not report["ann_file_exists"]:
        report["valid"] = False
        return report

    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except Exception as e:
        report["valid"] = False
        report["error"] = f"Failed to parse JSON: {e}"
        return report

    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    report["categories_found"] = categories

    images = coco_data.get("images", [])
    report["total_images_in_json"] = len(images)

    for img in images:
        fname = img["file_name"]
        if (img_path / fname).exists():
            report["existing_images"] += 1
        else:
            report["missing_images"].append(fname)

    annotations = coco_data.get("annotations", [])
    report["total_annotations"] = len(annotations)

    for ann in annotations:
        ann_id = ann.get("id")
        bbox = ann.get("bbox", [])
        cid = ann.get("category_id")

        if len(bbox) != 4:
            report["invalid_bboxes"].append({"ann_id": ann_id, "reason": "Length != 4", "bbox": bbox})
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            report["invalid_bboxes"].append({"ann_id": ann_id, "reason": "Non-positive dimensions", "bbox": bbox})

        if cid not in categories:
            report["invalid_bboxes"].append({"ann_id": ann_id, "reason": "Unknown category_id", "category_id": cid})

    if len(report["missing_images"]) > 0 or len(report["invalid_bboxes"]) > 0:
        report["valid"] = False

    if verbose:
        logger.info("Dataset Validation for %s:", ann_path.name)
        logger.info("  Images: %d/%d exist", report["existing_images"], report["total_images_in_json"])
        logger.info("  Annotations: %d total, %d invalid bboxes", report["total_annotations"], len(report["invalid_bboxes"]))
        logger.info("  Categories: %s", categories)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cardd_ann = "data/CarDD_COCO/annotations/instances_train2017.json"
    cardd_img = "data/CarDD_COCO/train2017"

    result = validate_coco_dataset(cardd_img, cardd_ann)
    print("Validation passed:", result["valid"])
'''

INIT_PY = '''"""Data utilities package for vehicle damage detection."""

from src.data.dataset import (
    DEFAULT_CLASSES,
    DatasetSample,
    VehicleDamageDataset,
    collate_fn,
)
from src.data.loader import (
    build_cardd_dataset,
    build_dataloader,
    build_dataset,
)
from src.data.transforms import (
    get_default_transform,
    get_train_transforms,
    get_val_transforms,
)
from src.data.validate_dataset import (
    validate_coco_dataset,
    validate_paths,
)

__all__ = [
    "DEFAULT_CLASSES",
    "DatasetSample",
    "VehicleDamageDataset",
    "collate_fn",
    "build_dataset",
    "build_cardd_dataset",
    "build_dataloader",
    "get_default_transform",
    "get_train_transforms",
    "get_val_transforms",
    "validate_paths",
    "validate_coco_dataset",
]
'''

TEST_DATASET_PY = '''from pathlib import Path
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
'''

def main():
    files = {
        BASE_DIR / "src" / "data" / "dataset.py": DATASET_PY,
        BASE_DIR / "src" / "data" / "transforms.py": TRANSFORMS_PY,
        BASE_DIR / "src" / "data" / "loader.py": LOADER_PY,
        BASE_DIR / "src" / "data" / "validate_dataset.py": VALIDATE_PY,
        BASE_DIR / "src" / "data" / "__init__.py": INIT_PY,
        BASE_DIR / "tests" / "test_dataset.py": TEST_DATASET_PY,
    }

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Written: {path} ({len(content)} bytes)")

if __name__ == "__main__":
    main()
