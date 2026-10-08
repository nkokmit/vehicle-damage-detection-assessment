"""Dataset utilities for vehicle damage detection in CarDD COCO format."""

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
    """Custom collate function to handle variable-size targets in a batch.
    
    Returns lists instead of stacked tensors for variable bounding box counts.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


# Alias tương thích ngược
conllate_fn = collate_fn


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # chuyển sang RGB
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
                try: # nhánh chính thực hiện với Albumentations cần truyền 3 tham số thay vì chỉ 1 tham số ảnh
                    transformed = self.transforms(
                        image=image,
                        bboxes=boxes_list,
                        category_ids=labels_list,
                    )
                    image = transformed["image"]
                    boxes_list = [list(b) for b in transformed["bboxes"]]
                    labels_list = [int(cid) for cid in transformed["category_ids"]]
                except TypeError: # nhánh dự phòng nếu transform chỉ nhận 1 tham số (ví dụ: ToTensorV2)
                    image = self.transforms(image)

        # Đảm bảo image là PyTorch Tensor [C, H, W] chuẩn hóa
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()

        # ảnh ban đầu (H, W, C) numpy array hoặc pil object không rõ ràng cần chuyển sang pytorch hoặc đổi trục sang (C,H,W) v
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
