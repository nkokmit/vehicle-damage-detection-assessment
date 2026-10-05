"""Validation helpers for dataset integrity checks."""

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
