"""CLI entrypoint for dataset preparation and validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert_coco_to_yolo import convert_split, generate_data_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate CarDD dataset for YOLO & PyTorch models")
    parser.add_argument("--cardd-dir", type=str, default="data/CarDD_COCO", help="Path to raw CarDD COCO dataset")
    parser.add_argument("--yolo-dir", type=str, default="data/CarDD_YOLO", help="Target path for YOLO segmentation dataset")
    parser.add_argument("--link-mode", type=str, default="hardlink", choices=["hardlink", "symlink", "copy"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cardd_path = Path(args.cardd_dir)
    yolo_path = Path(args.yolo_dir)

    if not cardd_path.exists():
        raise FileNotFoundError(f"CarDD dataset directory not found at: {cardd_path}")

    logger.info("=" * 65)
    logger.info("       DATA PREPARATION: CarDD COCO -> YOLO SEGMENTATION")
    logger.info("=" * 65)
    logger.info("CarDD COCO Path : %s", cardd_path)
    logger.info("YOLO Target Path: %s", yolo_path)
    logger.info("Link Mode       : %s", args.link_mode)
    logger.info("=" * 65)

    splits = [
        ("train", "train2017", "instances_train2017.json"),
        ("val", "val2017", "instances_val2017.json"),
        ("test", "test2017", "instances_test2017.json"),
    ]

    for split_name, img_dir_name, json_name in splits:
        json_file = cardd_path / "annotations" / json_name
        img_dir = cardd_path / img_dir_name
        if not json_file.exists() or not img_dir.exists():
            logger.warning("Split %s missing at %s, skipping.", split_name, json_file)
            continue

        out_img = yolo_path / "images" / split_name
        out_lbl = yolo_path / "labels" / split_name
        stats = convert_split(json_file, img_dir, out_img, out_lbl, link_mode=args.link_mode)
        logger.info(
            "Split [%s] converted: %d images, %d annotations, %d valid polygons",
            split_name,
            stats["images"],
            stats["annotations"],
            stats["polygons"],
        )

    yaml_file = generate_data_yaml(yolo_path)
    logger.info("=" * 65)
    logger.info("Dataset preparation complete! YOLO config saved to: %s", yaml_file)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
