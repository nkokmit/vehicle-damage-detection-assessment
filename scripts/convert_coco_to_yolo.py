"""Script chuyển đổi CarDD COCO dataset sang định dạng YOLO Segmentation.

Đọc instances_{split}2017.json, trích xuất tọa độ polygon, chuẩn hóa [0, 1]
và tạo cấu trúc thư mục data/CarDD_YOLO phục vụ huấn luyện YOLOv8-seg / YOLOv11-seg.
Sử dụng hard link (os.link) để không tốn thêm dung lượng ổ đĩa.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "dent",           # 0
    "scratch",        # 1
    "crack",          # 2
    "glass shatter",  # 3
    "lamp broken",    # 4
    "tire flat",      # 5
]


def convert_split(
    coco_json_path: Path,
    image_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    link_mode: str = "hardlink",
) -> dict[str, int]:
    """Chuyển đổi một split (train/val/test) sang định dạng YOLO Segmentation."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, encoding="utf-8") as f:
        coco_data = json.load(f)

    # Lập bản đồ categories: category_id -> class_index (0..5)
    cat_map = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    # Bản đồ image_id -> image metadata
    images_map = {img["id"]: img for img in coco_data["images"]}

    # Gom nhóm annotations theo image_id
    img_to_anns: dict[int, list[dict]] = {img_id: [] for img_id in images_map}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id in img_to_anns:
            img_to_anns[img_id].append(ann)

    stats: dict[str, int] = {
        "images": len(images_map),
        "annotations": len(coco_data["annotations"]),
        "polygons": 0,
        "empty_images": 0,
    }

    for img_id, img_info in images_map.items():
        file_name = img_info["file_name"]
        w = float(img_info["width"])
        h = float(img_info["height"])

        # 1. Liên kết / sao chép file ảnh
        src_img = image_dir / file_name
        dst_img = out_img_dir / file_name

        if not dst_img.exists() and src_img.exists():
            if link_mode == "hardlink":
                try:
                    os.link(src_img, dst_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)
            elif link_mode == "symlink":
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        # 2. Tạo nhãn YOLO segmentation .txt
        stem = Path(file_name).stem
        dst_lbl = out_lbl_dir / f"{stem}.txt"
        anns = img_to_anns.get(img_id, [])

        lines: list[str] = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_map:
                continue
            class_idx = cat_map[cat_id]
            segmentation = ann.get("segmentation", [])

            if isinstance(segmentation, list):
                for poly in segmentation:
                    if len(poly) < 6:  # Cần tối thiểu 3 điểm (6 tọa độ x, y)
                        continue
                    # Chuẩn hóa tọa độ về dải [0, 1]
                    norm_points: list[str] = []
                    for i in range(0, len(poly), 2):
                        px = max(0.0, min(1.0, poly[i] / w))
                        py = max(0.0, min(1.0, poly[i + 1] / h))
                        norm_points.append(f"{px:.6f}")
                        norm_points.append(f"{py:.6f}")

                    line = f"{class_idx} " + " ".join(norm_points)
                    lines.append(line)
                    stats["polygons"] += 1

        if not lines:
            stats["empty_images"] += 1

        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))

    return stats


def generate_data_yaml(output_dir: Path) -> Path:
    """Tạo file data.yaml cho Ultralytics YOLO."""
    yaml_path = output_dir / "data.yaml"
    abs_out = output_dir.resolve().as_posix()
    content = f"""# CarDD Vehicle Damage Segmentation Dataset for Ultralytics YOLO
path: {abs_out}
train: images/train
val: images/val
test: images/test

# Số lớp và tên lớp
names:
  0: dent
  1: scratch
  2: crack
  3: glass shatter
  4: lamp broken
  5: tire flat
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Generated YOLO data config: %s", yaml_path)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuyển đổi CarDD COCO sang YOLO Segmentation format")
    parser.add_argument("--cardd-dir", type=str, default="data/CarDD_COCO", help="Đường dẫn CarDD COCO dataset")
    parser.add_argument("--output-dir", type=str, default="data/CarDD_YOLO", help="Thư mục đầu ra cho YOLO dataset")
    parser.add_argument("--link-mode", type=str, default="hardlink", choices=["hardlink", "symlink", "copy"], help="Chế độ sao chép ảnh")
    args = parser.parse_args()

    base_in = Path(args.cardd-dir if hasattr(args, "cardd-dir") else args.cardd_dir)
    base_out = Path(args.output_dir)

    splits = [
        ("train", "train2017", "instances_train2017.json"),
        ("val", "val2017", "instances_val2017.json"),
        ("test", "test2017", "instances_test2017.json"),
    ]

    logger.info("=" * 65)
    logger.info("  BẮT ĐẦU CHUYỂN ĐỔI DATASET: CarDD COCO -> YOLO SEGMENTATION")
    logger.info("=" * 65)
    logger.info("Source directory : %s", base_in)
    logger.info("Target directory : %s", base_out)
    logger.info("Link mode        : %s (0 MB dung lượng thừa nếu là hardlink)", args.link_mode)

    total_images = 0
    total_polys = 0

    for split_name, img_folder, json_name in splits:
        json_path = base_in / "annotations" / json_name
        img_dir = base_in / img_folder
        out_img = base_out / "images" / split_name
        out_lbl = base_out / "labels" / split_name

        if not json_path.exists():
            logger.warning("Không tìm thấy %s, bỏ qua split %s", json_path, split_name)
            continue

        logger.info("Đang xử lý split: %s ...", split_name)
        stats = convert_split(json_path, img_dir, out_img, out_lbl, link_mode=args.link_mode)
        logger.info(
            "Split [%s] hoàn tất: %d ảnh, %d annotations, %d polygons (%d ảnh background)",
            split_name,
            stats["images"],
            stats["annotations"],
            stats["polygons"],
            stats["empty_images"],
        )
        total_images += stats["images"]
        total_polys += stats["polygons"]

    generate_data_yaml(base_out)

    logger.info("=" * 65)
    logger.info("  CHUYỂN ĐỔI THÀNH CÔNG!")
    logger.info("  Tổng số ảnh: %d | Tổng số polygons tổn thương: %d", total_images, total_polys)
    logger.info("  Config sẵn sàng tại: %s/data.yaml", base_out.as_posix())
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

