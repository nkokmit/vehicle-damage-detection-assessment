"""CLI entrypoint for vehicle damage detection and segmentation inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.0+ compatibility patch
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle damage detection & segmentation inference")
    parser.add_argument(
        "--source",
        type=str,
        default="data/CarDD_COCO/val2017/000001.jpg",
        help="Path to input image or directory of images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.onnx",
        help="Path to model weights (.onnx or .pt)",
    )
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence score threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--save", action="store_true", default=True, help="Save visualized predictions")
    parser.add_argument("--output-dir", type=str, default="artifacts/predictions", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists() and (PROJECT_ROOT / args.model).exists():
        model_path = PROJECT_ROOT / args.model

    if not model_path.exists():
        fallback_pt = (
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "yolo"
            / "yolov8m_seg_res800_ep40"
            / "weights"
            / "best.pt"
        )
        if fallback_pt.exists():
            model_path = fallback_pt
        else:
            raise FileNotFoundError(f"Model file not found: {args.model}")

    source_path = Path(args.source)
    if not source_path.exists() and (PROJECT_ROOT / args.source).exists():
        source_path = PROJECT_ROOT / args.source

    if not source_path.exists():
        raise FileNotFoundError(f"Input source not found: {args.source}")

    logger.info("=" * 65)
    logger.info("       VEHICLE DAMAGE DETECTION - INFERENCE PIPELINE")
    logger.info("=" * 65)
    logger.info("Model       : %s", model_path)
    logger.info("Source      : %s", source_path)
    logger.info("Image Size  : %dx%d", args.imgsz, args.imgsz)
    logger.info("Confidence  : %.2f", args.conf)
    logger.info("Output dir  : %s", args.output_dir)
    logger.info("=" * 65)

    from ultralytics import YOLO

    task = "segment" if "seg" in str(model_path).lower() or model_path.suffix == ".onnx" else None
    model = YOLO(str(model_path), task=task)
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        project=args.output_dir,
        name="run",
        exist_ok=True,
    )

    total_detections = 0
    for r in results:
        img_name = Path(r.path).name
        num_items = len(r.boxes)
        total_detections += num_items
        logger.info("Image: %s -> Found %d damage instances", img_name, num_items)
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf_val = float(box.conf.item())
            bbox_coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]
            mask_info = ""
            if r.masks is not None:
                mask_pixels = int(r.masks.data[i].sum().item())
                mask_info = f" | Mask Area: {mask_pixels} px"
            logger.info("  [%d] %s (Conf: %.2f) | BBox: %s%s", i + 1, cls_name, conf_val, bbox_coords, mask_info)

    logger.info("=" * 65)
    logger.info("Inference completed! Total detections: %d", total_detections)
    if args.save:
        logger.info("Visualized predictions saved to: %s/run", args.output_dir)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
