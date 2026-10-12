"""Script huấn luyện YOLOv8/YOLOv11 Instance Segmentation trên CarDD dataset.

Tích hợp tự động MLflow tracking vào SQLite db (sqlite:///mlflow.db)
giúp so sánh trực tiếp với các run Faster R-CNN (ResNet-50 FPN V1 và V2).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# Numpy 2.0+ compatibility patch cho Ultralytics metrics
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO Segmentation trên CarDD Dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="Kiến trúc YOLO ('yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolo11s-seg.pt')",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/yolo_cardd_seg.yaml",
        help="Đường dẫn file data yaml",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Số epoch huấn luyện")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (640 hoặc 800)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 cho 640, 4 cho 800 trên 4GB VRAM)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Learning rate ban đầu")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0' hoặc 'cpu')")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader num workers")
    parser.add_argument("--run-name", type=str, default=None, help="Tên run hiển thị trên MLflow / thư mục")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh (1 epoch)")
    return parser.parse_args()


def setup_mlflow(run_name: str) -> None:
    """Cấu hình biến môi trường cho Ultralytics MLflow callback và tắt wandb."""
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"wandb": False})

    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "vehicle-damage-detection"
    os.environ["MLFLOW_RUN"] = run_name


def main() -> None:
    args = parse_args()

    model_base = Path(args.model).stem
    default_run_name = f"{model_base}_img{args.imgsz}_b{args.batch}_ep{args.epochs}"
    run_name = args.run_name or default_run_name

    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        fallback = PROJECT_ROOT / "data" / "CarDD_YOLO" / "data.yaml"
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Không tìm thấy file cấu hình dữ liệu tại: {data_path}")

    setup_mlflow(run_name)

    logger.info("=" * 65)
    logger.info("       YOLO INSTANCE SEGMENTATION - TRAINING PIPELINE")
    logger.info("=" * 65)
    logger.info("Model Backbone       : %s", args.model)
    logger.info("Dataset Config       : %s", data_path)
    logger.info("Epochs               : %d", 1 if args.dry_run else args.epochs)
    logger.info("Image Size           : %d", args.imgsz)
    logger.info("Batch Size           : %d", args.batch)
    logger.info("Device               : %s", args.device)
    logger.info("MLflow Tracking URI  : sqlite:///mlflow.db")
    logger.info("Run Name             : %s", run_name)
    logger.info("=" * 65)

    from ultralytics import YOLO

    # 1. Khởi tạo mô hình pretrained
    model = YOLO(args.model)

    # 2. Huấn luyện mô hình
    epochs_to_train = 1 if args.dry_run else args.epochs
    train_results = model.train(
        data=str(data_path),
        epochs=epochs_to_train,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        project="artifacts/checkpoints/yolo",
        name=run_name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
    )

    # 3. Đánh giá chi tiết trên tập validation
    logger.info("Hoàn thành huấn luyện. Đang đánh giá chi tiết trên tập Validation...")
    val_metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
    )

    # 4. Lưu báo cáo tổng kết ra thư mục reports
    report_dir = PROJECT_ROOT / "artifacts" / "reports" / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    box_map50 = getattr(val_metrics.box, "map50", 0.0)
    box_map = getattr(val_metrics.box, "map", 0.0)
    seg_map50 = getattr(val_metrics.seg, "map50", 0.0)
    seg_map = getattr(val_metrics.seg, "map", 0.0)

    summary_md = f"""# Báo cáo Đánh giá Mô hình {run_name}

## 1. Các Chỉ số Tổng thể (Instance Segmentation & Detection)

| Metric | Bounding Box (Box) | Phân đoạn (Mask/Seg) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **mAP@0.50** | **{box_map50:.4f}** | **{seg_map50:.4f}** | Chuẩn VOC |
| **mAP@0.5:0.95** | **{box_map:.4f}** | **{seg_map:.4f}** | Chuẩn COCO đa ngưỡng |

## 2. Thông số Huấn luyện
- **Kiến trúc mô hình**: `{args.model}`
- **Kích thước ảnh**: `{args.imgsz}x{args.imgsz}`
- **Số epochs**: `{epochs_to_train}`
- **Batch size**: `{args.batch}`
- **Checkpoints lưu tại**: `artifacts/checkpoints/yolo/{run_name}/weights/best.pt`
"""
    with open(report_dir / "final_eval_report.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    logger.info("=" * 65)
    logger.info("  HUẤN LUYỆN YOLO HOÀN TẤT THÀNH CÔNG!")
    logger.info("  Box mAP@0.5  : %.4f | Box mAP@0.5:0.95 : %.4f", box_map50, box_map)
    logger.info("  Mask mAP@0.5 : %.4f | Mask mAP@0.5:0.95: %.4f", seg_map50, seg_map)
    logger.info("  Báo cáo lưu tại: %s", report_dir / "final_eval_report.md")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

