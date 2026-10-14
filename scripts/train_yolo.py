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
    parser = argparse.ArgumentParser(description="Train YOLO Segmentation on CarDD Dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="YOLO architecture or checkpoint path ('yolov8s-seg.pt', 'yolov8m-seg.pt', or path to best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/yolo_cardd_seg.yaml",
        help="Path to data config YAML file",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (640 or 800)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (e.g. 8 for 640, 4 for 800 on 4GB VRAM)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader num workers")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow / output directory run name")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Path to last.pt checkpoint to resume training (resume=True). If passed without path, finds latest last.pt.",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only run evaluation (Validation) on the loaded model/checkpoint without training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick test run (1 epoch)")
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

    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        fallback = PROJECT_ROOT / "data" / "CarDD_YOLO" / "data.yaml"
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Không tìm thấy file cấu hình dữ liệu tại: {data_path}")

    from ultralytics import YOLO

    epochs_to_train = 1 if args.dry_run else args.epochs

    # 1. Kịch bản Resume huấn luyện từ checkpoint
    if args.resume:
        if args.resume == "auto":
            candidate_lasts = list((PROJECT_ROOT / "artifacts" / "checkpoints" / "yolo").glob("*/weights/last.pt"))
            if not candidate_lasts:
                raise FileNotFoundError("Không tìm thấy checkpoint last.pt nào trong artifacts/checkpoints/yolo")
            candidate_lasts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            resume_path = candidate_lasts[0]
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists() and (PROJECT_ROOT / args.resume).exists():
                resume_path = PROJECT_ROOT / args.resume
            if not resume_path.exists():
                raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {args.resume}")

        run_name = args.run_name or resume_path.parent.parent.name
        setup_mlflow(run_name)

        logger.info("=" * 65)
        logger.info("       YOLO INSTANCE SEGMENTATION - RESUMING TRAINING")
        logger.info("=" * 65)
        logger.info("Checkpoint Path      : %s", resume_path)
        logger.info("Run Name             : %s", run_name)
        logger.info("=" * 65)

        model = YOLO(str(resume_path))

        if not args.val_only:
            logger.info("Tiếp tục huấn luyện từ checkpoint...")
            train_results = model.train(resume=True)

    else:
        # Kịch bản huấn luyện mới hoặc nạp checkpoint fine-tuning qua --model
        model_path = Path(args.model)
        if not model_path.exists() and (PROJECT_ROOT / args.model).exists():
            model_path = PROJECT_ROOT / args.model

        model_base = model_path.stem
        if model_base in ["best", "last"] and model_path.parent.name == "weights":
            parent_run = model_path.parent.parent.name
            default_run_name = f"{parent_run}_eval" if args.val_only else f"{parent_run}_finetune_ep{args.epochs}"
        else:
            default_run_name = f"{model_base}_img{args.imgsz}_b{args.batch}_ep{args.epochs}"

        run_name = args.run_name or default_run_name
        setup_mlflow(run_name)

        logger.info("=" * 65)
        logger.info("       YOLO INSTANCE SEGMENTATION - TRAINING PIPELINE")
        logger.info("=" * 65)
        logger.info("Model Backbone       : %s", model_path)
        logger.info("Dataset Config       : %s", data_path)
        logger.info("Mode                 : %s", "Evaluation Only" if args.val_only else "Training")
        if not args.val_only:
            logger.info("Epochs               : %d", epochs_to_train)
            logger.info("Batch Size           : %d", args.batch)
        logger.info("Image Size           : %d", args.imgsz)
        logger.info("Device               : %s", args.device)
        logger.info("MLflow Tracking URI  : sqlite:///mlflow.db")
        logger.info("Run Name             : %s", run_name)
        logger.info("=" * 65)

        model = YOLO(str(model_path))

        if not args.val_only:
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
    logger.info("Hoàn thành quá trình. Đang đánh giá chi tiết trên tập Validation...")
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

    mode_str = "Đánh giá (Validation only)" if args.val_only else ("Tiếp tục huấn luyện (Resume)" if args.resume else "Huấn luyện tiêu chuẩn")
    checkpoint_source = str(resume_path) if args.resume else str(args.model)

    summary_md = f"""# Báo cáo Đánh giá Mô hình {run_name}

## 1. Các Chỉ số Tổng thể (Instance Segmentation & Detection)

| Metric | Bounding Box (Box) | Phân đoạn (Mask/Seg) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **mAP@0.50** | **{box_map50:.4f}** | **{seg_map50:.4f}** | Chuẩn VOC |
| **mAP@0.5:0.95** | **{box_map:.4f}** | **{seg_map:.4f}** | Chuẩn COCO đa ngưỡng |

## 2. Thông số
- **Chế độ**: `{mode_str}`
- **Nguồn checkpoint / model**: `{checkpoint_source}`
- **Kích thước ảnh**: `{args.imgsz}x{args.imgsz}`
- **Số epochs**: `{epochs_to_train}`
- **Batch size**: `{args.batch}`
- **Checkpoints lưu tại**: `artifacts/checkpoints/yolo/{run_name}/weights/`
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

