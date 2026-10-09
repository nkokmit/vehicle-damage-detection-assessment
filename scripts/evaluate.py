"""CLI entrypoint for vehicle damage detection model evaluation.

Evaluates key detection metrics:
- IoU (Mean IoU of matched boxes)
- Precision @ 0.50
- Recall @ 0.50
- AP per class (@ 0.50 and @ 0.50:0.95)
- mAP@0.50 (Pascal VOC style)
- mAP@0.50:0.95 (COCO standard)
- Official pycocotools COCOeval
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.data.loader import build_cardd_dataset, build_dataloader
from src.evaluation.evaluator import Evaluator, EvaluatorConfig
from src.models.model_factory import create_model
from src.tracking.mlflow_tracker import MLflowConfig, MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on CarDD dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--backbone", type=str, default="resnet50_fpn", help="Model backbone ('resnet50_fpn' or 'resnet50_fpn_v2')")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on ('val' or 'test')")
    parser.add_argument("--cardd-dir", type=str, default="data/CarDD_COCO", help="Path to CarDD dataset root")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num workers (default 0)")
    parser.add_argument("--image-size", type=int, default=512, help="Image size (min and max)")
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Confidence score threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports", help="Directory to save evaluation reports")
    parser.add_argument("--log-mlflow", action="store_true", help="Log evaluation metrics to MLflow")
    parser.add_argument("--experiment-name", type=str, default="vehicle-damage-detection", help="MLflow experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Quick run with 4 batches for testing")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # UTF-8 stdout trên Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = torch.device(args.device)
    logger.info("=" * 65)
    logger.info("     VEHICLE DAMAGE DETECTION - MODEL EVALUATION")
    logger.info("=" * 65)
    logger.info("Device         : %s", device)
    logger.info("Backbone       : %s", args.backbone)
    logger.info("Checkpoint     : %s", args.checkpoint or "COCO Pretrained (Zero-shot / baseline)")
    logger.info("Split          : %s", args.split)
    logger.info("Score Threshold: %.2f", args.score_thresh)
    logger.info("Image Size     : (%d, %d)", args.image_size, args.image_size)
    logger.info("=" * 65)

    # 1. Nạp Dataset và DataLoader
    logger.info("Loading '%s' dataset from: %s", args.split, args.cardd_dir)
    dataset = build_cardd_dataset(
        split=args.split,
        cardd_dir=args.cardd_dir,
        img_size=(args.image_size, args.image_size),
        use_augmentation=False,
    )
    data_loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 2. Xây dựng Mô hình
    model = create_model(
        model_name="faster_rcnn",
        backbone=args.backbone,
        num_classes=7,
        pretrained=True if args.checkpoint is None else False,
        min_size=args.image_size,
        max_size=args.image_size,
    )

    if args.checkpoint:
        chkpt_path = Path(args.checkpoint)
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {chkpt_path}")
        logger.info("Loading checkpoint weights from: %s", chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model.to(device)

    # Đường dẫn annotation COCO chính thức
    ann_file = Path(args.cardd_dir) / "annotations" / f"instances_{args.split}2017.json"

    # 3. Khởi tạo Evaluator và thực thi
    eval_config = EvaluatorConfig(
        device=str(device),
        score_threshold=args.score_thresh,
        use_amp=True if device.type == "cuda" else False,
        max_batches=4 if args.dry_run else None,
    )
    evaluator = Evaluator(config=eval_config)

    metrics = evaluator.evaluate(
        model=model,
        data_loader=data_loader,
        coco_annotation_file=str(ann_file) if ann_file.exists() else None,
    )

    # 4. Hiển thị kết quả ra màn hình
    print("\n" + "=" * 70)
    print("                KET QUA DANH GIA MO HINH (SUMMARY)")
    print("=" * 70)
    summary_data = [
        {"Chi so": "mAP@0.5:0.95 (COCO)", "Gia tri": f"{metrics['mAP_50_95']:.4f}"},
        {"Chi so": "mAP@0.50 (VOC)", "Gia tri": f"{metrics['mAP_50']:.4f}"},
        {"Chi so": "Precision@0.50", "Gia tri": f"{metrics['precision_50']:.4f}"},
        {"Chi so": "Recall@0.50", "Gia tri": f"{metrics['recall_50']:.4f}"},
        {"Chi so": "Mean IoU (Matched)", "Gia tri": f"{metrics['mean_iou']:.4f}"},
        {"Chi so": "Tong Ground Truths", "Gia tri": str(metrics['total_gt'])},
        {"Chi so": "Tong Du doan", "Gia tri": str(metrics['total_predictions'])},
        {"Chi so": "Do tre (ms/anh)", "Gia tri": f"{metrics['latency_ms']:.2f} ms"},
        {"Chi so": "Toc do (FPS)", "Gia tri": f"{metrics['fps']:.1f}"},
    ]
    print(pd.DataFrame(summary_data).to_string(index=False))

    print("\n" + "=" * 70)
    print("             CHI TIET TUNG LOP TON THUONG (PER-CLASS)")
    print("=" * 70)
    per_class_rows = []
    for cls_name, cls_m in metrics.get("per_class", {}).items():
        per_class_rows.append({
            "Lop": cls_name,
            "GT": cls_m["num_gt"],
            "Pred": cls_m["num_pred"],
            "TP@0.5": cls_m["num_tp"],
            "FP@0.5": cls_m["num_fp"],
            "Precision": f"{cls_m['precision_50']:.4f}",
            "Recall": f"{cls_m['recall_50']:.4f}",
            "AP@0.50": f"{cls_m['ap_50']:.4f}",
            "AP@0.5:0.95": f"{cls_m['ap_50_95']:.4f}",
            "Mean IoU": f"{cls_m['mean_iou']:.4f}",
        })
    print(pd.DataFrame(per_class_rows).to_string(index=False))
    print("=" * 70 + "\n")

    # 5. Lưu báo cáo vào file
    prefix = f"eval_{args.backbone}_{args.split}"
    json_path, md_path = evaluator.save_report(metrics, output_dir=args.output_dir, prefix=prefix)
    logger.info("Reports saved: JSON -> %s | Markdown -> %s", json_path, md_path)

    # 6. Ghi log lên MLflow nếu có cờ --log-mlflow
    if args.log_mlflow:
        mlflow_config = MLflowConfig(tracking_uri="sqlite:///mlflow.db", experiment_name=args.experiment_name)
        tracker = MLflowTracker(config=mlflow_config)
        with tracker.run(run_name=f"eval_{args.backbone}_{args.split}", tags={"task": "evaluation", "split": args.split}):
            flat_metrics = {
                "eval_mAP_50": metrics["mAP_50"],
                "eval_mAP_50_95": metrics["mAP_50_95"],
                "eval_precision_50": metrics["precision_50"],
                "eval_recall_50": metrics["recall_50"],
                "eval_mean_iou": metrics["mean_iou"],
                "eval_latency_ms": metrics["latency_ms"],
                "eval_fps": metrics["fps"],
            }
            for cls_name, cls_m in metrics.get("per_class", {}).items():
                flat_metrics[f"eval_ap_50_{cls_name.replace(' ', '_')}"] = cls_m["ap_50"]
                flat_metrics[f"eval_ap_50_95_{cls_name.replace(' ', '_')}"] = cls_m["ap_50_95"]

            tracker.log_metrics(flat_metrics)
            tracker.log_artifact(str(md_path), artifact_path="reports")
            tracker.log_artifact(str(json_path), artifact_path="reports")
            logger.info("Evaluation metrics and reports logged to MLflow under experiment '%s'", args.experiment_name)


if __name__ == "__main__":
    main()
