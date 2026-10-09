"""Script to train, evaluate, and compare 2 Faster R-CNN ResNet-50 models in MLflow.

Experiment 1: Faster R-CNN ResNet-50 FPN V1 (Classic Baseline)
Experiment 2: Faster R-CNN ResNet-50 FPN V2 (Modernized Architecture with SyncBN & Improved FPN)
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Đảm bảo import được module 'src' khi chạy script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.data.loader import build_cardd_dataset, build_dataloader
from src.models.model_factory import create_model
from src.tracking.mlflow_tracker import MLflowConfig, MLflowTracker
from src.training.trainer import Trainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_experiments")


def measure_inference_latency(model: torch.nn.Module, device: torch.device, num_warmup: int = 5, num_runs: int = 20) -> float:
    """Measure average inference latency per image in milliseconds."""
    model.eval()
    dummy_input = [torch.rand(3, 800, 800).to(device)]

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start

    latency_ms = (total_time / num_runs) * 1000.0
    return round(latency_ms, 2)


def run_experiment(
    exp_name: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_loader: Any,
    val_loader: Any,
    device: str,
    tracker: MLflowTracker,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Execute training and tracking for a single Faster R-CNN model variant."""
    logger.info("=" * 65)
    logger.info("STARTING EXPERIMENT: %s (Backbone: %s)", exp_name, backbone)
    logger.info("=" * 65)

    # 1. Khởi tạo mô hình
    model = create_model(
        model_name="faster_rcnn",
        backbone=backbone,
        num_classes=7,
        pretrained=True,
        trainable_backbone_layers=3,
    )
    param_counts = model.count_parameters()
    logger.info(
        "Model %s created: Total params: %s | Trainable: %s",
        backbone,
        f"{param_counts['total_parameters']:,}",
        f"{param_counts['trainable_parameters']:,}",
    )

    # Đo độ trễ suy luận
    device_obj = torch.device(device)
    model.to(device_obj)
    latency_ms = measure_inference_latency(model, device_obj)
    logger.info("Inference latency on %s: %.2f ms / image", device, latency_ms)

    # 2. Khởi tạo Trainer
    chkpt_dir = Path("artifacts/checkpoints") / exp_name
    trainer_config = TrainerConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=str(chkpt_dir),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tracker=tracker,
    )

    # 3. Ghi log tham số và chạy huấn luyện với MLflow
    tags = {
        "model.family": "Faster R-CNN",
        "model.backbone": backbone,
        "task": "vehicle-damage-detection",
        "classes": "7",
    }

    with tracker.run(run_name=exp_name, tags=tags, description=f"Faster R-CNN with {backbone} backbone for vehicle damage detection"):
        # Log Hyperparameters
        tracker.log_params({
            "model_architecture": "faster_rcnn",
            "backbone": backbone,
            "num_classes": 7,
            "pretrained": True,
            "trainable_backbone_layers": 3,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "device": device,
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
            "inference_latency_ms": latency_ms,
        })

        tracker.log_metric("inference_latency_ms", latency_ms)

        # Chạy Training Loop
        history_summary = trainer.train()

        # Log Final Summary Metrics
        tracker.log_metrics({
            "final_train_loss": history_summary["final_train_loss"],
            "best_val_loss": history_summary["best_val_loss"],
        })

    logger.info("FINISHED EXPERIMENT: %s | Best Val Loss: %.4f", exp_name, history_summary["best_val_loss"])

    return {
        "run_name": exp_name,
        "backbone": backbone,
        "total_parameters": param_counts["total_parameters"],
        "trainable_parameters": param_counts["trainable_parameters"],
        "latency_ms": latency_ms,
        "final_train_loss": history_summary["final_train_loss"],
        "best_val_loss": history_summary["best_val_loss"],
    }


def generate_comparison_report(results: list[dict[str, Any]], tracker: MLflowTracker, output_dir: Path) -> Path:
    """Generate a markdown report comparing the two experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "mlflow_model_comparison.md"

    # Tạo bảng so sánh
    rows = []
    for r in results:
        rows.append({
            "Mô hình": r["run_name"],
            "Backbone": r["backbone"],
            "Tổng Params": f"{r['total_parameters']:,}",
            "Trainable Params": f"{r['trainable_parameters']:,}",
            "Độ trễ (ms/ảnh)": f"{r['latency_ms']:.2f} ms",
            "Train Loss": f"{r['final_train_loss']:.4f}",
            "Best Val Loss": f"{r['best_val_loss']:.4f}",
        })

    df = pd.DataFrame(rows)
    markdown_table = df.to_markdown(index=False)

    report_content = f"""# Báo cáo So sánh 2 Mô hình Faster R-CNN trong MLflow

Báo cáo so sánh hai kiến trúc Faster R-CNN ResNet-50 trên tập dữ liệu Vehicle Damage Detection (CarDD COCO format, 7 classes).

## 1. Bảng So sánh Tổng hợp (Summary Table)

{markdown_table}

## 2. Đặc điểm Kiến trúc 2 Mô hình

* **Model 1: `faster_rcnn_resnet50_v1` (ResNet-50-FPN V1)**:
  * Kiến trúc ResNet-50-FPN cổ điển (Lin et al. 2017).
  * Trọng số nền: `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (COCO mAP: 37.0).
  * Đầu ROI box predictor: 2 lớp Linear thông thường.
  * Ưu điểm: Ổn định, tiêu tốn ít bộ nhớ GPU, thời gian huấn luyện nhanh.

* **Model 2: `faster_rcnn_resnet50_v2` (ResNet-50-FPN V2)**:
  * Kiến trúc ResNet-50-FPN cải tiến (Torchvision V2 modernization).
  * Trọng số nền: `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` (COCO mAP: 46.7, vượt trội +9.7 mAP).
  * Cải tiến: FPN thế hệ mới, tích hợp Synchronized BatchNorm trong RoI heads, anchor generator tối ưu.
  * Ưu điểm: Độ chính xác phát hiện và hội tụ tốt hơn trên các vật thể nhỏ và đa kích thước.

## 3. Xem chi tiết trên Giao diện MLflow UI

Để xem biểu đồ loss curves, siêu tham số và artifacts trực tiếp trên trình duyệt:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Truy cập: [http://localhost:5000](http://localhost:5000) -> Experiment: `vehicle-damage-detection`.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Saved comparison report to: %s", report_file)
    return report_file


def main():
    parser = argparse.ArgumentParser(description="Compare Faster R-CNN ResNet-50 v1 vs v2 in MLflow")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train each model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max train batches per epoch (useful for dry-run)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Max val batches per epoch")
    parser.add_argument("--cardd-dir", type=str, default="data/CarDD_COCO", help="Path to CarDD COCO dataset")
    parser.add_argument("--dry-run", action="store_true", help="Quick run with 5 batches to verify pipeline end-to-end")

    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        args.max_train_batches = 4
        args.max_val_batches = 2
        logger.info("Running in DRY-RUN mode (1 epoch, max 4 train batches, 2 val batches)")

    # 1. Nạp Dataset và DataLoader
    logger.info("Loading CarDD dataset from: %s", args.cardd_dir)
    train_dataset = build_cardd_dataset("train", cardd_dir=args.cardd_dir, use_augmentation=True)
    val_dataset = build_cardd_dataset("val", cardd_dir=args.cardd_dir, use_augmentation=False)

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Khởi tạo MLflow Tracker
    mlflow_config = MLflowConfig(tracking_uri="sqlite:///mlflow.db", experiment_name="vehicle-damage-detection")
    tracker = MLflowTracker(config=mlflow_config)

    # 3. Chạy Experiment 1: Faster R-CNN ResNet-50 FPN V1
    res_v1 = run_experiment(
        exp_name="faster_rcnn_resnet50_v1",
        backbone="resnet50_fpn",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        tracker=tracker,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    # 4. Chạy Experiment 2: Faster R-CNN ResNet-50 FPN V2
    res_v2 = run_experiment(
        exp_name="faster_rcnn_resnet50_v2",
        backbone="resnet50_fpn_v2",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        tracker=tracker,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    # 5. Xuất báo cáo so sánh
    results = [res_v1, res_v2]
    report_file = generate_comparison_report(results, tracker, PROJECT_ROOT / "artifacts" / "reports")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("           KET QUA SO SANH 2 MO HINH TRONG MLFLOW")
    print("=" * 70)
    df = pd.DataFrame(results)[["run_name", "backbone", "total_parameters", "latency_ms", "final_train_loss", "best_val_loss"]]
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"\nBao cao chi tiet da luu tai: {report_file}")
    print("De xem giao dien MLflow UI, chay lenh: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000\n")


if __name__ == "__main__":
    main()
