"""CLI entrypoint for training Faster R-CNN vehicle damage detection models.

Supports:
- Device: CUDA
- Custom epochs, batch size, and image resolution (min_size, max_size = 512)
- SGD optimizer with momentum=0.9, weight_decay=0.0005, lr=0.005
- StepLR scheduler (step_size=5, gamma=0.1)
- Mixed Precision (AMP) via torch.amp
- Gradient Accumulation
- Full MLflow experiment tracking & model checkpointing
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.loader import build_cardd_dataset, build_dataloader
from src.evaluation.evaluator import Evaluator, EvaluatorConfig
from src.models.model_factory import create_model
from src.tracking.mlflow_tracker import MLflowConfig, MLflowTracker
from src.training.trainer import Trainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file if it exists."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on CarDD dataset")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config YAML")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU step")
    parser.add_argument("--image-size", type=int, default=None, help="Image size (both min_size and max_size)")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--scheduler-type", type=str, default=None, help="Scheduler type ('step' or 'cosine')")
    parser.add_argument("--step-size", type=int, default=None, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma factor for StepLR")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable Mixed Precision (AMP)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable Mixed Precision")
    parser.add_argument("--accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone ('resnet50_fpn' or 'resnet50_fpn_v2')")
    parser.add_argument("--cardd-dir", type=str, default="data/CarDD_COCO", help="Path to CarDD dataset")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader num workers")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume training")
    parser.add_argument("--eval-map-interval", type=int, default=None, help="Interval (epochs) to evaluate mAP (default 5, 0 to disable)")
    parser.add_argument("--dry-run", action="store_true", help="Quick run with 4 train batches and 2 val batches for testing")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Nạp cấu hình từ train.yaml và model.yaml làm mặc định
    train_yaml = load_yaml_config(PROJECT_ROOT / args.config)
    model_yaml = load_yaml_config(PROJECT_ROOT / args.model_config)

    # Lấy giá trị cấu hình ưu tiên: CLI args > YAML config > default
    device_str = args.device or train_yaml.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or train_yaml.get("epochs", 40)
    batch_size = args.batch_size or train_yaml.get("batch_size", 2)
    image_size = args.image_size or train_yaml.get("image_size", 512)
    min_size = image_size
    max_size = image_size
    lr = args.lr or train_yaml.get("learning_rate", 0.005)
    momentum = args.momentum or train_yaml.get("momentum", 0.9)
    weight_decay = args.weight_decay or train_yaml.get("weight_decay", 0.0005)
    backbone = args.backbone or model_yaml.get("backbone", "resnet50_fpn")
    num_workers = args.num_workers if args.num_workers is not None else train_yaml.get("num_workers", 2)

    # Scheduler settings
    sched_conf = train_yaml.get("scheduler", {})
    scheduler_type = args.scheduler_type or sched_conf.get("type", "step")
    step_size = args.step_size or sched_conf.get("step_size", 5)
    gamma = args.gamma or sched_conf.get("gamma", 0.1)

    # AMP & Gradient Accumulation
    use_amp = args.amp if args.amp is not None else train_yaml.get("use_amp", True)
    accum_steps = args.accum_steps or train_yaml.get("gradient_accumulation_steps", 2)
    clip_grad_norm = train_yaml.get("clip_grad_norm", 10.0)
    eval_map_interval = args.eval_map_interval if args.eval_map_interval is not None else train_yaml.get("eval_map_interval", 5)

    # Dry-run override nếu cần kiểm tra nhanh
    max_train_batches = None
    max_val_batches = None
    if args.dry_run:
        epochs = 1
        max_train_batches = 4
        max_val_batches = 2
        logger.info("Executing in DRY-RUN mode (1 epoch, 4 train batches, 2 val batches)")

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    logger.info("=" * 65)
    logger.info("        VEHICLE DAMAGE DETECTION - TRAINING PIPELINE")
    logger.info("=" * 65)
    logger.info("Device                   : %s (%s)", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    logger.info("Backbone                 : %s", backbone)
    logger.info("Epochs                   : %d", epochs)
    logger.info("Batch Size               : %d", batch_size)
    logger.info("Image Size (min, max)    : (%d, %d)", min_size, max_size)
    logger.info("Optimizer                : SGD (lr=%.4f, momentum=%.2f, weight_decay=%.5f)", lr, momentum, weight_decay)
    logger.info("Scheduler                : %s (step_size=%d, gamma=%.2f)", scheduler_type, step_size, gamma)
    logger.info("Mixed Precision (AMP)    : %s", use_amp)
    logger.info("Gradient Accumulation    : %d (Effective Batch Size = %d)", accum_steps, batch_size * accum_steps)
    logger.info("=" * 65)

    # 2. Xây dựng Dataset với image_size=512 (min và max)
    logger.info("Loading CarDD dataset from: %s", args.cardd_dir)
    train_dataset = build_cardd_dataset(
        split="train",
        cardd_dir=args.cardd_dir,
        img_size=(image_size, image_size),
        use_augmentation=True,
    )
    val_dataset = build_cardd_dataset(
        split="val",
        cardd_dir=args.cardd_dir,
        img_size=(image_size, image_size),
        use_augmentation=False,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 3. Khởi tạo Faster R-CNN Model với 7 classes và min_size=512, max_size=512
    logger.info("Building Faster R-CNN with backbone: %s", backbone)
    model = create_model(
        model_name="faster_rcnn",
        backbone=backbone,
        num_classes=7,
        pretrained=True,
        trainable_backbone_layers=3,
        min_size=min_size,
        max_size=max_size,
    )
    param_counts = model.count_parameters()
    logger.info(
        "Model Parameters: Total: %s | Trainable: %s",
        f"{param_counts['total_parameters']:,}",
        f"{param_counts['trainable_parameters']:,}",
    )

    # 4. Thiết lập MLflow Tracker
    mlflow_config = MLflowConfig(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="vehicle-damage-detection",
    )
    tracker = MLflowTracker(config=mlflow_config)

    run_name = args.run_name or f"faster_rcnn_{backbone}_amp_accum_b{batch_size}_ep{epochs}"
    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / run_name

    trainer_config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        optimizer="SGD",
        device=str(device),
        scheduler_type=scheduler_type,
        scheduler_step_size=step_size,
        scheduler_gamma=gamma,
        use_amp=use_amp,
        gradient_accumulation_steps=accum_steps,
        clip_grad_norm=clip_grad_norm,
        min_size=min_size,
        max_size=max_size,
        eval_map_interval=eval_map_interval,
        checkpoint_dir=str(checkpoint_dir),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        log_interval=10,
    )

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tracker=tracker,
    )

    if args.resume:
        logger.info("Resuming training from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    tags = {
        "model.family": "Faster R-CNN",
        "model.backbone": backbone,
        "task": "vehicle-damage-detection",
        "training.amp": str(use_amp),
        "training.grad_accum": str(accum_steps),
        "training.image_size": f"{image_size}x{image_size}",
    }

    # 5. Huấn luyện mô hình và log vào MLflow
    with tracker.run(run_name=run_name, tags=tags, description=f"Faster R-CNN {backbone} trained with AMP & Gradient Accumulation"):
        tracker.log_params({
            "model_architecture": "faster_rcnn",
            "backbone": backbone,
            "num_classes": 7,
            "epochs": epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * accum_steps,
            "image_size": image_size,
            "min_size": min_size,
            "max_size": max_size,
            "optimizer": "SGD",
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "scheduler_type": scheduler_type,
            "scheduler_step_size": step_size,
            "scheduler_gamma": gamma,
            "use_amp": use_amp,
            "gradient_accumulation_steps": accum_steps,
            "eval_map_interval": eval_map_interval,
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
            "device": str(device),
        })

        logger.info("Starting training run: %s", run_name)
        history_summary = trainer.train()

        tracker.log_metrics({
            "final_train_loss": history_summary["final_train_loss"],
            "best_val_loss": history_summary["best_val_loss"],
            "best_val_mAP_50": history_summary.get("best_val_map", 0.0),
        })

        # 6. Đánh giá toàn diện trên tập Validation với checkpoint tốt nhất
        logger.info("Performing final comprehensive evaluation on validation set...")
        best_map_chkpt = checkpoint_dir / "best_map_model.pth"
        best_loss_chkpt = checkpoint_dir / "best_model.pth"
        eval_chkpt = best_map_chkpt if best_map_chkpt.exists() else best_loss_chkpt
        if eval_chkpt.exists():
            logger.info("Loading checkpoint for final evaluation: %s", eval_chkpt)
            chkpt_data = torch.load(eval_chkpt, map_location=device, weights_only=False)
            model.load_state_dict(chkpt_data["model_state_dict"])

        eval_cfg = EvaluatorConfig(
            device=str(device),
            score_threshold=0.05,
            use_amp=use_amp,
            max_batches=max_val_batches,
        )
        evaluator = Evaluator(eval_cfg)
        coco_val_json = PROJECT_ROOT / args.cardd_dir / "annotations" / "instances_val.json"
        final_eval_metrics = evaluator.evaluate(
            model=model,
            data_loader=val_loader,
            coco_annotation_file=coco_val_json if coco_val_json.exists() else None,
        )

        reports_dir = PROJECT_ROOT / "artifacts" / "reports" / run_name
        json_path, md_path = evaluator.save_report(
            final_eval_metrics,
            output_dir=reports_dir,
            prefix="final_eval_report",
        )

        tracker.log_metrics({
            "final_eval_mAP_50": final_eval_metrics["mAP_50"],
            "final_eval_mAP_50_95": final_eval_metrics["mAP_50_95"],
            "final_eval_precision_50": final_eval_metrics["precision_50"],
            "final_eval_recall_50": final_eval_metrics["recall_50"],
            "final_eval_mean_iou": final_eval_metrics["mean_iou"],
            "final_eval_fps": final_eval_metrics["fps"],
            "final_eval_latency_ms": final_eval_metrics["latency_ms"],
        })
        tracker.log_artifact(str(md_path), artifact_path="reports")
        tracker.log_artifact(str(json_path), artifact_path="reports")
        if best_map_chkpt.exists():
            tracker.log_artifact(str(best_map_chkpt), artifact_path="checkpoints")
        if best_loss_chkpt.exists():
            tracker.log_artifact(str(best_loss_chkpt), artifact_path="checkpoints")

    logger.info("=" * 65)
    logger.info("TRAINING & EVALUATION FINISHED SUCCESSFULLY!")
    logger.info("Best Validation Loss    : %.4f", history_summary["best_val_loss"])
    logger.info("Best Validation mAP@0.5 : %.4f", history_summary.get("best_val_map", 0.0))
    logger.info("Final Val mAP@0.5: %.4f | mAP@0.5:0.95: %.4f", final_eval_metrics["mAP_50"], final_eval_metrics["mAP_50_95"])
    logger.info("Final Val Precision: %.4f | Recall: %.4f", final_eval_metrics["precision_50"], final_eval_metrics["recall_50"])
    logger.info("Checkpoints Saved In    : %s", checkpoint_dir)
    logger.info("Evaluation Report       : %s", md_path)
    logger.info("MLflow Experiment       : 'vehicle-damage-detection' (Run: %s)", run_name)
    logger.info("To view metrics & graphs: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
