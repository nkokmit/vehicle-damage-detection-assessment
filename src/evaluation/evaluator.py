"""Evaluator engine for vehicle damage detection models."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.confusion_matrix import DetectionConfusionMatrix
from src.evaluation.metrics import (
    compute_detection_metrics,
    COCO_IOU_THRESHOLDS,
    DEFAULT_DAMAGE_CLASSES,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluatorConfig:
    """Configuration for model evaluation."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    score_threshold: float = 0.05
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS
    class_names: tuple[str, ...] = DEFAULT_DAMAGE_CLASSES
    use_amp: bool = True
    max_batches: int | None = None


class Evaluator:
    """Detection Evaluator supporting IoU, Precision, Recall, AP, mAP@0.5, mAP@0.5:0.95 and COCOeval."""

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self.config = config or EvaluatorConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available for evaluator. Falling back to CPU.")
            self.device = torch.device("cpu")

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader[Any],
        coco_annotation_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run complete evaluation over a dataset loader.

        Args:
            model: PyTorch Faster R-CNN model.
            data_loader: Evaluation DataLoader yielding (images, targets).
            coco_annotation_file: Optional path to COCO JSON for official pycocotools evaluation.

        Returns:
            Dictionary of aggregate and per-class metrics.
        """
        model.eval()
        model.to(self.device)

        all_predictions: list[dict[str, Any]] = []
        all_targets: list[dict[str, Any]] = []
        coco_results: list[dict[str, Any]] = []

        logger.info(
            "Evaluating model on %s (%d batches, score_thresh=%.2f, AMP=%s)...",
            self.device,
            len(data_loader),
            self.config.score_threshold,
            self.config.use_amp and self.device.type == "cuda",
        )

        start_time = time.perf_counter()
        use_amp = self.config.use_amp and self.device.type == "cuda"

        for batch_idx, (images, targets) in enumerate(data_loader):
            if self.config.max_batches and batch_idx >= self.config.max_batches:
                break

            images_device = [img.to(self.device) for img in images]

            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                # Faster R-CNN in eval mode returns list of prediction dicts
                preds = model(images_device)

            for pred, target in zip(preds, targets):
                p_cpu = {
                    "boxes": pred["boxes"].cpu(),
                    "labels": pred["labels"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "image_id": target.get("image_id", None),
                }
                t_cpu = {
                    "boxes": target["boxes"].cpu() if isinstance(target.get("boxes"), torch.Tensor) else target.get("boxes", []),
                    "labels": target["labels"].cpu() if isinstance(target.get("labels"), torch.Tensor) else target.get("labels", []),
                    "image_id": target.get("image_id", None),
                }

                all_predictions.append(p_cpu)
                all_targets.append(t_cpu)

                # Thu thập định dạng COCO results: [x, y, w, h]
                if coco_annotation_file and target.get("image_id") is not None:
                    img_id = int(target["image_id"].item() if isinstance(target["image_id"], torch.Tensor) else target["image_id"])
                    p_boxes_np = p_cpu["boxes"].numpy()
                    p_labels_np = p_cpu["labels"].numpy()
                    p_scores_np = p_cpu["scores"].numpy()

                    for b, l, s in zip(p_boxes_np, p_labels_np, p_scores_np):
                        if float(s) >= self.config.score_threshold:
                            w = float(b[2] - b[0])
                            h = float(b[3] - b[1])
                            coco_results.append({
                                "image_id": img_id,
                                "category_id": int(l),
                                "bbox": [float(b[0]), float(b[1]), max(0.0, w), max(0.0, h)],
                                "score": float(s),
                            })

        elapsed = time.perf_counter() - start_time
        num_images = len(all_predictions)
        fps = num_images / max(1e-4, elapsed)
        latency_ms = (elapsed / max(1, num_images)) * 1000.0

        logger.info("Inference completed: %d images in %.2fs (%.2f ms/image, %.1f FPS)", num_images, elapsed, latency_ms, fps)

        # 1. Tính toán metric phát hiện chuẩn
        metrics = compute_detection_metrics(
            predictions=all_predictions,
            targets=all_targets,
            iou_thresholds=self.config.iou_thresholds,
            class_names=self.config.class_names,
            score_threshold=self.config.score_threshold,
        )

        metrics["latency_ms"] = latency_ms
        metrics["fps"] = fps
        metrics["total_images"] = num_images

        # 2. Tính Confusion Matrix
        cm = DetectionConfusionMatrix(
            class_names=self.config.class_names,
            iou_threshold=0.5,
            score_threshold=self.config.score_threshold,
        )
        cm.update(all_predictions, all_targets)
        metrics["confusion_matrix"] = cm.get_matrix().tolist()

        # 3. Tính COCOeval chính thức nếu có file annotation COCO
        if coco_annotation_file and Path(coco_annotation_file).exists():
            try:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                coco_gt = COCO(str(coco_annotation_file))
                if len(coco_results) > 0:
                    coco_dt = coco_gt.loadRes(coco_results)
                    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()

                    metrics["coco_official"] = {
                        "mAP_50_95": float(coco_eval.stats[0]),
                        "mAP_50": float(coco_eval.stats[1]),
                        "mAP_75": float(coco_eval.stats[2]),
                        "mAP_small": float(coco_eval.stats[3]),
                        "mAP_medium": float(coco_eval.stats[4]),
                        "mAP_large": float(coco_eval.stats[5]),
                        "mAR_1": float(coco_eval.stats[6]),
                        "mAR_10": float(coco_eval.stats[7]),
                        "mAR_100": float(coco_eval.stats[8]),
                    }
                else:
                    metrics["coco_official"] = {"mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
            except Exception as e:
                logger.warning("Could not execute official pycocotools COCOeval: %s", e)

        return metrics

    def format_markdown_report(self, metrics: dict[str, Any], title: str = "Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe") -> str:
        """Format evaluation metrics into a detailed markdown report."""
        overall_rows = [
            {"Metric": "mAP@0.5:0.95 (COCO)", "Giá trị": f"{metrics.get('mAP_50_95', 0.0):.4f}"},
            {"Metric": "mAP@0.50 (VOC)", "Giá trị": f"{metrics.get('mAP_50', 0.0):.4f}"},
            {"Metric": "Precision@0.50", "Giá trị": f"{metrics.get('precision_50', 0.0):.4f}"},
            {"Metric": "Recall@0.50", "Giá trị": f"{metrics.get('recall_50', 0.0):.4f}"},
            {"Metric": "Mean IoU (Matched)", "Giá trị": f"{metrics.get('mean_iou', 0.0):.4f}"},
            {"Metric": "Tổng Ground Truths", "Giá trị": str(metrics.get("total_gt", 0))},
            {"Metric": "Tổng Dự đoán", "Giá trị": str(metrics.get("total_predictions", 0))},
            {"Metric": "Độ trễ suy luận (ms/ảnh)", "Giá trị": f"{metrics.get('latency_ms', 0.0):.2f} ms"},
            {"Metric": "Tốc độ xử lý (FPS)", "Giá trị": f"{metrics.get('fps', 0.0):.1f}"},
        ]
        df_overall = pd.DataFrame(overall_rows)

        # Bảng per-class breakdown
        per_class = metrics.get("per_class", {})
        class_rows = []
        for cls_name, cls_m in per_class.items():
            class_rows.append({
                "Class": cls_name,
                "GT Boxes": cls_m.get("num_gt", 0),
                "Predictions": cls_m.get("num_pred", 0),
                "TP (@0.5)": cls_m.get("num_tp", 0),
                "FP (@0.5)": cls_m.get("num_fp", 0),
                "Precision": f"{cls_m.get('precision_50', 0.0):.4f}",
                "Recall": f"{cls_m.get('recall_50', 0.0):.4f}",
                "AP@0.50": f"{cls_m.get('ap_50', 0.0):.4f}",
                "AP@0.5:0.95": f"{cls_m.get('ap_50_95', 0.0):.4f}",
                "Mean IoU": f"{cls_m.get('mean_iou', 0.0):.4f}",
            })
        df_class = pd.DataFrame(class_rows)

        report = f"""# {title}

## 1. Các Chỉ số Đánh giá Chính (Key Detection Metrics)

{df_overall.to_markdown(index=False)}

## 2. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

{df_class.to_markdown(index=False)}
"""

        # Nếu có thông số COCO chính thức
        if "coco_official" in metrics:
            coco_m = metrics["coco_official"]
            coco_rows = [
                {"COCO Metric": "AP @[ IoU=0.50:0.95 | area=all ]", "Giá trị": f"{coco_m.get('mAP_50_95', 0.0):.4f}"},
                {"COCO Metric": "AP @[ IoU=0.50 | area=all ]", "Giá trị": f"{coco_m.get('mAP_50', 0.0):.4f}"},
                {"COCO Metric": "AP @[ IoU=0.75 | area=all ]", "Giá trị": f"{coco_m.get('mAP_75', 0.0):.4f}"},
                {"COCO Metric": "AR @[ IoU=0.50:0.95 | maxDets=100 ]", "Giá trị": f"{coco_m.get('mAR_100', 0.0):.4f}"},
            ]
            df_coco = pd.DataFrame(coco_rows)
            report += f"\n## 3. Kết quả pycocotools COCOeval Chuẩn\n\n{df_coco.to_markdown(index=False)}\n"

        return report

    def save_report(
        self,
        metrics: dict[str, Any],
        output_dir: str | Path,
        prefix: str = "eval_results",
    ) -> tuple[Path, Path]:
        """Save evaluation results as JSON and Markdown reports."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / f"{prefix}.json"
        md_file = out_path / f"{prefix}.md"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        md_content = self.format_markdown_report(metrics)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Saved evaluation JSON to: %s", json_file)
        logger.info("Saved evaluation report to: %s", md_file)
        return json_file, md_file


# Hàm tiện ích tương thích ngược
def evaluate() -> dict[str, float]:
    """Return placeholder evaluation metrics."""
    return {"mAP_50": 0.0, "mAP_50_95": 0.0, "precision": 0.0, "recall": 0.0}
