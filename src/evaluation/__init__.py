"""Evaluation package for vehicle damage detection."""

from src.evaluation.confusion_matrix import DetectionConfusionMatrix, empty_confusion_matrix
from src.evaluation.evaluator import Evaluator, EvaluatorConfig, evaluate
from src.evaluation.metrics import (
    box_iou,
    compute_ap_101_point,
    compute_detection_metrics,
    COCO_IOU_THRESHOLDS,
    DEFAULT_DAMAGE_CLASSES,
)

__all__ = [
    "Evaluator",
    "EvaluatorConfig",
    "DetectionConfusionMatrix",
    "box_iou",
    "compute_ap_101_point",
    "compute_detection_metrics",
    "empty_confusion_matrix",
    "evaluate",
    "COCO_IOU_THRESHOLDS",
    "DEFAULT_DAMAGE_CLASSES",
]
