"""Confusion matrix computation and visualization for object detection."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.metrics import box_iou, DEFAULT_DAMAGE_CLASSES

logger = logging.getLogger(__name__)


class DetectionConfusionMatrix:
    """Confusion Matrix for object detection with background class handling.

    Shape: (num_classes + 1, num_classes + 1)
    Rows represent Ground Truth, Columns represent Predictions.
    The last index (num_classes) represents 'background' (missed GT or hallucinated pred).
    """

    def __init__(
        self,
        class_names: tuple[str, ...] | list[str] = DEFAULT_DAMAGE_CLASSES,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.3,
    ) -> None:
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        # Kích thước (N+1, N+1): N classes + 1 background
        self.matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int64)

    @property
    def background_index(self) -> int:
        return self.num_classes

    def update(
        self,
        predictions: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> None:
        """Update confusion matrix with a batch of predictions and ground truths."""
        for pred, target in zip(predictions, targets):
            gt_boxes = target.get("boxes", [])
            gt_labels = target.get("labels", [])
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.detach().cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.detach().cpu().numpy()

            p_boxes = pred.get("boxes", [])
            p_labels = pred.get("labels", [])
            p_scores = pred.get("scores", [])
            if isinstance(p_boxes, torch.Tensor):
                p_boxes = p_boxes.detach().cpu().numpy()
            if isinstance(p_labels, torch.Tensor):
                p_labels = p_labels.detach().cpu().numpy()
            if isinstance(p_scores, torch.Tensor):
                p_scores = p_scores.detach().cpu().numpy()

            # Lọc dự đoán theo score threshold
            valid_mask = [float(s) >= self.score_threshold for s in p_scores] if len(p_scores) > 0 else []
            if len(valid_mask) > 0 and any(valid_mask):
                p_boxes = np.array([b for b, v in zip(p_boxes, valid_mask) if v], dtype=np.float32)
                p_labels = np.array([l for l, v in zip(p_labels, valid_mask) if v], dtype=np.int64)
            else:
                p_boxes = np.empty((0, 4), dtype=np.float32)
                p_labels = np.empty((0,), dtype=np.int64)

            # Trường hợp 1: Không có cả GT và Prediction
            if len(gt_boxes) == 0 and len(p_boxes) == 0:
                continue

            # Trường hợp 2: Có dự đoán nhưng không có GT -> Tất cả dự đoán là False Positive (Background -> Pred)
            if len(gt_boxes) == 0:
                for p_lbl in p_labels:
                    pred_idx = int(p_lbl) - 1
                    if 0 <= pred_idx < self.num_classes:
                        self.matrix[self.background_index, pred_idx] += 1
                continue

            # Trường hợp 3: Có GT nhưng không có dự đoán -> Tất cả GT bị bỏ lỡ (GT -> Background)
            if len(p_boxes) == 0:
                for g_lbl in gt_labels:
                    gt_idx = int(g_lbl) - 1
                    if 0 <= gt_idx < self.num_classes:
                        self.matrix[gt_idx, self.background_index] += 1
                continue

            # Trường hợp 4: Cả 2 đều có -> Tính ma trận IoU và khớp
            ious = box_iou(gt_boxes, p_boxes)
            matched_gt = set()
            matched_pred = set()

            # Khớp theo thứ tự IoU cao nhất
            gt_indices, pred_indices = np.where(ious >= self.iou_threshold)
            matches = []
            for g_i, p_i in zip(gt_indices, pred_indices):
                matches.append((ious[g_i, p_i], g_i, p_i))
            matches.sort(reverse=True, key=lambda x: x[0])

            for _, g_i, p_i in matches:
                if g_i not in matched_gt and p_i not in matched_pred:
                    matched_gt.add(g_i)
                    matched_pred.add(p_i)
                    gt_idx = int(gt_labels[g_i]) - 1
                    pred_idx = int(p_labels[p_i]) - 1
                    if 0 <= gt_idx < self.num_classes and 0 <= pred_idx < self.num_classes:
                        self.matrix[gt_idx, pred_idx] += 1

            # Các GT chưa được khớp -> Missed (GT -> Background)
            for g_i in range(len(gt_boxes)):
                if g_i not in matched_gt:
                    gt_idx = int(gt_labels[g_i]) - 1
                    if 0 <= gt_idx < self.num_classes:
                        self.matrix[gt_idx, self.background_index] += 1

            # Các Prediction chưa được khớp -> Hallucination (Background -> Pred)
            for p_i in range(len(p_boxes)):
                if p_i not in matched_pred:
                    pred_idx = int(p_labels[p_i]) - 1
                    if 0 <= pred_idx < self.num_classes:
                        self.matrix[self.background_index, pred_idx] += 1

    def get_matrix(self) -> np.ndarray:
        return self.matrix.copy()

    def get_labels(self) -> list[str]:
        return self.class_names + ["background"]

    def to_dataframe(self):
        import pandas as pd
        labels = self.get_labels()
        return pd.DataFrame(self.matrix, index=labels, columns=labels)


# Tương thích ngược với empty_confusion_matrix
def empty_confusion_matrix(num_classes: int) -> list[list[int]]:
    """Create a square zero confusion matrix."""
    return [[0 for _ in range(num_classes)] for _ in range(num_classes)]
