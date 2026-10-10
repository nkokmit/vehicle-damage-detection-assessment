"""Evaluation metrics for object detection (IoU, Precision, Recall, AP, mAP@0.5, mAP@0.5:0.95).

Implements standard COCO and Pascal VOC detection evaluation algorithms:
- Pairwise Bounding Box IoU calculation
- Greedy IoU-based Prediction-to-Ground-Truth matching
- Cumulative True Positive (TP) & False Positive (FP) tracking
- Precision-Recall curves & 101-point COCO interpolated Average Precision (AP)
- mAP@0.5 and mAP@0.5:0.95 across all evaluation thresholds
"""

from collections import defaultdict
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# COCO standard 10 IoU thresholds from 0.50 to 0.95 with step 0.05
COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.50, 1.00, 0.05), 2))
# Standard 6 damage classes + background
DEFAULT_DAMAGE_CLASSES = (
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
)


def box_iou(boxes1: torch.Tensor | np.ndarray, boxes2: torch.Tensor | np.ndarray) -> np.ndarray:
    """Compute pairwise Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1: Array or Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Array or Tensor of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M).
    """
    if isinstance(boxes1, torch.Tensor):
        boxes1 = boxes1.detach().cpu().numpy()
    if isinstance(boxes2, torch.Tensor):
        boxes2 = boxes2.detach().cpu().numpy()

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis, :]
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis, :]

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    # Coordinates of intersection
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    # Areas of individual boxes
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection_area
    union_area = np.maximum(union_area, 1e-8)

    iou = intersection_area / union_area
    return np.clip(iou, 0.0, 1.0)


def compute_ap_101_point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using COCO 101-point interpolation.

    AP = (1 / 101) * sum_{r in [0, 0.01, ..., 1.0]} max_{r' >= r} precision(r')
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = []

    for r_thresh in recall_thresholds:
        # Lấy tất cả precision tại các vị trí recall >= r_thresh
        matched_precisions = precisions[recalls >= r_thresh]
        if len(matched_precisions) > 0:
            interpolated_precisions.append(np.max(matched_precisions))
        else:
            interpolated_precisions.append(0.0)

    return float(np.mean(interpolated_precisions))


def compute_ap_all_points(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using continuous all-point PR curve interpolation."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Chèn điểm neo (0, 1) ở đầu và (1, 0) ở cuối
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Đảm bảo precision là hàm đơn điệu giảm: p(r) = max_{r' >= r} p(r')
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Tìm các điểm thay đổi recall để tính diện tích hình chữ nhật
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def evaluate_class_at_iou(
    pred_records: list[dict[str, Any]],
    gt_records_by_img: dict[Any, list[dict[str, Any]]],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Evaluate detections for a single class at a given IoU threshold.

    Args:
        pred_records: List of prediction dicts: {'image_id', 'box', 'score'}.
        gt_records_by_img: Dict mapping image_id -> list of GT dicts: {'box'}.
        iou_thresh: IoU overlap threshold for a True Positive match.

    Returns:
        Dict with keys: 'ap', 'precision', 'recall', 'mean_iou', 'num_tp', 'num_fp', 'num_gt'.
    """
    total_gt = sum(len(gts) for gts in gt_records_by_img.values())

    if len(pred_records) == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_iou": 0.0,
            "num_tp": 0,
            "num_fp": 0,
            "num_gt": total_gt,
            "matched_ious": [],
            "tp_03": 0,
            "fp_03": 0,
            "tp_05": 0,
            "fp_05": 0,
        }

    # Sắp xếp các dự đoán giảm dần theo confidence score
    sorted_preds = sorted(pred_records, key=lambda x: float(x["score"]), reverse=True)

    # Đánh dấu các GT box đã được match cho từng ảnh
    matched_gt: dict[Any, set[int]] = {img_id: set() for img_id in gt_records_by_img}

    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)
    matched_ious: list[float] = []

    for idx, pred in enumerate(sorted_preds):
        img_id = pred["image_id"]
        pred_box = np.asarray(pred["box"], dtype=np.float32)

        gts = gt_records_by_img.get(img_id, [])
        if len(gts) == 0:
            fp[idx] = 1.0
            continue

        gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32)
        ious = box_iou(pred_box[np.newaxis, :], gt_boxes)[0]

        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_thresh and best_gt_idx not in matched_gt[img_id]:
            tp[idx] = 1.0
            matched_gt[img_id].add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp[idx] = 1.0

    # Tính Precision & Recall tích lũy theo ngưỡng score
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / max(1, total_gt)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)

    ap = compute_ap_101_point(recalls, precisions) if total_gt > 0 else 0.0
    final_prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_rec = float(recalls[-1]) if len(recalls) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0

    # Operational metrics at confidence thresholds 0.30 and 0.50
    scores_arr = np.array([float(p["score"]) for p in sorted_preds], dtype=np.float32)
    mask_03 = scores_arr >= 0.30
    mask_05 = scores_arr >= 0.50
    tp_03 = int(np.sum(tp[mask_03])) if len(scores_arr) > 0 else 0
    fp_03 = int(np.sum(fp[mask_03])) if len(scores_arr) > 0 else 0
    tp_05 = int(np.sum(tp[mask_05])) if len(scores_arr) > 0 else 0
    fp_05 = int(np.sum(fp[mask_05])) if len(scores_arr) > 0 else 0

    return {
        "ap": ap,
        "precision": final_prec,
        "recall": final_rec,
        "mean_iou": mean_iou,
        "num_tp": int(tp_cumsum[-1]),
        "num_fp": int(fp_cumsum[-1]),
        "num_gt": total_gt,
        "matched_ious": matched_ious,
        "tp_03": tp_03,
        "fp_03": fp_03,
        "tp_05": tp_05,
        "fp_05": fp_05,
    }


def compute_detection_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
    class_names: tuple[str, ...] | list[str] = DEFAULT_DAMAGE_CLASSES,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compute comprehensive detection metrics across all classes and IoU thresholds.

    Evaluates:
        - Mean IoU of matched bounding boxes
        - Precision & Recall @ IoU 0.50
        - Per-class AP@0.50 and AP@0.50:0.95
        - mAP@0.50 (Mean Average Precision at IoU=0.50)
        - mAP@0.50:0.95 (COCO standard Mean Average Precision)

    Args:
        predictions: List of dicts, each containing:
            - 'boxes': Tensor or array [M, 4]
            - 'labels': Tensor or array [M] (1-indexed class ID)
            - 'scores': Tensor or array [M]
            - 'image_id' (optional): int or str
        targets: List of dicts, each containing:
            - 'boxes': Tensor or array [K, 4]
            - 'labels': Tensor or array [K] (1-indexed class ID)
            - 'image_id' (optional): int or str
        iou_thresholds: Sequence of IoU thresholds (default 0.50 to 0.95, step 0.05).
        class_names: Names of damage classes (default 6 classes of CarDD).
        score_threshold: Minimum confidence score to consider for predictions.

    Returns:
        Structured dictionary with aggregate metrics and per-class breakdowns.
    """
    num_classes = len(class_names)
    # Gom nhóm dữ liệu theo từng class (class ID 1 đến num_classes)
    preds_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gts_by_class: dict[int, dict[Any, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    total_matched_ious_05: list[float] = []

    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        img_id = pred.get("image_id", target.get("image_id", idx))

        # 1. Trích xuất Ground Truths
        gt_boxes = target.get("boxes", [])
        gt_labels = target.get("labels", [])
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.detach().cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.detach().cpu().numpy()

        for box, label in zip(gt_boxes, gt_labels):
            cls_id = int(label)
            gts_by_class[cls_id][img_id].append({"box": box})

        # 2. Trích xuất Predictions (lọc theo score_threshold)
        p_boxes = pred.get("boxes", [])
        p_labels = pred.get("labels", [])
        p_scores = pred.get("scores", [])
        if isinstance(p_boxes, torch.Tensor):
            p_boxes = p_boxes.detach().cpu().numpy()
        if isinstance(p_labels, torch.Tensor):
            p_labels = p_labels.detach().cpu().numpy()
        if isinstance(p_scores, torch.Tensor):
            p_scores = p_scores.detach().cpu().numpy()

        for box, label, score in zip(p_boxes, p_labels, p_scores):
            if float(score) >= score_threshold:
                cls_id = int(label)
                preds_by_class[cls_id].append({
                    "image_id": img_id,
                    "box": box,
                    "score": float(score),
                })

    per_class_summary: dict[str, dict[str, Any]] = {}
    aps_at_50: list[float] = []
    aps_across_all_ious: list[float] = []
    class_precisions_50: list[float] = []
    class_recalls_50: list[float] = []
    total_tp_50 = 0
    total_fp_50 = 0
    total_gt_all = 0
    total_tp_03 = 0
    total_fp_03 = 0
    total_tp_05 = 0
    total_fp_05 = 0

    for cls_idx, cls_name in enumerate(class_names, start=1):
        class_preds = preds_by_class.get(cls_idx, [])
        class_gts = gts_by_class.get(cls_idx, {})

        # Đánh giá tại IoU 0.50
        res_50 = evaluate_class_at_iou(class_preds, class_gts, iou_thresh=0.50)
        ap_50 = res_50["ap"]
        prec_50 = res_50["precision"]
        rec_50 = res_50["recall"]
        mean_iou_cls = res_50["mean_iou"]
        total_matched_ious_05.extend(res_50["matched_ious"])

        total_tp_50 += res_50["num_tp"]
        total_fp_50 += res_50["num_fp"]
        total_gt_all += res_50["num_gt"]
        total_tp_03 += res_50.get("tp_03", 0)
        total_fp_03 += res_50.get("fp_03", 0)
        total_tp_05 += res_50.get("tp_05", 0)
        total_fp_05 += res_50.get("fp_05", 0)

        # Operational metrics per class at score >= 0.50
        cls_tp_05 = res_50.get("tp_05", 0)
        cls_fp_05 = res_50.get("fp_05", 0)
        cls_prec_05 = float(cls_tp_05 / max(1, cls_tp_05 + cls_fp_05)) if (cls_tp_05 + cls_fp_05) > 0 else 0.0
        cls_rec_05 = float(cls_tp_05 / max(1, res_50["num_gt"])) if res_50["num_gt"] > 0 else 0.0
        cls_f1_05 = float((2 * cls_prec_05 * cls_rec_05) / max(1e-8, cls_prec_05 + cls_rec_05))

        # Đánh giá qua toàn bộ dải IoU [0.50:0.95]
        aps_iou_series: list[float] = []
        for thresh in iou_thresholds:
            if abs(thresh - 0.50) < 1e-4:
                aps_iou_series.append(ap_50)
            else:
                res_t = evaluate_class_at_iou(class_preds, class_gts, iou_thresh=thresh)
                aps_iou_series.append(res_t["ap"])

        ap_50_95 = float(np.mean(aps_iou_series)) if len(aps_iou_series) > 0 else 0.0

        aps_at_50.append(ap_50)
        aps_across_all_ious.append(ap_50_95)
        class_precisions_50.append(prec_50)
        class_recalls_50.append(rec_50)

        per_class_summary[cls_name] = {
            "ap_50": ap_50,
            "ap_50_95": ap_50_95,
            "precision_50": prec_50,
            "recall_50": rec_50,
            "mean_iou": mean_iou_cls,
            "num_gt": res_50["num_gt"],
            "num_pred": len(class_preds),
            "num_tp": res_50["num_tp"],
            "num_fp": res_50["num_fp"],
            "tp_score05": cls_tp_05,
            "fp_score05": cls_fp_05,
            "precision_score05": cls_prec_05,
            "recall_score05": cls_rec_05,
            "f1_score05": cls_f1_05,
        }

    # Tổng hợp toàn cục
    mAP_50 = float(np.mean(aps_at_50)) if len(aps_at_50) > 0 else 0.0
    mAP_50_95 = float(np.mean(aps_across_all_ious)) if len(aps_across_all_ious) > 0 else 0.0
    macro_precision_50 = float(np.mean(class_precisions_50)) if len(class_precisions_50) > 0 else 0.0
    macro_recall_50 = float(np.mean(class_recalls_50)) if len(class_recalls_50) > 0 else 0.0
    micro_precision_50 = float(total_tp_50 / max(1, total_tp_50 + total_fp_50))
    micro_recall_50 = float(total_tp_50 / max(1, total_gt_all))
    f1_50 = float((2 * micro_precision_50 * micro_recall_50) / max(1e-8, micro_precision_50 + micro_recall_50))
    overall_mean_iou = float(np.mean(total_matched_ious_05)) if len(total_matched_ious_05) > 0 else 0.0

    # Operational metrics (ngưỡng tin cậy thực tế 0.30 và 0.50)
    precision_score03 = float(total_tp_03 / max(1, total_tp_03 + total_fp_03)) if (total_tp_03 + total_fp_03) > 0 else 0.0
    recall_score03 = float(total_tp_03 / max(1, total_gt_all)) if total_gt_all > 0 else 0.0
    f1_score03 = float((2 * precision_score03 * recall_score03) / max(1e-8, precision_score03 + recall_score03))

    precision_score05 = float(total_tp_05 / max(1, total_tp_05 + total_fp_05)) if (total_tp_05 + total_fp_05) > 0 else 0.0
    recall_score05 = float(total_tp_05 / max(1, total_gt_all)) if total_gt_all > 0 else 0.0
    f1_score05 = float((2 * precision_score05 * recall_score05) / max(1e-8, precision_score05 + recall_score05))

    return {
        "mean_iou": overall_mean_iou,
        "precision_50": micro_precision_50,
        "recall_50": micro_recall_50,
        "f1_50": f1_50,
        "macro_precision_50": macro_precision_50,
        "macro_recall_50": macro_recall_50,
        "precision_score03": precision_score03,
        "recall_score03": recall_score03,
        "f1_score03": f1_score03,
        "precision_score05": precision_score05,
        "recall_score05": recall_score05,
        "f1_score05": f1_score05,
        "tp_score05": total_tp_05,
        "fp_score05": total_fp_05,
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "num_classes": num_classes,
        "total_gt": total_gt_all,
        "total_predictions": sum(len(p) for p in preds_by_class.values()),
        "per_class": per_class_summary,
        "iou_thresholds": list(iou_thresholds),
    }
