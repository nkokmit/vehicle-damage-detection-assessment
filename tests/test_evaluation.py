import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.confusion_matrix import DetectionConfusionMatrix
from src.evaluation.evaluator import Evaluator, EvaluatorConfig
from src.evaluation.metrics import (
    box_iou,
    compute_ap_101_point,
    compute_detection_metrics,
    evaluate_class_at_iou,
)


def test_box_iou_perfect_and_disjoint() -> None:
    # 1. Hai box trùng khít hoàn toàn -> IoU = 1.0
    box_a = np.array([[10.0, 10.0, 50.0, 50.0]])
    box_b = np.array([[10.0, 10.0, 50.0, 50.0]])
    iou = box_iou(box_a, box_b)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(1.0)

    # 2. Hai box tách rời hoàn toàn -> IoU = 0.0
    box_c = np.array([[100.0, 100.0, 200.0, 200.0]])
    iou_disjoint = box_iou(box_a, box_c)
    assert iou_disjoint[0, 0] == pytest.approx(0.0)

    # 3. Hai box giao nhau 50%
    # Box 1: [0, 0, 10, 10] -> Area = 100
    # Box 2: [5, 0, 15, 10] -> Area = 100
    # Intersection: [5, 0, 10, 10] -> Area = 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3 ~ 0.3333
    b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    b2 = np.array([[5.0, 0.0, 15.0, 10.0]])
    iou_partial = box_iou(b1, b2)
    assert iou_partial[0, 0] == pytest.approx(50.0 / 150.0)


def test_compute_ap_101_point() -> None:
    # Perfect detection
    recalls = np.array([0.2, 0.5, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0])
    ap = compute_ap_101_point(recalls, precisions)
    assert ap == pytest.approx(1.0)

    # Empty detection
    empty_ap = compute_ap_101_point(np.array([]), np.array([]))
    assert empty_ap == pytest.approx(0.0)


def test_evaluate_class_at_iou() -> None:
    # 1 GT box tại [10, 10, 50, 50]
    gt_records = {
        "img1": [{"box": [10.0, 10.0, 50.0, 50.0]}]
    }

    # 2 predictions: 1 đúng (score 0.9, IoU=1.0) và 1 trượt (score 0.4, tách rời)
    pred_records = [
        {"image_id": "img1", "box": [10.0, 10.0, 50.0, 50.0], "score": 0.9},
        {"image_id": "img1", "box": [200.0, 200.0, 300.0, 300.0], "score": 0.4},
    ]

    res = evaluate_class_at_iou(pred_records, gt_records, iou_thresh=0.5)
    assert res["num_gt"] == 1
    assert res["num_tp"] == 1
    assert res["num_fp"] == 1
    assert res["precision"] == pytest.approx(0.5)  # 1 / (1 + 1)
    assert res["recall"] == pytest.approx(1.0)     # 1 / 1
    assert res["ap"] == pytest.approx(1.0)         # Max precision at r=1.0 was 1.0 before the FP
    assert res["mean_iou"] == pytest.approx(1.0)


def test_compute_detection_metrics_all() -> None:
    # Test toàn bộ metrics trên tập dữ liệu nhỏ
    targets = [
        {
            "image_id": 1,
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]),
            "labels": torch.tensor([1, 2]),  # dent, scratch
        }
    ]

    predictions = [
        {
            "image_id": 1,
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.95, 0.88]),
        }
    ]

    metrics = compute_detection_metrics(predictions, targets)
    assert "mAP_50" in metrics
    assert "mAP_50_95" in metrics
    assert "precision_50" in metrics
    assert "recall_50" in metrics
    assert "mean_iou" in metrics
    assert metrics["mAP_50"] > 0.0
    assert metrics["mAP_50_95"] > 0.0
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["precision_50"] == pytest.approx(1.0)
    assert metrics["recall_50"] == pytest.approx(1.0)
    assert "precision_score05" in metrics
    assert "recall_score05" in metrics
    assert "f1_score05" in metrics
    assert "f1_50" in metrics
    assert metrics["precision_score05"] == pytest.approx(1.0)
    assert metrics["recall_score05"] == pytest.approx(1.0)
    assert metrics["f1_score05"] == pytest.approx(1.0)


def test_confusion_matrix_structure() -> None:
    cm = DetectionConfusionMatrix(class_names=["dent", "scratch", "crack"], iou_threshold=0.5)
    assert cm.matrix.shape == (4, 4)  # 3 classes + 1 background

    targets = [
        {
            "boxes": [[10, 10, 50, 50]],
            "labels": [1],  # dent
        }
    ]
    predictions = [
        {
            "boxes": [[10, 10, 50, 50]],
            "labels": [1],
            "scores": [0.9],
        }
    ]
    cm.update(predictions, targets)
    mat = cm.get_matrix()
    # Row 0 (dent) -> Col 0 (dent) = 1 (TP)
    assert mat[0, 0] == 1
    assert mat.sum() == 1


class DummyModel(torch.nn.Module):
    def forward(self, images):
        # Trả về 1 prediction dummy cho mỗi ảnh
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
            for _ in images
        ]


class DummyEvalDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        img = torch.rand(3, 100, 100)
        target = {
            "image_id": idx + 1,
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
        }
        return img, target


def dummy_collate(batch):
    imgs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return imgs, targets


def test_evaluator_end_to_end() -> None:
    model = DummyModel()
    dataset = DummyEvalDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=dummy_collate)

    evaluator = Evaluator(EvaluatorConfig(device="cpu", score_threshold=0.5))
    metrics = evaluator.evaluate(model, loader)

    assert "mAP_50" in metrics
    assert "mAP_50_95" in metrics
    assert "mean_iou" in metrics
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["mAP_50"] == pytest.approx(1.0 / 6.0)


def test_plot_confusion_matrix(tmp_path) -> None:
    from src.evaluation.visualization import plot_confusion_matrix
    mat = np.eye(7, dtype=int)
    labels = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat", "background"]
    save_file = tmp_path / "test_cm.png"
    out = plot_confusion_matrix(mat, labels, save_path=save_file)
    assert out is not None
    assert save_file.exists()

