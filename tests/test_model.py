import numpy as np
import pytest
import torch
from src.models.faster_rcnn import FasterRCNNConfig, FasterRCNNModel
from src.models.model_factory import create_model


def test_create_model_default() -> None:
    # Sử dụng pretrained=False để unit test chạy nhanh và offline
    model = create_model(pretrained=False)
    assert model.config.num_classes == 7
    assert len(model.class_names) == 7
    assert model.class_names[0] == "background"
    assert model.class_names[1] == "dent"

    # Kiểm tra classifier head đã được thay thế chính xác
    box_predictor = model.model.roi_heads.box_predictor
    assert box_predictor.cls_score.out_features == 7
    assert box_predictor.bbox_pred.out_features == 28  # 7 classes * 4 coords


def test_model_train_forward_pass() -> None:
    model = create_model(pretrained=False)
    model.train()

    # Tạo dummy input
    dummy_images = [torch.rand(3, 300, 300)]
    dummy_targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]

    losses = model(dummy_images, dummy_targets)
    assert isinstance(losses, dict)

    expected_loss_keys = {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    assert expected_loss_keys.issubset(set(losses.keys()))
    for loss_tensor in losses.values():
        assert isinstance(loss_tensor, torch.Tensor)
        assert not torch.isnan(loss_tensor)


def test_model_eval_forward_pass() -> None:
    model = create_model(pretrained=False)
    model.eval()

    dummy_images = [torch.rand(3, 300, 300)]
    with torch.no_grad():
        predictions = model(dummy_images)

    assert isinstance(predictions, list)
    assert len(predictions) == 1
    pred = predictions[0]
    assert "boxes" in pred
    assert "labels" in pred
    assert "scores" in pred


def test_model_predict_method() -> None:
    model = create_model(pretrained=False)
    dummy_img_np = np.zeros((300, 300, 3), dtype=np.uint8)

    results = model.predict(dummy_img_np, score_threshold=0.0)
    assert isinstance(results, list)
    if len(results) > 0:
        first = results[0]
        assert "bbox" in first
        assert "label" in first
        assert "class_name" in first
        assert "score" in first
