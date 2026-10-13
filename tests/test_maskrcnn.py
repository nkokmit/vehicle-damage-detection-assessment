"""Unit tests for Mask R-CNN model and mask loading pipeline."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.loader import build_cardd_dataset, build_dataloader
from src.models.mask_rcnn import MaskRCNNConfig, MaskRCNNModel
from src.models.model_factory import create_model

CARDD_TRAIN_JSON = Path("data/CarDD_COCO/annotations/instances_train2017.json")


def test_create_maskrcnn_model() -> None:
    model = create_model(model_name="mask_rcnn", backbone="resnet50_fpn_v2", pretrained=False)
    assert isinstance(model, MaskRCNNModel)
    assert model.config.num_classes == 7
    assert model.config.backbone == "resnet50_fpn_v2"
    assert len(model.class_names) == 7

    # Kiểm tra box predictor head
    box_predictor = model.model.roi_heads.box_predictor
    assert box_predictor.cls_score.out_features == 7
    assert box_predictor.bbox_pred.out_features == 28

    # Kiểm tra mask predictor head
    mask_predictor = model.model.roi_heads.mask_predictor
    assert mask_predictor.mask_fcn_logits.out_channels == 7


def test_maskrcnn_train_forward_pass() -> None:
    model = create_model(model_name="mask_rcnn", backbone="resnet50_fpn", pretrained=False)
    model.train()

    dummy_images = [torch.rand(3, 256, 256)]
    dummy_masks = torch.zeros((1, 256, 256), dtype=torch.uint8)
    dummy_masks[0, 10:50, 10:50] = 1

    dummy_targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": dummy_masks,
        }
    ]

    losses = model(dummy_images, dummy_targets)
    assert isinstance(losses, dict)

    expected_loss_keys = {
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    assert expected_loss_keys.issubset(set(losses.keys()))
    for loss_tensor in losses.values():
        assert isinstance(loss_tensor, torch.Tensor)
        assert not torch.isnan(loss_tensor)
        assert not torch.isinf(loss_tensor)


def test_maskrcnn_eval_forward_pass() -> None:
    model = create_model(model_name="mask_rcnn", backbone="resnet50_fpn", pretrained=False)
    model.eval()

    dummy_images = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        predictions = model(dummy_images)

    assert isinstance(predictions, list)
    assert len(predictions) == 1
    pred = predictions[0]
    assert "boxes" in pred
    assert "labels" in pred
    assert "scores" in pred
    assert "masks" in pred
    assert pred["masks"].dim() == 4  # [N, 1, H, W]


def test_maskrcnn_predict_method() -> None:
    model = create_model(model_name="mask_rcnn", backbone="resnet50_fpn", pretrained=False)
    dummy_img_np = np.zeros((256, 256, 3), dtype=np.uint8)

    results = model.predict(dummy_img_np, score_threshold=0.0)
    assert isinstance(results, list)
    if len(results) > 0:
        first = results[0]
        assert "bbox" in first
        assert "label" in first
        assert "class_name" in first
        assert "score" in first
        assert "mask" in first
        assert isinstance(first["mask"], np.ndarray)


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataset_with_masks() -> None:
    dataset = build_cardd_dataset(
        "train",
        cardd_dir="data/CarDD_COCO",
        img_size=(512, 512),
        use_augmentation=False,
        load_masks=True,
    )
    img_tensor, target = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert "masks" in target
    assert isinstance(target["masks"], torch.Tensor)
    assert target["masks"].dtype == torch.uint8
    assert target["masks"].shape[0] == len(target["boxes"])
    assert target["masks"].shape[1:] == img_tensor.shape[1:]


@pytest.mark.skipif(not CARDD_TRAIN_JSON.exists(), reason="CarDD dataset not present")
def test_cardd_dataloader_collate_with_masks() -> None:
    dataset = build_cardd_dataset(
        "train",
        cardd_dir="data/CarDD_COCO",
        img_size=(256, 256),
        use_augmentation=False,
        load_masks=True,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_imgs, batch_targets = next(iter(loader))

    assert len(batch_imgs) == 2
    assert len(batch_targets) == 2
    for img, target in zip(batch_imgs, batch_targets):
        assert "masks" in target
        assert target["masks"].dtype == torch.uint8
        assert target["masks"].shape[0] == len(target["boxes"])

