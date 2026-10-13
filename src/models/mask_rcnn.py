"""Mask R-CNN model implementation for vehicle damage instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

logger = logging.getLogger(__name__)

# Danh sách nhãn: 0 là background, 1..6 là các loại tổn thương
DAMAGE_CLASSES: list[str] = [
    "background",
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


@dataclass
class MaskRCNNConfig:
    """Cấu hình cho mô hình Mask R-CNN Instance Segmentation."""

    model_name: str = "mask_rcnn"
    backbone: str = "resnet50_fpn_v2"  # 'resnet50_fpn' hoặc 'resnet50_fpn_v2'
    pretrained: bool = True
    trainable_backbone_layers: int = 3
    num_classes: int = 7
    class_names: list[str] = field(default_factory=lambda: list(DAMAGE_CLASSES))
    min_size: int = 512
    max_size: int = 512
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100


def build_mask_rcnn_model(config: MaskRCNNConfig | None = None) -> MaskRCNN:
    """Xây dựng và thay thế classification head & mask head cho Mask R-CNN.

    Args:
        config: Cấu hình MaskRCNNConfig.

    Returns:
        Mô hình MaskRCNN đã được tùy chỉnh cho 7 classes.
    """
    cfg = config or MaskRCNNConfig()
    backbone_name = cfg.backbone.lower()

    if backbone_name in ("resnet50_fpn_v2", "resnet50_v2", "v2"):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if cfg.pretrained else None
        logger.info(
            "Building Mask R-CNN ResNet-50 FPN V2 (pretrained=%s, weights=%s)",
            cfg.pretrained,
            weights,
        )
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=cfg.trainable_backbone_layers,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            box_score_thresh=cfg.box_score_thresh,
            box_nms_thresh=cfg.box_nms_thresh,
            box_detections_per_img=cfg.box_detections_per_img,
        )
    elif backbone_name in ("resnet50_fpn", "resnet50", "v1"):
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if cfg.pretrained else None
        logger.info(
            "Building Mask R-CNN ResNet-50 FPN V1 (pretrained=%s, weights=%s)",
            cfg.pretrained,
            weights,
        )
        model = maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=cfg.trainable_backbone_layers,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            box_score_thresh=cfg.box_score_thresh,
            box_nms_thresh=cfg.box_nms_thresh,
            box_detections_per_img=cfg.box_detections_per_img,
        )
    else:
        raise ValueError(
            f"Unsupported backbone: '{cfg.backbone}'. Choose 'resnet50_fpn' (v1) or 'resnet50_fpn_v2' (v2)."
        )

    # 1. Thay thế Box Predictor cho số class của CarDD (7 classes: 1 background + 6 damages)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features,
        num_classes=cfg.num_classes,
    )

    # 2. Thay thế Mask Predictor cho 7 classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=dim_reduced,
        num_classes=cfg.num_classes,
    )

    logger.info(
        "Configured Mask R-CNN [%s]: box in_features=%d, mask in_channels=%d -> num_classes=%d",
        cfg.backbone,
        in_features,
        in_features_mask,
        cfg.num_classes,
    )

    return model


class MaskRCNNModel(nn.Module):
    """Lớp bao bọc (Wrapper) cho mô hình Mask R-CNN CarDD."""

    def __init__(self, config: MaskRCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskRCNNConfig()
        self.class_names = DAMAGE_CLASSES[: self.config.num_classes]
        self.model = build_mask_rcnn_model(self.config)

    def forward(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Lan truyền thuận (Forward pass) cho Mask R-CNN."""
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                images = [img for img in images]
            elif images.dim() == 3:
                images = [images]

        return self.model(images, targets)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Đóng băng / Mở băng các tham số của backbone."""
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze
        logger.info("Backbone frozen: %s", freeze)

    def get_parameter_groups(self, lr: float = 1e-3, backbone_lr_ratio: float = 0.1) -> list[dict[str, Any]]:
        """Phân nhóm tham số học với learning rate phân biệt cho backbone vs heads."""
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {"params": backbone_params, "lr": lr * backbone_lr_ratio},
            {"params": head_params, "lr": lr},
        ]

    def count_parameters(self) -> dict[str, int]:
        """Đếm số lượng tham số tổng và tham số huấn luyện được."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
        }

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor | np.ndarray,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Dự đoán instance segmentation trên một ảnh duy nhất.

        Args:
            image: Tensor [3, H, W] hoặc numpy array [H, W, 3].
            score_threshold: Ngưỡng tự tin tối thiểu để giữ lại dự đoán.
            mask_threshold: Ngưỡng xác suất để nhị phân hóa mask pixel (mặc định 0.5).

        Returns:
            Danh sách dict gồm 'bbox', 'label', 'class_name', 'score', và 'mask'.
        """
        self.eval()

        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = image

        device = next(self.parameters()).device
        tensor = tensor.to(device)

        predictions = self.model([tensor])
        pred = predictions[0]

        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        masks = pred["masks"].cpu().numpy() if "masks" in pred else None

        results = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score >= score_threshold:
                cname = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
                item: dict[str, Any] = {
                    "bbox": [float(round(coord, 2)) for coord in box],
                    "label": int(label),
                    "class_name": cname,
                    "score": float(round(score, 4)),
                }
                if masks is not None:
                    item["mask"] = masks[i, 0] > mask_threshold
                results.append(item)

        return results

