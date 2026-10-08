"""Faster R-CNN model definition for Vehicle Damage Detection."""

from dataclasses import dataclass
import logging
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logger = logging.getLogger(__name__)

# Danh sách 7 lớp (Lớp 0 là background theo quy chuẩn PyTorch Torchvision)
DAMAGE_CLASSES: list[str] = [
    "background",
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


@dataclass(slots=True)
class FasterRCNNConfig:
    """Configuration for Faster R-CNN model."""

    num_classes: int = 7
    backbone: str = "resnet50_fpn"  # 'resnet50_fpn' (v1) hoặc 'resnet50_fpn_v2' (v2)
    pretrained: bool = True
    trainable_backbone_layers: int = 3
    min_size: int = 512
    max_size: int = 512
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100


def build_faster_rcnn_model(
    config: FasterRCNNConfig | None = None,
) -> FasterRCNN:
    """Build a Faster R-CNN model with a COCO-pretrained backbone and custom 7-class head.

    Supports:
        - Model 1: 'resnet50_fpn' (Faster R-CNN ResNet-50 FPN V1)
        - Model 2: 'resnet50_fpn_v2' (Faster R-CNN ResNet-50 FPN V2 with improved FPN and SyncBN)

    Workflow:
        COCO pretrained model (ResNet-50-FPN v1 or v2)
                ↓
        Faster R-CNN Backbone + RPN
                ↓
        Extract in_features (1024) from ROI Heads
                ↓
        Replace FastRCNNPredictor classifier head
                ↓
        7 output classes (background + 6 damage classes)

    Args:
        config: FasterRCNNConfig instance.

    Returns:
        Configured FasterRCNN model.
    """
    cfg = config or FasterRCNNConfig()
    backbone_name = cfg.backbone.lower()

    if backbone_name in ("resnet50_fpn_v2", "resnet50_v2", "v2"):
        # Model 2: Faster R-CNN ResNet-50 FPN V2
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if cfg.pretrained else None
        logger.info(
            "Building Faster R-CNN ResNet-50 FPN V2 (pretrained=%s, weights=%s)",
            cfg.pretrained,
            weights,
        )
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=cfg.trainable_backbone_layers,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            box_score_thresh=cfg.box_score_thresh,
            box_nms_thresh=cfg.box_nms_thresh,
            box_detections_per_img=cfg.box_detections_per_img,
        )
    elif backbone_name in ("resnet50_fpn", "resnet50", "v1"):
        # Model 1: Faster R-CNN ResNet-50 FPN V1 (classic baseline)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if cfg.pretrained else None
        logger.info(
            "Building Faster R-CNN ResNet-50 FPN V1 (pretrained=%s, weights=%s)",
            cfg.pretrained,
            weights,
        )
        model = fasterrcnn_resnet50_fpn(
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

    # Lấy số chiều đặc trưng đầu vào của box predictor hiện tại (1024)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Thay thế classifier head bằng FastRCNNPredictor mới với 7 classes:
    # 0: background, 1: dent, 2: scratch, 3: crack, 4: glass shatter, 5: lamp broken, 6: tire flat
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features,
        num_classes=cfg.num_classes,
    )
    logger.info(
        "Replaced box_predictor [%s]: in_features=%d -> num_classes=%d",
        cfg.backbone,
        in_features,
        cfg.num_classes,
    )

    return model


class FasterRCNNModel(nn.Module):
    """Wrapper class for Faster R-CNN vehicle damage detection model."""

    def __init__(self, config: FasterRCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or FasterRCNNConfig()
        self.class_names = DAMAGE_CLASSES[: self.config.num_classes]
        self.model = build_faster_rcnn_model(self.config)

    def forward(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Forward pass for Faster R-CNN.

        Args:
            images: List of Tensor[C, H, W] or batch Tensor[B, C, H, W].
            targets: List of dicts (during training) containing 'boxes' and 'labels'.

        Returns:
            - During training (self.training=True, targets provided):
              Dict of losses: {
                  'loss_classifier': Tensor,
                  'loss_box_reg': Tensor,
                  'loss_objectness': Tensor,
                  'loss_rpn_box_reg': Tensor
              }
            - During evaluation (self.training=False):
              List of dicts: [
                  {'boxes': Tensor[N, 4], 'labels': Tensor[N], 'scores': Tensor[N]}, ...
              ]
        """
        # Đảm bảo images là dạng list of Tensors
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                images = [img for img in images]
            elif images.dim() == 3:
                images = [images]

        return self.model(images, targets)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the backbone parameters for transfer learning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze
        logger.info("Backbone frozen: %s", freeze)

    def get_parameter_groups(self, lr: float = 1e-3, backbone_lr_ratio: float = 0.1) -> list[dict[str, Any]]:
        """Get parameter groups with differential learning rates for backbone vs heads."""
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
        """Count total and trainable parameters in the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor | np.ndarray,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Perform high-level prediction on a single image.

        Args:
            image: Image Tensor [3, H, W] or numpy array [H, W, 3].
            score_threshold: Minimum confidence score to retain detection.

        Returns:
            List of detection dicts with 'bbox', 'label', 'score', and 'class_name'.
        """
        self.eval()

        if isinstance(image, np.ndarray):
            # [H, W, 3] uint8 -> [3, H, W] float
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

        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:
                cname = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
                results.append(
                    {
                        "bbox": [float(round(coord, 2)) for coord in box],
                        "label": int(label),
                        "class_name": cname,
                        "score": float(round(score, 4)),
                    }
                )

        return results
