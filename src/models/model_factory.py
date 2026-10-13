"""Model factory utilities for vehicle damage detection."""

from typing import Any

from src.models.faster_rcnn import (
    DAMAGE_CLASSES,
    FasterRCNNConfig,
    FasterRCNNModel,
    build_faster_rcnn_model,
)
from src.models.mask_rcnn import (
    MaskRCNNConfig,
    MaskRCNNModel,
    build_mask_rcnn_model,
)


def create_model(
    model_name: str = "faster_rcnn",
    backbone: str = "resnet50_fpn",
    num_classes: int = 7,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    **kwargs: Any,
) -> FasterRCNNModel | MaskRCNNModel:
    """Create a damage detection or instance segmentation model instance.

    Args:
        model_name: Name of model architecture ('faster_rcnn' or 'mask_rcnn').
        backbone: Backbone architecture ('resnet50_fpn' for V1, 'resnet50_fpn_v2' for V2).
        num_classes: Number of output classes (default 7: 1 background + 6 damages).
        pretrained: Whether to load COCO pretrained weights for transfer learning.
        trainable_backbone_layers: Number of trainable backbone layers (0 to 5).
        **kwargs: Additional model config keyword arguments.

    Returns:
        Configured FasterRCNNModel or MaskRCNNModel instance.
    """
    model_key = model_name.lower().replace("-", "_")

    if model_key == "faster_rcnn":
        config = FasterRCNNConfig(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
        return FasterRCNNModel(config=config)
    elif model_key in ("mask_rcnn", "maskrcnn"):
        config = MaskRCNNConfig(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
        return MaskRCNNModel(config=config)
    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Choose 'faster_rcnn' or 'mask_rcnn'."
        )


__all__ = [
    "create_model",
    "FasterRCNNModel",
    "FasterRCNNConfig",
    "build_faster_rcnn_model",
    "MaskRCNNModel",
    "MaskRCNNConfig",
    "build_mask_rcnn_model",
    "DAMAGE_CLASSES",
]
