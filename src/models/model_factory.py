"""Model factory utilities for vehicle damage detection."""

from typing import Any

from src.models.faster_rcnn import (
    DAMAGE_CLASSES,
    FasterRCNNConfig,
    FasterRCNNModel,
    build_faster_rcnn_model,
)


def create_model(
    model_name: str = "faster_rcnn",
    backbone: str = "resnet50_fpn",
    num_classes: int = 7,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    **kwargs: Any,
) -> FasterRCNNModel:
    """Create a damage detection model instance by name and backbone.

    Args:
        model_name: Name of model architecture ('faster_rcnn').
        backbone: Backbone architecture ('resnet50_fpn' for V1, 'resnet50_fpn_v2' for V2).
        num_classes: Number of output classes (default 7: 1 background + 6 damages).
        pretrained: Whether to load COCO pretrained weights for transfer learning.
        trainable_backbone_layers: Number of trainable backbone layers (0 to 5).
        **kwargs: Additional FasterRCNNConfig keyword arguments.

    Returns:
        Configured FasterRCNNModel instance.
    """
    if model_name != "faster_rcnn":
        raise ValueError(f"Unsupported model: '{model_name}'. Only 'faster_rcnn' is currently supported.")

    config = FasterRCNNConfig(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
    return FasterRCNNModel(config=config)


__all__ = [
    "create_model",
    "FasterRCNNModel",
    "FasterRCNNConfig",
    "build_faster_rcnn_model",
    "DAMAGE_CLASSES",
]
