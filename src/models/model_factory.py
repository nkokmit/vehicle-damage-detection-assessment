"""Model factory utilities."""

from src.models.faster_rcnn import FasterRCNNConfig, FasterRCNNModel


def create_model(model_name: str = "faster_rcnn", num_classes: int = 2) -> FasterRCNNModel:
    """Create a model instance by name."""

    if model_name != "faster_rcnn":
        raise ValueError(f"Unsupported model: {model_name}")
    return FasterRCNNModel(FasterRCNNConfig(num_classes=num_classes))
