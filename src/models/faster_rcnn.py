"""Model definition placeholder for Faster R-CNN."""

from dataclasses import dataclass


@dataclass(slots=True)
class FasterRCNNConfig:
    """Configuration for Faster R-CNN placeholder."""

    num_classes: int = 2


class FasterRCNNModel:
    """Minimal model placeholder class."""

    def __init__(self, config: FasterRCNNConfig | None = None) -> None:
        self.config = config or FasterRCNNConfig()

    def predict(self, _image: object) -> list[dict[str, float]]:
        """Return an empty prediction list."""

        return []
