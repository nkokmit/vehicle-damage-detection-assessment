"""Prediction utilities."""

from src.models.model_factory import create_model


class Predictor:
    """Minimal predictor wrapper."""

    def __init__(self, model_name: str = "faster_rcnn") -> None:
        self.model = create_model(model_name=model_name)

    def predict(self, image: object) -> list[dict[str, float]]:
        """Run model prediction."""

        return self.model.predict(image)
