"""Detection service abstraction."""

from src.inference.predictor import Predictor


class DetectionService:
    """Service layer for model inference."""

    def __init__(self) -> None:
        self.predictor = Predictor()

    def predict(self, _image_path: str) -> list[dict[str, float]]:
        """Run prediction from image path placeholder."""

        return self.predictor.predict(image=None)
