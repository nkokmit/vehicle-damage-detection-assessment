"""API data models."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input schema for prediction."""

    image_path: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Output schema for prediction."""

    predictions: list[dict[str, float]]
