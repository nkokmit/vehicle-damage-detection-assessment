"""FastAPI application entrypoint."""

from fastapi import FastAPI

from api.schemas import PredictRequest, PredictResponse
from api.services.detection_service import DetectionService

app = FastAPI(title="Vehicle Damage Detection API")
service = DetectionService()


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Inference endpoint."""

    predictions = service.predict(payload.image_path)
    return PredictResponse(predictions=predictions)
