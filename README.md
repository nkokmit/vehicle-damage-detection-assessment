# vehicle-damage-detection-assessment

Python/MLOps scaffold for a vehicle damage detection system.

## Project structure

- `configs/`: data, model, training, and MLflow configuration.
- `data/`: raw data, processed splits, and annotation files.
- `src/`: core source modules (data, models, training, evaluation, inference, tracking, utils).
- `scripts/`: runnable CLI entry scripts for preparation, training, evaluation, prediction, export.
- `api/`: API layer (FastAPI app, schemas, and detection service).
- `tests/`: minimal tests for import/smoke checks.
- `notebooks/`: analysis notebook placeholders.
- `artifacts/`: generated checkpoints, predictions, reports.
- `mlruns/`: MLflow local tracking artifacts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

mflow:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

train:
.\env\Scripts\python.exe scripts/train.py `
  --config configs/train_res800.yaml `
  --resume artifacts/checkpoints/faster_rcnn_resnet50_fpn_amp_accum_b2_ep40/best_map_model.pth `
  --reset-lr `
  --lr 0.0003 `
  --scheduler-type cosine `
  --epochs 20

.\env\Scripts\python.exe scripts/train.py `
  --config configs/train_maskrcnn.yaml `
  --resume artifacts/checkpoints/mask_rcnn_resnet50_fpn_v2_amp_accum_b2_ep35/latest_model.pth `
  --run-name "mask_rcnn_resnet50_fpn_v2_amp_accum_b2_ep35" `
  --epochs 35