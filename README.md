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
