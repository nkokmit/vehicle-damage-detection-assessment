"""MLflow tracking abstraction for vehicle damage detection experiments."""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import mlflow
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLflowConfig:
    """MLflow configuration settings."""

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "vehicle-damage-detection"


class MLflowTracker:
    """Production MLflow tracker supporting params, metrics, artifacts, and run comparisons."""

    def __init__(self, config: MLflowConfig | None = None) -> None:
        self.config = config or MLflowConfig()
        # Đặt tracking URI cục bộ hoặc server từ xa
        mlflow.set_tracking_uri(self.config.tracking_uri)
        # Tạo hoặc đặt experiment
        self.experiment = mlflow.set_experiment(self.config.experiment_name)
        self.active_run: mlflow.ActiveRun | None = None
        logger.info(
            "MLflow tracking initialized: URI=%s, Experiment=%s (ID=%s)",
            self.config.tracking_uri,
            self.config.experiment_name,
            self.experiment.experiment_id,
        )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
        run_id: str | None = None,
    ) -> mlflow.ActiveRun:
        """Start or resume an MLflow run."""
        run_tags = tags or {}
        if description:
            run_tags["mlflow.note.content"] = description

        self.active_run = mlflow.start_run(
            run_id=run_id,
            experiment_id=self.experiment.experiment_id if run_id is None else None,
            run_name=run_name,
            tags=run_tags,
        )
        logger.info("Started/Resumed MLflow run: %s (ID: %s)", run_name or run_id, self.active_run.info.run_id)
        return self.active_run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the currently active MLflow run."""
        if self.active_run is not None:
            mlflow.end_run(status=status)
            logger.info("Ended MLflow run: %s", self.active_run.info.run_id)
            self.active_run = None

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
        run_id: str | None = None,
    ) -> Iterator["MLflowTracker"]:
        """Context manager for an MLflow run."""
        self.start_run(run_name=run_name, tags=tags, description=description, run_id=run_id)
        try:
            yield self
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.debug("Parameter %s already exists: %s", key, e)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyperparameters safely (ignoring existing ones on resume)."""
        # Convert non-primitive values to strings for MLflow safety
        clean_params = {k: str(v) if isinstance(v, (list, tuple, dict)) else v for k, v in params.items()}
        try:
            mlflow.log_params(clean_params)
        except Exception as e:
            logger.debug("Parameters may already exist in resumed run: %s", e)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric value."""
        mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at a given step/epoch."""
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file or directory as an MLflow artifact."""
        path_obj = Path(local_path)
        if path_obj.exists():
            if path_obj.is_dir():
                mlflow.log_artifacts(str(path_obj), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(path_obj), artifact_path=artifact_path)
        else:
            logger.warning("Artifact not found to log: %s", local_path)

    def log_model(self, model: torch.nn.Module, artifact_path: str = "model") -> None:
        """Log PyTorch model to MLflow."""
        try:
            mlflow.pytorch.log_model(model, artifact_path=artifact_path)
        except Exception as e:
            logger.warning("Failed to log PyTorch model artifact: %s", e)

    def get_runs_dataframe(self) -> pd.DataFrame:
        """Retrieve all runs from the current experiment as a pandas DataFrame."""
        return mlflow.search_runs(experiment_ids=[self.experiment.experiment_id])

    def compare_runs(self, run_names: list[str] | None = None) -> pd.DataFrame:
        """Generate a comparison table across runs in the experiment.

        Args:
            run_names: Optional filter for specific run names.

        Returns:
            DataFrame comparing parameters and key metrics across runs.
        """
        df = self.get_runs_dataframe()
        if df.empty:
            return pd.DataFrame()

        if run_names:
            df = df[df["tags.mlflow.runName"].isin(run_names)]

        # Lọc các cột quan trọng
        cols = [c for c in df.columns if c.startswith("params.") or c.startswith("metrics.") or c in ["run_id", "tags.mlflow.runName", "status", "start_time"]]
        return df[cols]
