import tempfile
from pathlib import Path
import pytest
from src.tracking.mlflow_tracker import MLflowConfig, MLflowTracker


def test_mlflow_tracker_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MLflowConfig(
            tracking_uri=f"file:///{tmp_dir}",
            experiment_name="test-experiment",
        )
        tracker = MLflowTracker(config)

        # Test context manager run
        with tracker.run(run_name="test_run_1", tags={"type": "unit_test"}):
            tracker.log_param("backbone", "resnet50_fpn")
            tracker.log_param("num_classes", 7)
            tracker.log_metric("loss", 0.42, step=1)
            tracker.log_metrics({"acc": 0.95, "mAP": 0.65}, step=1)

        # Verify run exists in dataframe
        df = tracker.get_runs_dataframe()
        assert not df.empty
        assert len(df) == 1
        assert df["tags.mlflow.runName"].iloc[0] == "test_run_1"
        assert df["params.backbone"].iloc[0] == "resnet50_fpn"
        assert float(df["metrics.loss"].iloc[0]) == pytest.approx(0.42)
