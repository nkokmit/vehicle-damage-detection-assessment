"""MLflow tracking abstraction placeholder."""

from dataclasses import dataclass


@dataclass(slots=True)
class MLflowConfig:
    """MLflow configuration settings."""

    tracking_uri: str = "mlruns"
    experiment_name: str = "vehicle-damage-detection"


class MLflowTracker:
    """Lightweight tracker abstraction."""

    def __init__(self, config: MLflowConfig | None = None) -> None:
        self.config = config or MLflowConfig()

    def log_params(self, params: dict[str, object]) -> dict[str, object]:
        """Return params as a no-op log action."""

        return dict(params)
