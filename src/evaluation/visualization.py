"""Visualization placeholders for evaluation outputs."""

from pathlib import Path


def save_report_stub(output_dir: str | Path) -> Path:
    """Create output directory and return report path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / "report.txt"
