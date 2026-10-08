"""Visualization utilities for detection evaluation (Confusion Matrix, PR Curves)."""

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    normalize: bool = False,
    title: str = "Detection Confusion Matrix",
) -> Path | None:
    """Plot and optionally save the detection confusion matrix heatmap."""
    matrix = np.asarray(matrix, dtype=np.float32)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(row_sums, 1e-8), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")

    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = matrix[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix plot to: %s", out)
        return out

    plt.close(fig)
    return None


def save_report_stub(output_dir: str | Path) -> Path:
    """Create output directory and return report path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / "report.txt"
