"""Metric helper functions."""


def accuracy(tp: int, total: int) -> float:
    """Compute simple accuracy, safe for empty totals."""

    return float(tp / total) if total else 0.0
