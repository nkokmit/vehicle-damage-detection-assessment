"""Post-processing helper utilities."""

from typing import Any


def filter_predictions(predictions: list[dict[str, Any]], score_threshold: float = 0.5) -> list[dict[str, Any]]:
    """Filter predictions by score threshold when score exists."""

    filtered: list[dict[str, Any]] = []
    for item in predictions:
        score = item.get("score", 1.0)
        if isinstance(score, (int, float)) and score >= score_threshold:
            filtered.append(item)
    return filtered
