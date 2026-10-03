"""Confusion matrix helper placeholders."""


def empty_confusion_matrix(num_classes: int) -> list[list[int]]:
    """Create a square zero confusion matrix."""

    return [[0 for _ in range(num_classes)] for _ in range(num_classes)]
