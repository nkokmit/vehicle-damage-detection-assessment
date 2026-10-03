"""Transform helpers for preprocessing and augmentation."""

from typing import Any, Callable


def get_default_transform() -> Callable[[Any], Any]:
    """Return a no-op transform placeholder."""

    return lambda item: item
