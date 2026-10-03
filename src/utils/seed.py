"""Random seed utilities."""

import random


def set_seed(seed: int = 42) -> int:
    """Set Python random seed and return it."""

    random.seed(seed)
    return seed
