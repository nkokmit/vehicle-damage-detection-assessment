"""Loss function placeholders."""


def total_loss(losses: list[float]) -> float:
    """Compute the sum of scalar losses."""

    return float(sum(losses))
