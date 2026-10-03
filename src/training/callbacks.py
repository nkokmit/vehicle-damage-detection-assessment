"""Training callback placeholders."""

from typing import Protocol


class Callback(Protocol):
    """Protocol for callback classes."""

    def on_epoch_end(self, epoch: int) -> None:
        """Called when an epoch ends."""
