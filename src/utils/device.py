"""Device selection helpers."""


def get_device(preferred: str = "auto") -> str:
    """Return selected compute device name."""

    return "cpu" if preferred == "auto" else preferred
