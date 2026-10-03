"""CLI entrypoint for model training."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import Trainer


def main() -> None:
    """Run model training placeholder."""

    result = Trainer().train()
    print(f"train: {result}")


if __name__ == "__main__":
    main()
