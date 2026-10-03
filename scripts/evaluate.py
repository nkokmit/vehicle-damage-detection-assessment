"""CLI entrypoint for model evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate


def main() -> None:
    """Run model evaluation placeholder."""

    print(f"evaluate: {evaluate()}")


if __name__ == "__main__":
    main()
