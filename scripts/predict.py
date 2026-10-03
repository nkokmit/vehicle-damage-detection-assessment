"""CLI entrypoint for inference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor


def main() -> None:
    """Run prediction placeholder."""

    predictor = Predictor()
    print(f"predict: {predictor.predict(image=None)}")


if __name__ == "__main__":
    main()
