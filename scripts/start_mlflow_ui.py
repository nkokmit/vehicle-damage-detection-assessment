"""Convenience script to launch MLflow UI with FileStore compatibility enabled."""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Start MLflow UI with FileStore enabled")
    parser.add_argument("--port", type=int, default=5000, help="Port to run MLflow UI on (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--backend-store-uri", type=str, default="./mlruns", help="Backend store URI (default: ./mlruns)")
    args = parser.parse_args()

    # Bật cờ cho phép FileStore trong MLflow 3.x+
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--backend-store-uri",
        args.backend_store_uri,
    ]

    print("=" * 65)
    print(f"Starting MLflow UI on: http://{args.host}:{args.port}")
    print(f"Backend store URI     : {args.backend_store_uri}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 65)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped.")


if __name__ == "__main__":
    main()
