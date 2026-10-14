"""CLI entrypoint for exporting vehicle damage detection and segmentation models to ONNX.

Supports:
- YOLOv8-seg models (YOLOv8s-seg, YOLOv8m-seg)
- Automatic ONNX graph simplification (onnxslim)
- Verification via onnx.checker and ONNX Runtime test inference
- Metadata inspection (input/output node shapes and datatypes)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.0+ compatibility patch for Ultralytics metrics / exports
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Vehicle Damage Models to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.pt",
        help="Path to PyTorch model weights (.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=800, help="Image resolution (640 or 800)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export ('cpu' or '0')")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (12 recommended)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch axes")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="Disable onnxslim simplification")
    parser.set_defaults(simplify=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to copy the exported ONNX file to",
    )
    return parser.parse_args()


def export_yolo_to_onnx(
    weights_path: Path,
    imgsz: int = 800,
    device: str = "cpu",
    opset: int = 12,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """Export YOLO segmentation model to ONNX format."""
    from ultralytics import YOLO

    logger.info("Loading PyTorch model from: %s", weights_path)
    model = YOLO(str(weights_path))

    logger.info(
        "Starting ONNX export (imgsz=%d, opset=%d, dynamic=%s, simplify=%s, device=%s)...",
        imgsz,
        opset,
        dynamic,
        simplify,
        device,
    )
    start_time = time.perf_counter()

    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    elapsed = time.perf_counter() - start_time
    exported_path = Path(exported_path_str)
    logger.info("Export completed in %.2fs -> %s", elapsed, exported_path)
    return exported_path


def verify_onnx_model(onnx_path: Path, imgsz: int = 800) -> dict[str, Any]:
    """Verify exported ONNX model validity and run a test forward pass with ONNX Runtime."""
    import onnx
    import onnxruntime as ort

    logger.info("Verifying ONNX model structure with onnx.checker...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure is VALID!")

    # Khởi tạo ONNX Runtime InferenceSession
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()

    inputs_meta = {
        "name": input_info.name,
        "shape": input_info.shape,
        "type": input_info.type,
    }
    outputs_meta = [{"name": o.name, "shape": o.shape, "type": o.type} for o in output_info]

    logger.info("ONNX Input : name=%s, shape=%s, type=%s", inputs_meta["name"], inputs_meta["shape"], inputs_meta["type"])
    for idx, o in enumerate(outputs_meta):
        logger.info("ONNX Output #%d: name=%s, shape=%s, type=%s", idx, o["name"], o["shape"], o["type"])

    # Chạy thử 1 forward pass với dummy input
    dummy_input = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    start_time = time.perf_counter()
    preds = session.run(None, {input_info.name: dummy_input})
    infer_time_ms = (time.perf_counter() - start_time) * 1000.0

    logger.info("Test forward pass succeeded in %.2f ms!", infer_time_ms)
    for idx, pred in enumerate(preds):
        logger.info("Output #%d actual shape: %s", idx, pred.shape)

    return {
        "inputs": inputs_meta,
        "outputs": outputs_meta,
        "test_latency_ms": infer_time_ms,
        "providers": session.get_providers(),
    }


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists() and (PROJECT_ROOT / args.weights).exists():
        weights_path = PROJECT_ROOT / args.weights

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    logger.info("=" * 65)
    logger.info("         VEHICLE DAMAGE MODEL - ONNX EXPORT ENGINE")
    logger.info("=" * 65)
    logger.info("Weights file : %s (%.1f MB)", weights_path, weights_path.stat().st_size / (1024 * 1024))
    logger.info("Image size   : %dx%d", args.imgsz, args.imgsz)
    logger.info("Opset version: %d", args.opset)
    logger.info("Dynamic axes : %s", args.dynamic)
    logger.info("Simplify     : %s", args.simplify)
    logger.info("=" * 65)

    # 1. Thực hiện xuất mô hình sang ONNX
    onnx_file = export_yolo_to_onnx(
        weights_path=weights_path,
        imgsz=args.imgsz,
        device=args.device,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )

    # 2. Nếu chỉ định output-dir, sao chép file ONNX vào thư mục đó
    final_onnx_path = onnx_file
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / onnx_file.name
        shutil.copy2(onnx_file, dest_path)
        final_onnx_path = dest_path
        logger.info("Copied ONNX model to: %s", final_onnx_path)

    # 3. Kiểm định mô hình ONNX với onnx.checker và onnxruntime
    verification_results = verify_onnx_model(final_onnx_path, imgsz=args.imgsz)

    file_size_mb = final_onnx_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 65)
    logger.info("ONNX EXPORT & VERIFICATION COMPLETED SUCCESSFULLY!")
    logger.info("File path        : %s", final_onnx_path)
    logger.info("File size        : %.2f MB", file_size_mb)
    logger.info("Input Node       : %s -> %s", verification_results["inputs"]["name"], verification_results["inputs"]["shape"])
    logger.info("Output Nodes     : %d nodes", len(verification_results["outputs"]))
    logger.info("Execution Engine : %s", verification_results["providers"])
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
