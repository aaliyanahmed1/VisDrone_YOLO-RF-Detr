"""Export YOLO26 best.pt to ONNX for inference.

Run once to generate models/Yolo26/weights/best.onnx. The inference API
will use the ONNX model when available.

Usage:
  python scripts/export_yolo26_onnx.py
  python scripts/export_yolo26_onnx.py --weights models/Yolo26/weights/best.pt --output models/Yolo26/weights/best.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO26 to ONNX")
    parser.add_argument("--weights", type=str, default="", help="Path to best.pt")
    parser.add_argument("--output", type=str, default="", help="Output .onnx path (default: same dir as weights)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size for export")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else REPO_ROOT / "models" / "Yolo26" / "weights" / "best.pt"
    if not weights.is_absolute():
        weights = (REPO_ROOT / weights).resolve()
    if not weights.exists():
        alt = REPO_ROOT / "runs" / "detect" / "runs" / "yolo26" / "visdrone" / "weights" / "best.pt"
        if alt.exists():
            weights = alt
        else:
            print(f"ERROR: Weights not found: {weights}")
            return 1

    out_path = Path(args.output) if args.output else weights.with_suffix(".onnx")
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {weights.name} -> ONNX")
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        # Export writes .onnx next to the weight file
        model.export(format="onnx", imgsz=args.imgsz, half=False, simplify=True)
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Export writes to same dir as weights with .onnx suffix
    onnx_path = weights.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"Done: {onnx_path}")
    else:
        print("Export finished; check model directory for .onnx file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
