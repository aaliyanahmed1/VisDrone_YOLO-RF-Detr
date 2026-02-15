"""Inference pipeline: load model, preprocess image, run detection, return/output results.

Usage:
  python scripts/inference.py --model yolo26 --image path/to/image.jpg
  python scripts/inference.py --model yolo26 --weights runs/detect/runs/yolo26/visdrone/weights/best.pt --image path/to/image.jpg
  python scripts/inference.py --model rfdetr --weights runs/rfdetr/visdrone_20260214_145610 --image path/to/image.jpg [--out out.jpg]

Best available model (default for yolo26): runs/detect/runs/yolo26/visdrone/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Prefer models/ layout; fallback to runs/
BEST_AVAILABLE_WEIGHTS = REPO_ROOT / "models" / "Yolo26" / "weights" / "best.pt"
if not BEST_AVAILABLE_WEIGHTS.exists():
    BEST_AVAILABLE_WEIGHTS = REPO_ROOT / "runs" / "detect" / "runs" / "yolo26" / "visdrone" / "weights" / "best.pt"
RFDETR_RUN_DIR = REPO_ROOT / "models" / "rfdetr" / "weights"
if not RFDETR_RUN_DIR.exists() or not list(RFDETR_RUN_DIR.glob("checkpoint*.pth")):
    RFDETR_RUN_DIR = REPO_ROOT / "runs" / "rfdetr" / "visdrone_20260214_145610"

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def run_yolo26(image_path: str, weights_path: str, device: str = "") -> list[dict]:
    from ultralytics import YOLO
    model = YOLO(weights_path)
    kwargs = {"device": device} if device else {}
    results = model.predict(image_path, **kwargs)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            xyxy = r.boxes.xyxy[i].tolist()
            out.append({
                "class_id": cls_id,
                "class_name": VISDRONE_NAMES[cls_id] if cls_id < len(VISDRONE_NAMES) else str(cls_id),
                "confidence": round(conf, 4),
                "bbox": [round(x, 2) for x in xyxy],
            })
    return out


def _resolve_rfdetr_checkpoint(weights_path: str) -> str:
    """Resolve RF-DETR checkpoint: path to .pth file or dir. Prefer best_total, then best_ema, then best."""
    p = Path(weights_path)
    if p.suffix == ".pth" and p.exists():
        return str(p)
    # Directory: try preferred checkpoint names in order
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best.pth", "checkpoint_last.pth"):
        candidate = p / name
        if candidate.exists():
            return str(candidate)
    # Any checkpoint_*.pth
    for candidate in sorted(p.glob("checkpoint_*.pth"), reverse=True):
        return str(candidate)
    return str(p / "checkpoint_best_total.pth")  # let RF-DETR raise if missing


def run_rfdetr(image_path: str, weights_path: str, device: str = "") -> list[dict]:
    from rfdetr import RFDETRBase
    ckpt = _resolve_rfdetr_checkpoint(weights_path)
    model = RFDETRBase(pretrain_weights=ckpt)
    detections = model.predict(image_path)
    out = []
    if hasattr(detections, "__iter__"):
        for d in detections:
            if isinstance(d, dict):
                cls_id = d.get("category_id", d.get("class_id", 0)) - 1
                if cls_id < 0:
                    cls_id = 0
                out.append({
                    "class_id": cls_id,
                    "class_name": VISDRONE_NAMES[cls_id] if cls_id < len(VISDRONE_NAMES) else str(cls_id),
                    "confidence": round(float(d.get("score", d.get("confidence", 0))), 4),
                    "bbox": d.get("bbox", d.get("box", [])),
                })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run object detection on an image")
    parser.add_argument("--model", type=str, required=True, choices=["yolo26", "rfdetr"])
    parser.add_argument("--weights", type=str, default="", help="Path to weights (default: best available for yolo26)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default="", help="Optional: save visualization image")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    # Default to best available: YOLO26 best.pt or RF-DETR run dir
    weights = args.weights
    if not weights:
        if args.model == "yolo26":
            weights = str(BEST_AVAILABLE_WEIGHTS)
        else:
            weights = str(RFDETR_RUN_DIR)
    if args.model == "yolo26" and not Path(weights).exists():
        print(f"ERROR: Weights not found: {weights}")
        print("Train YOLO26 first or set --weights to your best.pt path.")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1

    if args.model == "yolo26":
        detections = run_yolo26(str(image_path), weights, args.device)
    else:
        detections = run_rfdetr(str(image_path), weights, args.device)

    print(json.dumps({"image": str(image_path), "detections": detections}, indent=2))
    if args.out:
        try:
            from ultralytics import YOLO
            model = YOLO(weights) if args.model == "yolo26" else None
            if model and args.model == "yolo26":
                model.predict(str(image_path), save=True, project=Path(args.out).parent, name=Path(args.out).stem)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    exit(main())
