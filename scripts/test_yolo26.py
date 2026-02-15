"""Test YOLO26 on VisDrone test set: metrics, plots, optional predictions.

Usage:
  python scripts/test_yolo26.py
  python scripts/test_yolo26.py --weights models/Yolo26/weights/best.pt --output-dir models/Yolo26/test_results --save-predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_weights():
    for p in [
        REPO_ROOT / "models" / "Yolo26" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "detect" / "runs" / "yolo26" / "visdrone" / "weights" / "best.pt",
    ]:
        if p.exists():
            return p
    return REPO_ROOT / "models" / "Yolo26" / "weights" / "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test YOLO26 on test set; save metrics and plots")
    parser.add_argument("--weights", type=str, default="", help="Path to best.pt (default: models/Yolo26/weights/best.pt or runs/...)")
    parser.add_argument("--data", type=str, default="", help="Path to data.yaml (default: data/visdrone/data.yaml)")
    parser.add_argument("--output-dir", type=str, default="", help="Output dir for metrics and plots")
    parser.add_argument("--save-predictions", action="store_true", help="Save test images with boxes")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else _default_weights()
    if not weights_path.is_absolute():
        weights_path = (REPO_ROOT / weights_path).resolve()
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        return 1

    data_path = Path(args.data) if args.data else (REPO_ROOT / "data" / "visdrone" / "data.yaml")
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        print(f"ERROR: Data yaml not found: {data_path}")
        return 1

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "test_results" / "yolo26")
    if args.output_dir and not Path(args.output_dir).is_absolute():
        out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = data_path.parent
    if not str(data_root).endswith("visdrone"):
        data_root = data_path.parent.parent
    with open(data_path, encoding="utf-8") as f:
        content = f.read()
    if "val:" in content and "test:" in content:
        lines = content.split("\n")
        new_lines = [line if not line.strip().startswith("val:") else "val: test/images" for line in lines]
        eval_yaml_content = "\n".join(new_lines)
    else:
        eval_yaml_content = content
    eval_yaml_path = out_dir / "data_test_eval.yaml"
    eval_yaml_path.write_text(eval_yaml_content, encoding="utf-8")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    print(f"Testing YOLO26 on test set: {weights_path.name}")
    print(f"Output: {out_dir}\n")

    metrics = model.val(
        data=str(eval_yaml_path),
        imgsz=args.imgsz,
        batch=args.batch,
        save_json=True,
        plots=True,
        save_dir=str(plots_dir),
    )

    if hasattr(metrics, "box"):
        box = metrics.box
        metrics_dict = {
            "mAP50": float(getattr(box, "map50", 0) or 0),
            "mAP50-95": float(getattr(box, "map", 0) or 0),
            "mAP75": float(getattr(box, "map75", 0) or 0),
            "precision": float(getattr(box, "mp", 0) or 0),
            "recall": float(getattr(box, "mr", 0) or 0),
            "class_names": getattr(metrics, "names", {}),
        }
        if hasattr(box, "maps") and box.maps is not None:
            metrics_dict["mAP50_per_class"] = [float(x) for x in box.maps]
        if hasattr(box, "ap50") and box.ap50 is not None:
            metrics_dict["AP50_per_class"] = [float(x) for x in box.ap50]
        if hasattr(box, "ap") and box.ap is not None:
            metrics_dict["AP50-95_per_class"] = [float(x) for x in box.ap]
    else:
        metrics_dict = {"raw": str(metrics)}

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics: {metrics_file}")
    print(f"Plots: {plots_dir}")

    if args.save_predictions and hasattr(metrics, "data"):
        test_images = Path(metrics.data.get("path", data_root)) / "test" / "images"
        if not test_images.exists():
            test_images = data_root / "test" / "images"
        if test_images.exists():
            pred_dir = out_dir / "annotated"
            pred_dir.mkdir(parents=True, exist_ok=True)
            model.predict(source=str(test_images), save=True, project=str(out_dir), name="annotated", imgsz=args.imgsz)
            print(f"Annotated images: {pred_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
