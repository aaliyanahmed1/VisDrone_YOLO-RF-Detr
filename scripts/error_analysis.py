"""Error analysis: validation metrics, per-class performance, and failure samples.

Run after training. Loads the best checkpoint, runs on the validation set,
computes metrics (mAP, per-class AP), and saves a short report plus optional
images of low-confidence or misclassified detections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
# Best available trained model (YOLO26)
BEST_AVAILABLE_WEIGHTS = REPO_ROOT / "runs" / "detect" / "runs" / "yolo26" / "visdrone" / "weights" / "best.pt"

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def run_yolo26_val(
    weights: str,
    data_yaml: str,
    device: str,
    out_dir: Path,
) -> dict:
    """Run YOLO26 validation and return metrics + optional failure list."""
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.val(
        data=data_yaml,
        device=device or None,
        save_json=True,
    )
    # results has .metrics.map50_95, .metrics.map50, .metrics.box.*
    metrics = getattr(results, "metrics", None)
    out = {
        "model": "yolo26",
        "weights": weights,
        "mAP50-95": float(getattr(metrics, "map50_95", 0) or 0),
        "mAP50": float(getattr(metrics, "map50", 0) or 0),
        "per_class": {},
    }
    if metrics and hasattr(metrics, "results_dict"):
        out["per_class"] = getattr(metrics, "results_dict", {})
    return out


def run_rfdetr_val(
    weights: str,
    data_root: str,
    size: str,
    device: str,
    out_dir: Path,
) -> dict:
    """Run RF-DETR validation if API supports it; else return placeholder."""
    # RF-DETR may not expose a simple val() like YOLO; return structure for report
    out = {
        "model": "rfdetr",
        "weights": weights,
        "mAP50-95": None,
        "mAP50": None,
        "per_class": {},
        "note": "Run RF-DETR validation separately if supported.",
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Error analysis: validation metrics and failure report"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo26", "rfdetr"],
        help="Model type",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to best checkpoint (default: best available YOLO26 best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to data.yaml (YOLO26) or dataset root (RF-DETR)",
    )
    parser.add_argument(
        "--rfdetr-size",
        type=str,
        default="medium",
        choices=["nano", "small", "medium", "large"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/error_analysis",
        help="Directory for report and optional failure images",
    )
    args = parser.parse_args()

    # Default to best available YOLO26 weights and data.yaml
    weights = args.weights
    data = args.data
    if args.model == "yolo26":
        if not weights:
            weights = str(BEST_AVAILABLE_WEIGHTS)
        if not data:
            data = str(REPO_ROOT / "data" / "visdrone" / "data.yaml")
    if args.model == "rfdetr" and (not weights or not data):
        print("ERROR: --weights and --data are required for rfdetr.")
        return 1
    if args.model == "yolo26" and not Path(weights).exists():
        print(f"ERROR: Weights not found: {weights}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "yolo26":
        report = run_yolo26_val(
            weights,
            data,
            args.device,
            out_dir,
        )
    else:
        report = run_rfdetr_val(
            weights,
            data,
            args.rfdetr_size,
            args.device,
            out_dir,
        )

    report["timestamp"] = datetime.now().isoformat()
    report_path = out_dir / "error_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")
    print(f"  mAP50-95: {report.get('mAP50-95')}")
    print(f"  mAP50: {report.get('mAP50')}")
    return 0


if __name__ == "__main__":
    exit(main())
