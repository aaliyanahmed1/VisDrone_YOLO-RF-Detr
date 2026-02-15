"""Run tests for both models and write results to a standard output layout.

Standard layout:
  <output-dir>/
    yolo26/
      metrics.json
      plots/          (F1_curve.png, PR_curve.png, confusion_matrix, etc.)
      annotated/      (test images with bounding boxes)
    rfdetr/
      metrics.json
      plots/          (simple metrics plot if generated)
      annotated/      (test images with bounding boxes)

Usage:
  python scripts/run_test_suite.py --data <path/to/data.yaml> --data-coco <path/to/coco> --output-dir test_results
  python scripts/run_test_suite.py --data data/visdrone/data.yaml --data-coco data/visdrone_coco --output-dir test_results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO26 and RF-DETR tests; save annotated images and metrics/plots in standard layout")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml (test set must be defined)")
    parser.add_argument("--data-coco", type=str, required=True, help="Path to COCO-format dataset root (for RF-DETR)")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Base output directory (default: test_results)")
    parser.add_argument("--weights-yolo", type=str, default="", help="YOLO26 weights (default: auto-detect)")
    parser.add_argument("--checkpoint-rfdetr", type=str, default="", help="RF-DETR checkpoint (default: auto-detect)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for RF-DETR")
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = (REPO_ROOT / out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    yolo_out = out_base / "yolo26"
    rfdetr_out = out_base / "rfdetr"

    # --- YOLO26 ---
    cmd_yolo = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "test_yolo26.py"),
        "--data", args.data,
        "--output-dir", str(yolo_out),
        "--save-predictions",
    ]
    if args.weights_yolo:
        cmd_yolo.extend(["--weights", args.weights_yolo])
    print("Running YOLO26 test...")
    r = subprocess.run(cmd_yolo, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print("YOLO26 test failed.")
        return r.returncode

    # --- RF-DETR ---
    cmd_rf = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "test_rfdetr.py"),
        "--dataset-dir", args.data_coco,
        "--output-dir", str(rfdetr_out),
        "--save-annotated",
        "--device", args.device,
    ]
    if args.checkpoint_rfdetr:
        cmd_rf.extend(["--checkpoint", args.checkpoint_rfdetr])
    print("\nRunning RF-DETR test...")
    r = subprocess.run(cmd_rf, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print("RF-DETR test failed.")
        return r.returncode

    # Optional: simple metrics plot for RF-DETR
    rf_metrics = rfdetr_out / "metrics.json"
    if rf_metrics.exists():
        try:
            with open(rf_metrics, encoding="utf-8") as f:
                m = json.load(f)
            plots_dir = rfdetr_out / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            coco = m.get("coco_eval") or {}
            if coco:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                keys = ["AP", "AP50", "AP75"]
                vals = [coco.get(k) for k in keys]
                vals = [float(v) if v is not None else 0 for v in vals]
                ax.bar(keys, vals, color=["#2E86AB", "#A23B72", "#F18F01"])
                ax.set_ylabel("Score")
                ax.set_title("RF-DETR test metrics (COCO-style)")
                ax.set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(plots_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"RF-DETR plot: {plots_dir / 'metrics_summary.png'}")
        except Exception as e:
            print(f"Could not generate RF-DETR plot: {e}")

    print(f"\nTest results written to: {out_base}")
    print(f"  YOLO26:   {yolo_out} (metrics.json, plots/, annotated/)")
    print(f"  RF-DETR: {rfdetr_out} (metrics.json, plots/, annotated/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
