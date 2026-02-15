"""Prepare VisDrone2019-DET-test-dev, run both models, and save metrics and annotated images.

Expects VisDrone2019-DET-test-dev under a directory (e.g. repo root or a folder you pass).
Creates YOLO and COCO versions of the test set, runs YOLO26 and RF-DETR tests, and writes
results in a standard layout.

Standard output layout:
  <output-dir>/
    yolo_data/           (YOLO format: test/images, test/labels, data.yaml)
    coco_data/           (COCO format: test/)
    results/
      yolo26/             (metrics.json, plots/, annotated/)
      rfdetr/             (metrics.json, plots/, annotated/)

Usage:
  # Full run (convert + test):
  python scripts/run_test_on_testdev.py --testdev-dir . --output-dir test_results_testdev

  # Test only (dataset already converted):
  python scripts/run_test_on_testdev.py --test-only --output-dir test_results_testdev
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET-test-dev, run both models, save metrics and annotated images"
    )
    parser.add_argument(
        "--testdev-dir",
        type=str,
        default=".",
        help="Directory containing VisDrone2019-DET-test-dev (e.g. repo root or data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results_testdev",
        help="Base directory for converted data and results",
    )
    parser.add_argument(
        "--weights-yolo",
        type=str,
        default="",
        help="YOLO26 weights path (default: auto-detect)",
    )
    parser.add_argument(
        "--checkpoint-rfdetr",
        type=str,
        default="",
        help="RF-DETR checkpoint path (default: auto-detect)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip conversion; run only YOLO26 and RF-DETR tests (requires existing yolo_data/ and coco_data/ under output-dir)",
    )
    parser.add_argument(
        "--rfdetr-only",
        action="store_true",
        help="With --test-only: run only RF-DETR test (skip YOLO26). Use when YOLO26 results are already saved.",
    )
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = (REPO_ROOT / out_base).resolve()
    yolo_root = out_base / "yolo_data"
    coco_root = out_base / "coco_data"
    results_base = out_base / "results"
    data_yaml = yolo_root / "data.yaml"

    if args.test_only:
        # Require existing converted data
        if not data_yaml.exists():
            print(f"ERROR: --test-only used but data not found. Expected: {data_yaml}")
            print("  Run without --test-only once to convert the dataset, or pass an output-dir that already has yolo_data/data.yaml and coco_data/.")
            return 1
        test_images = yolo_root / "test" / "images"
        if not test_images.exists() or not list(test_images.glob("*")):
            print(f"ERROR: No test images in {test_images}")
            return 1
        coco_test = coco_root / "test"
        if not coco_test.exists():
            print(f"ERROR: COCO test dir not found: {coco_test}")
            return 1
        print("Using existing converted data (--test-only).")
        print(f"  YOLO: {yolo_root}")
        print(f"  COCO: {coco_root}\n")
    else:
        testdev_dir = Path(args.testdev_dir)
        if not testdev_dir.is_absolute():
            testdev_dir = (REPO_ROOT / testdev_dir).resolve()
        testdev_folder = testdev_dir / "VisDrone2019-DET-test-dev"
        if not testdev_folder.exists():
            testdev_folder = testdev_dir / "test-dev"
        if not testdev_folder.exists() and testdev_dir.name.lower() in ("visdrone2019-det-test-dev", "test-dev"):
            testdev_folder = testdev_dir
            testdev_dir = testdev_dir.parent
        if not testdev_folder.exists():
            print(f"ERROR: VisDrone2019-DET-test-dev not found under {testdev_dir}")
            print("  Expected: <testdev-dir>/VisDrone2019-DET-test-dev/images and .../annotations")
            return 1

        out_base.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Convert VisDrone test-dev to YOLO (test split only) ---
        print("Step 1: Converting VisDrone2019-DET-test-dev to YOLO format...")
        r = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert_visdrone_to_yolo.py"),
                "--input-dir", str(testdev_dir),
                "--output-dir", str(yolo_root),
                "--splits", "test",
            ],
            cwd=str(REPO_ROOT),
        )
        if r.returncode != 0:
            print("Conversion to YOLO failed.")
            return r.returncode

        test_images = yolo_root / "test" / "images"
        if not test_images.exists() or not list(test_images.glob("*")):
            print("ERROR: No test images found after conversion.")
            return 1

        # --- Step 2: Write data.yaml for YOLO ---
        path_val = str(yolo_root.resolve()).replace("\\", "/")
        yaml_content = f"""# VisDrone test-dev (test split only)
path: {path_val}
train: test/images
val: test/images
test: test/images

nc: {len(VISDRONE_NAMES)}
names:
"""
        for i, name in enumerate(VISDRONE_NAMES):
            yaml_content += f"  {i}: {name}\n"
        data_yaml.write_text(yaml_content, encoding="utf-8")
        print(f"  Wrote {data_yaml}")

        # --- Step 3: Convert YOLO to COCO for RF-DETR ---
        print("\nStep 2: Converting YOLO to COCO format...")
        r = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert_yolo_to_coco.py"),
                "--input", str(yolo_root),
                "--output", str(coco_root),
            ],
            cwd=str(REPO_ROOT),
        )
        if r.returncode != 0:
            print("Conversion to COCO failed.")
            return r.returncode

        coco_test = coco_root / "test"
        if coco_test.exists() and not (coco_test / "images").exists():
            images_dir = coco_test / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for f in coco_test.glob("*.jpg"):
                f.rename(images_dir / f.name)
            for f in coco_test.glob("*.png"):
                f.rename(images_dir / f.name)
            for f in coco_test.glob("*.jpeg"):
                f.rename(images_dir / f.name)

    # --- Run YOLO26 test (skip if --rfdetr-only) ---
    if not args.rfdetr_only:
        print("\nStep 3: Running YOLO26 test...")
        cmd_yolo = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_yolo26.py"),
            "--data", str(data_yaml),
            "--output-dir", str(results_base / "yolo26"),
            "--save-predictions",
        ]
        if args.weights_yolo:
            cmd_yolo.extend(["--weights", args.weights_yolo])
        r = subprocess.run(cmd_yolo, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print("YOLO26 test failed.")
            return r.returncode
    else:
        print("\nSkipping YOLO26 (--rfdetr-only).")

    # --- Run RF-DETR test (optional if rfdetr not installable, e.g. Python 3.8) ---
    print("\nStep 4: Running RF-DETR test...")
    cmd_rf = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "test_rfdetr.py"),
        "--dataset-dir", str(coco_root),
        "--output-dir", str(results_base / "rfdetr"),
        "--save-annotated",
    ]
    if args.checkpoint_rfdetr:
        cmd_rf.extend(["--checkpoint", args.checkpoint_rfdetr])
    r = subprocess.run(cmd_rf, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print("RF-DETR test skipped or failed (YOLO26 results are saved).")
        print("  To run RF-DETR: use Python 3.10+ and install from GitHub:")
        print("    pip install \"git+https://github.com/roboflow/rf-detr.git\"")

    # --- Optional: RF-DETR metrics plot ---
    rf_metrics = results_base / "rfdetr" / "metrics.json"
    if rf_metrics.exists():
        try:
            import json
            with open(rf_metrics, encoding="utf-8") as f:
                m = json.load(f)
            plots_dir = results_base / "rfdetr" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            coco = m.get("coco_eval") or {}
            if coco:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                keys = ["AP", "AP50", "AP75"]
                vals = [float(coco.get(k)) if coco.get(k) is not None else 0 for k in keys]
                ax.bar(keys, vals, color=["#2E86AB", "#A23B72", "#F18F01"])
                ax.set_ylabel("Score")
                ax.set_title("RF-DETR test metrics (VisDrone2019-DET-test-dev)")
                ax.set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(plots_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"  (Optional RF-DETR plot skipped: {e})")

    print(f"\nDone. Results under: {out_base}")
    print(f"  YOLO26:   {results_base / 'yolo26'} (metrics.json, plots/, annotated/)")
    if (results_base / "rfdetr" / "metrics.json").exists():
        print(f"  RF-DETR: {results_base / 'rfdetr'} (metrics.json, plots/, annotated/)")
    else:
        print(f"  RF-DETR: skipped (see message above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
