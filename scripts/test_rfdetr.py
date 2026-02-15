"""Test RF-DETR checkpoint on VisDrone test set; print and save metrics.

Usage:
  python scripts/test_rfdetr.py
  python scripts/test_rfdetr.py --checkpoint models/rfdetr/weights/checkpoint_best_ema.pth --dataset-dir data/visdrone_coco --output-dir models/rfdetr/test_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_checkpoint():
    for d in [REPO_ROOT / "models" / "rfdetr" / "weights", REPO_ROOT / "runs" / "rfdetr" / "visdrone_20260214_145610"]:
        if not d.exists():
            continue
        for name in ("checkpoint_best_regular.pth", "checkpoint_best_ema.pth", "checkpoint_best_total.pth", "checkpoint_best.pth", "checkpoint_last.pth"):
            p = d / name
            if p.exists():
                return p
        for p in sorted(d.glob("checkpoint_*.pth"), reverse=True):
            return p
    return REPO_ROOT / "models" / "rfdetr" / "weights" / "checkpoint_best_regular.pth"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test RF-DETR on VisDrone test set")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to .pth checkpoint")
    parser.add_argument("--dataset-dir", type=str, default="", help="COCO-format dataset root (e.g. data/visdrone_coco)")
    parser.add_argument("--output-dir", type=str, default="", help="Output dir for metrics and annotated images")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated test images to output-dir/annotated/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _default_checkpoint()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (REPO_ROOT / "data" / "visdrone_coco")
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found: {dataset_dir}")
        return 1

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "test_results" / "rfdetr")
    if args.output_dir and not Path(args.output_dir).is_absolute():
        out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print("ERROR: rfdetr not installed")
        return 1

    print("Loading RF-DETR:", checkpoint_path.name)
    try:
        # Roboflow rfdetr: load weights via constructor (no load_checkpoint)
        model = RFDETRBase(pretrain_weights=str(checkpoint_path))
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return 1

    print("Running test evaluation...")
    try:
        results = model.evaluate(
            dataset_dir=str(dataset_dir),
            dataset_name="test",
            device=args.device,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    if "coco_eval" in results:
        m = results["coco_eval"]
        print("\n--- COCO metrics (test) ---")
        for k, v in [("AP .50:.95", m.get("AP")), ("AP50", m.get("AP50")), ("AP75", m.get("AP75"))]:
            print(f"  {k}: {v if v is None else f'{float(v):.3f}'}")
    if "loss" in results:
        print(f"  Test loss: {results['loss']:.4f}")

    # Save metrics
    to_save = {}
    if "coco_eval" in results:
        to_save["coco_eval"] = {k: float(v) if v is not None else None for k, v in results["coco_eval"].items()}
    if "loss" in results:
        to_save["loss"] = float(results["loss"])
    if to_save:
        metrics_file = out_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nMetrics saved: {metrics_file}")

    if args.save_annotated:
        test_images_dir = dataset_dir / "test" / "images"
        if not test_images_dir.exists():
            test_images_dir = dataset_dir / "test"
        if test_images_dir.exists():
            annotated_dir = out_dir / "annotated"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            import sys
            sys.path.insert(0, str(REPO_ROOT))
            from api.engine import get_annotated_image_rfdetr
            w = str(checkpoint_path.parent) if checkpoint_path.suffix == ".pth" else str(checkpoint_path)
            count = 0
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in test_images_dir.glob(ext):
                    out_path = annotated_dir / img_path.name
                    try:
                        get_annotated_image_rfdetr(str(img_path), str(out_path), w)
                        count += 1
                    except Exception as e:
                        print(f"  Skip {img_path.name}: {e}")
            print(f"Annotated images: {annotated_dir} ({count} files)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
