"""Dataset checking: structure, label format, class distribution, and basic stats.

Run this before training to validate your dataset and spot missing files,
invalid labels, or severe class imbalance. Works with YOLO-format datasets
(images in train/val/test, labels as .txt per image).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

# VisDrone class names (10 classes). Override with --classes if using another dataset.
VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def check_split(
    root: Path,
    split: str,
    class_names: list[str],
    require_images: bool,
) -> dict:
    """Check one split: image/label counts, missing pairs, invalid lines, class counts."""
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    # Some datasets use "val", others "valid"
    if not labels_dir.exists() and split == "val":
        labels_dir = root / "valid" / "labels"
        images_dir = root / "valid" / "images"

    report = {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "images_dir_exists": images_dir.exists(),
        "labels_dir_exists": labels_dir.exists(),
        "n_images": 0,
        "n_labels": 0,
        "n_paired": 0,
        "n_images_without_label": 0,
        "n_labels_without_image": 0,
        "invalid_label_lines": 0,
        "class_counts": Counter(),
        "errors": [],
    }

    if not images_dir.exists():
        report["errors"].append(f"Missing directory: {images_dir}")
        return report
    if not labels_dir.exists():
        report["errors"].append(f"Missing directory: {labels_dir}")
        return report

    image_stems = set()
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for p in images_dir.glob(ext):
            image_stems.add(p.stem)
    label_stems = set()
    for p in labels_dir.glob("*.txt"):
        label_stems.add(p.stem)

    report["n_images"] = len(image_stems)
    report["n_labels"] = len(label_stems)
    report["n_paired"] = len(image_stems & label_stems)
    report["n_images_without_label"] = len(image_stems - label_stems)
    report["n_labels_without_image"] = len(label_stems - image_stems)

    nc = len(class_names)
    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        if stem not in image_stems:
            continue
        try:
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        report["invalid_label_lines"] += 1
                        continue
                    cls_id = int(parts[0])
                    if 0 <= cls_id < nc:
                        report["class_counts"][cls_id] += 1
                    else:
                        report["invalid_label_lines"] += 1
        except (ValueError, OSError) as e:
            report["errors"].append(f"{label_path.name}: {e}")

    report["class_counts"] = dict(report["class_counts"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check YOLO-format dataset: structure, labels, class distribution"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/visdrone",
        help="Dataset root (containing train/val/test or train/valid/test)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names. Default: VisDrone 10 classes.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional: write JSON report to this path",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to check",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: Data root not found: {root}")
        return 1

    class_names = args.classes.split(",") if args.classes else VISDRONE_NAMES
    splits = [s.strip() for s in args.splits.split(",")]

    all_reports = []
    for split in splits:
        report = check_split(root, split, class_names, require_images=True)
        all_reports.append(report)

    # Print summary
    print("=" * 60)
    print("DATASET CHECK REPORT")
    print("=" * 60)
    print(f"Data root: {root}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print()
    for report in all_reports:
        print(f"--- {report['split'].upper()} ---")
        if report["errors"]:
            for e in report["errors"]:
                print(f"  ERROR: {e}")
        print(f"  images dir exists: {report['images_dir_exists']}")
        print(f"  labels dir exists: {report['labels_dir_exists']}")
        print(f"  n_images: {report['n_images']}, n_labels: {report['n_labels']}")
        print(f"  n_paired: {report['n_paired']}")
        print(f"  images without label: {report['n_images_without_label']}")
        print(f"  labels without image: {report['n_labels_without_image']}")
        print(f"  invalid label lines: {report['invalid_label_lines']}")
        if report["class_counts"]:
            print("  class distribution:")
            for cid in sorted(report["class_counts"].keys()):
                name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                print(f"    {name}: {report['class_counts'][cid]}")
        print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"splits": all_reports, "class_names": class_names}, f, indent=2)
        print(f"Report written to {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
