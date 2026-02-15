"""Analyze VisDrone dataset: size, class distribution, label statistics."""

import argparse
from collections import Counter
from pathlib import Path

# VisDrone 10 classes
VISDRONE_CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze VisDrone dataset splits and class distribution"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/visdrone",
        help="Dataset root (train/val/test with images and labels)",
    )
    args = parser.parse_args()
    root = Path(args.data)

    for split in ("train", "valid", "test"):
        if split == "valid":
            labels_dir = root / "val" / "labels"
            if not labels_dir.exists():
                labels_dir = root / "valid" / "labels"
        else:
            labels_dir = root / split / "labels"
        if not labels_dir.exists():
            print(f"[{split}] No labels dir: {labels_dir}")
            continue
        n_images = 0
        n_boxes = 0
        class_counts = Counter()
        for label_file in labels_dir.glob("*.txt"):
            n_images += 1
            try:
                with open(label_file, encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            n_boxes += 1
                            class_counts[int(parts[0])] += 1
            except (ValueError, OSError) as e:
                print(f"[{split}] Error reading {label_file}: {e}")
        avg = n_boxes / max(1, n_images)
        print(f"\n{split}: images={n_images}, boxes={n_boxes}, avg boxes/img={avg:.2f}")
        print("  class distribution:")
        for cid in sorted(class_counts.keys()):
            name = VISDRONE_CLASS_NAMES[cid] if cid < len(VISDRONE_CLASS_NAMES) else str(cid)
            print(f"    {cid} {name}: {class_counts[cid]}")


if __name__ == "__main__":
    main()
