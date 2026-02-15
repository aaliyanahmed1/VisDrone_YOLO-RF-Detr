"""Convert VisDrone2019-DET format to YOLO format for use with train_yolo26.py.

VisDrone annotation format (per line): 
  bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
  object_category: 0=ignored, 1=pedestrian..10=motor, 11=others. We use 1-10 -> YOLO 0-9.

YOLO format (per line): 
  class_id x_center y_center width height  (normalized 0-1, relative to image size)

Usage:
  python scripts/convert_visdrone_to_yolo.py \\
    --input-dir ~/Downloads \\
    --output-dir data/visdrone
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# VisDrone 10 classes (category 1-10 in source -> 0-9 in YOLO)
VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) of image."""
    if Image is not None:
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]
    # Fallback: assume 1920x1080 if PIL missing
    return 1920, 1080


def convert_annotation_line(
    line: str,
    img_w: int,
    img_h: int,
) -> str | None:
    """Convert one VisDrone line to YOLO line. Returns None if line should be skipped."""
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) < 6:
        return None
    try:
        x_min = int(parts[0])
        y_min = int(parts[1])
        w = int(parts[2])
        h = int(parts[3])
        cat = int(parts[5])
    except (ValueError, IndexError):
        return None
    # Use categories 1-10 only (0=ignored, 11=others -> skip)
    if cat < 1 or cat > 10:
        return None
    class_id = cat - 1  # 0-9 for YOLO
    # Convert to center and normalize
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    x_center_n = x_center / img_w
    y_center_n = y_center / img_h
    w_n = w / img_w
    h_n = h / img_h
    # Clamp to [0, 1]
    x_center_n = max(0, min(1, x_center_n))
    y_center_n = max(0, min(1, y_center_n))
    w_n = max(0, min(1, w_n))
    h_n = max(0, min(1, h_n))
    return f"{class_id} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}"


def convert_split(
    input_root: Path,
    output_root: Path,
    split_name: str,
    copy_images: bool = True,
    source_subdir: str | None = None,
) -> tuple[int, int]:
    """Convert one split (train/val/test). Returns (n_images, n_labels).
    source_subdir: if set, use this as the source folder name (e.g. test-dev -> write to output test/).
    """
    src_name = source_subdir or split_name
    # VisDrone folder names; support nested: VisDrone2019-DET-train/VisDrone2019-DET-train/images
    possible_dirs = [
        input_root / f"VisDrone2019-DET-{src_name}",
        input_root / src_name,
        (input_root / f"VisDrone2019-DET-{src_name}") / f"VisDrone2019-DET-{src_name}",
        (input_root / src_name) / src_name,
    ]
    images_src = ann_src = None
    for d in possible_dirs:
        if not d.exists():
            continue
        src_dir = d.resolve()
        for im_name, an_name in [("images", "annotations"), ("Images", "Annotations")]:
            im_dir = src_dir / im_name
            an_dir = src_dir / an_name
            if im_dir.exists() and an_dir.exists():
                images_src, ann_src = im_dir, an_dir
                break
        if images_src is not None:
            break
        if (next(src_dir.glob("*.jpg"), None) or next(src_dir.glob("*.png"), None)) and next(src_dir.glob("*.txt"), None):
            images_src = ann_src = src_dir
            break
    if images_src is None or ann_src is None:
        return 0, 0

    # Write to output split name (e.g. test) even when source is test-dev
    out_images = output_root / split_name / "images"
    out_labels = output_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    import shutil
    pairs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for img_path in images_src.glob(ext):
            ann_path = ann_src / f"{img_path.stem}.txt"
            if ann_path.exists():
                pairs.append((img_path, ann_path))
    total = len(pairs)
    if total == 0:
        return 0, 0

    n_images = 0
    n_labels = 0
    step = max(1, min(500, total // 10))
    for i, (img_path, ann_path) in enumerate(pairs):
        stem = img_path.stem
        img_w, img_h = get_image_size(img_path)
        yolo_lines = []
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                yolo_line = convert_annotation_line(line, img_w, img_h)
                if yolo_line:
                    yolo_lines.append(yolo_line)
        if not yolo_lines:
            continue
        label_path = out_labels / f"{stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        n_labels += 1
        dest_img = out_images / img_path.name
        if copy_images:
            shutil.copy2(img_path, dest_img)
        else:
            try:
                dest_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dest_img)
        n_images += 1
        if (i + 1) % step == 0 or (i + 1) == total:
            print(f"    {split_name}: {i + 1}/{total} images", flush=True)
    return n_images, n_labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET to YOLO format (train/val/test)"
    )
    # Default: workspace/Downloads so VisDrone2019-DET-train & VisDrone2019-DET-val are found there
    _ws = Path(__file__).resolve().parent.parent.parent
    _default_input = str(_ws / "Downloads")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=_default_input,
        help="Directory containing VisDrone2019-DET-train and VisDrone2019-DET-val (default: workspace/Downloads)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/visdrone",
        help="Output root (will create train/images, train/labels, val/..., test/...)",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Symlink images instead of copying (saves disk space)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Only convert these splits (default: train val test). Use --splits test for test-dev only.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir or ".").expanduser().resolve()
    if not input_root.exists():
        print(f"ERROR: Input directory not found: {input_root}")
        return 1
    # If they passed a path to VisDrone2019-DET-train or VisDrone2019-DET-val, use its parent
    if input_root.name.startswith("VisDrone2019-DET-"):
        input_root = input_root.parent

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    copy_images = not args.no_copy_images

    print("Converting VisDrone -> YOLO format")
    print(f"  Input:  {input_root}")
    print(f"  Output: {output_root}")
    print(f"  Images: {'copy' if copy_images else 'symlink'}")
    print()

    splits = args.splits if args.splits else ["train", "val", "test"]
    total_img = 0
    total_lbl = 0
    for split in splits:
        n_img, n_lbl = convert_split(input_root, output_root, split, copy_images=copy_images)
        if n_img or n_lbl:
            print(f"  {split}: {n_img} images, {n_lbl} label files")
            total_img += n_img
            total_lbl += n_lbl
        elif split == "test":
            n_img, n_lbl = convert_split(
                input_root, output_root, "test", copy_images=copy_images, source_subdir="test-dev"
            )
            if n_img or n_lbl:
                print(f"  test (from test-dev): {n_img} images, {n_lbl} label files")
                total_img += n_img
                total_lbl += n_lbl

    print()
    if total_img == 0 and total_lbl == 0:
        print("No images converted. Expected layout:")
        print("  VisDrone2019-DET-train/images/  and  VisDrone2019-DET-train/annotations/")
        print("  VisDrone2019-DET-val/images/    and  VisDrone2019-DET-val/annotations/")
        input_resolved = input_root.resolve()
        for name in ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]:
            d = input_resolved / name
            im_lo, an_lo = d / "images", d / "annotations"
            im_hi, an_hi = d / "Images", d / "Annotations"
            n_ann = (len(list(an_lo.glob("*.txt"))) if an_lo.exists() else 0) or (len(list(an_hi.glob("*.txt"))) if an_hi.exists() else 0)
            n_img = (len(list(im_lo.glob("*.jpg"))) + len(list(im_lo.glob("*.png"))) if im_lo.exists() else 0) or (len(list(im_hi.glob("*.jpg"))) + len(list(im_hi.glob("*.png"))) if im_hi.exists() else 0)
            print(f"  {d}")
            print(f"    exists={d.exists()}, images dir={im_lo.exists() or im_hi.exists()} (files: {n_img}), annotations dir={an_lo.exists() or an_hi.exists()} (files: {n_ann})")
            if n_ann and not n_img:
                print("    -> Annotations present but no images. Extract the image zip (e.g. VisDrone2019-DET-train-images.zip) into the 'images' subfolder.")
        return 1
    print(f"Done. Total: {total_img} images, {total_lbl} label files.")
    print("Next: ensure data.yaml exists in", output_root, "(path, train/val/test images, nc: 10, names)")
    print("      python scripts/check_dataset.py --data-root", str(output_root))
    return 0


if __name__ == "__main__":
    exit(main())
