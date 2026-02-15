"""Convert YOLO-format VisDrone (data/visdrone) to COCO format for RF-DETR training.

RF-DETR expects: dataset_dir/train/, valid/, test/ each with _annotations.coco.json and images.

Usage:
  python scripts/convert_yolo_to_coco.py --input data/visdrone --output data/visdrone_coco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

# (name, supercategory) for RF-DETR (requires supercategory != "none")
VISDRONE_CATEGORIES = [
    ("pedestrian", "person"),
    ("people", "person"),
    ("bicycle", "vehicle"),
    ("car", "vehicle"),
    ("van", "vehicle"),
    ("truck", "vehicle"),
    ("tricycle", "vehicle"),
    ("awning-tricycle", "vehicle"),
    ("bus", "vehicle"),
    ("motor", "vehicle"),
]
VISDRONE_NAMES = [c[0] for c in VISDRONE_CATEGORIES]


def get_image_size(img_path: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]
    return 1920, 1080


def convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    val_folder: str = "val",
) -> int:
    """Convert one split from YOLO to COCO. Returns number of images."""
    # YOLO: train/images, train/labels; valid can be "val" or "valid"
    if split == "valid":
        img_dir = input_root / val_folder / "images"
        lbl_dir = input_root / val_folder / "labels"
    else:
        img_dir = input_root / split / "images"
        lbl_dir = input_root / split / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        return 0

    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": i + 1, "name": name, "supercategory": supercat}
        for i, (name, supercat) in enumerate(VISDRONE_CATEGORIES)
    ]
    images = []
    annotations = []
    ann_id = 1

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for img_path in img_dir.glob(ext):
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue

            img_w, img_h = get_image_size(img_path)
            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            })

            # Copy image to output
            import shutil
            dest_img = out_dir / img_path.name
            if not dest_img.exists() or dest_img.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, dest_img)

            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except (ValueError, IndexError):
                        continue
                    if cls_id < 0 or cls_id >= len(VISDRONE_NAMES):
                        continue
                    # YOLO normalized -> absolute bbox (COCO: x_min, y_min, width, height)
                    x_min = (xc - w / 2) * img_w
                    y_min = (yc - h / 2) * img_h
                    w_abs = w * img_w
                    h_abs = h * img_h
                    x_min = max(0, min(img_w - 1, x_min))
                    y_min = max(0, min(img_h - 1, y_min))
                    w_abs = max(1, min(img_w - x_min, w_abs))
                    h_abs = max(1, min(img_h - y_min, h_abs))
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,  # COCO 1-indexed
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w_abs, 2), round(h_abs, 2)],
                        "area": round(w_abs * h_abs, 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_json = out_dir / "_annotations.coco.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"  {split}: {len(images)} images, {len(annotations)} annotations -> {out_json}")
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO VisDrone to COCO for RF-DETR")
    parser.add_argument("--input", type=str, default="data/visdrone", help="YOLO dataset root")
    parser.add_argument("--output", type=str, default="data/visdrone_coco", help="COCO output root")
    parser.add_argument("--val-folder", type=str, default="val", help="Validation folder name (val or valid)")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    if not input_root.exists():
        print(f"ERROR: Input not found: {input_root}")
        return 1

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"YOLO -> COCO: {input_root} -> {output_root}")
    print()

    n_train = convert_split(input_root, output_root, "train", args.val_folder)
    n_valid = convert_split(input_root, output_root, "valid", args.val_folder)
    n_test = convert_split(input_root, output_root, "test", args.val_folder)

    print()
    print(f"Done. Total: {n_train} train, {n_valid} valid, {n_test} test.")
    print("Next: python scripts/train_rfdetr.py --dataset-dir", str(output_root))
    return 0


if __name__ == "__main__":
    exit(main())
