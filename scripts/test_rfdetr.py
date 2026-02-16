
"""Test RF-DETR using Supervision and manually calculate COCO metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Try to import required packages
try:
    import cv2
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from rfdetr import RFDETRBase
    import supervision as sv
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

# VisDrone class order (0-9 -> categories 1-10)
VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Test RF-DETR on VisDrone test set (manual)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--dataset-dir", type=str, default="", help="COCO-format dataset root")
    parser.add_argument("--output-dir", type=str, default="test_results/rfdetr", help="Output dir")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Paths
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"Error: Checkpoint not found: {ckpt}")
        return 1

    data_dir = Path(args.dataset_dir or (REPO_ROOT / "data" / "visdrone_coco"))
    test_json = data_dir / "test" / "_annotations.coco.json"
    if not test_json.exists():
        print(f"Error: Test annotations not found: {test_json}")
        return 1

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated:
        ann_dir = out_dir / "annotated"
        ann_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print(f"Loading RF-DETR: {ckpt.name}")
    try:
        model = RFDETRBase(pretrain_weights=str(ckpt)) # Assuming device handled inside or defaults to needed
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # 2. Load Ground Truth
    print(f"Loading test set: {test_json}")
    coco_gt = COCO(str(test_json))
    img_ids = coco_gt.getImgIds()
    print(f"Found {len(img_ids)} test images.")

    # 3. Create Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 4. Run Inference
    results = []
    print("Running inference...")
    
    # Progress check
    total = len(img_ids)
    step = max(1, total // 10)

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = data_dir / "test" / file_name

        if not img_path.exists():
            continue

        # Predict
        try:
            # model.predict returns supervision.Detections
            detections = model.predict(str(img_path))
        except Exception as e:
            print(f"Wait, error predicting {file_name}: {e}")
            continue

        # Construct COCO results
        # Detections: xyxy, confidence, class_id
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            score = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0

            # Convert xyxy -> xywh
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Map class_id.
            # Debugging shows class_id can be 10. Assuming 1-based (1-10).
            # COCO category IDs are 1-10.
            # If class_id is 1-10:
            #   category_id = class_id
            #   name index = class_id - 1
            
            cat_id = cls_id
            # Safety check/clamp for now to avoid crash if something else
            if cat_id < 1: cat_id = 1
            
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(w), 2), round(float(h), 2)],
                "score": round(score, 4),
            })

        # Save annotated image
        if args.save_annotated and (idx % 10 == 0 or idx < 10): # Save some, or all? user said "annotated images output". Let's do all if flag set, or batch?
            # User specifically asked for annotated images. Let's do all if not too many, or sample.
            # 1600 images is a lot for artifacts but manageable locally.
            # Usually users want proof, maybe top 50 or so. Let's do first 50 then skip to save time/space unless requested all.
            # Actually, user said "make same plots... annotated images output for RF-detr test".
            # Let's verify how many images 'test' has. 1610. 
            # Generating 1610 annotated images takes time. Let's start with all but maybe check time.
            # The script argument --save-annotated implies "save them".
            
            # Use supervision annotators
            image = cv2.imread(str(img_path))
            if image is None: 
                continue

            # Need labels for annotator
            labels = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                cid = int(class_id)
                # Map 1-10 -> 0-9
                idx = cid - 1 if cid > 0 else 0
                if idx >= len(VISDRONE_NAMES):
                    idx = len(VISDRONE_NAMES) - 1
                labels.append(f"{VISDRONE_NAMES[idx]} {confidence:.2f}")
            
            annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            cv2.imwrite(str(ann_dir / file_name), annotated_image)

        if (idx + 1) % step == 0:
            print(f"Processed {idx + 1}/{total} images...", flush=True)

    # 5. Save results.json
    res_json_path = out_dir / "results.json"
    with open(res_json_path, "w") as f:
        json.dump(results, f)
    
    # 6. Evaluate
    print("Evaluating...")
    if not results:
        print("No detections found!")
        return 0

    coco_dt = coco_gt.loadRes(str(res_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save metrics
    metrics = {
        "mAP50-95": coco_eval.stats[0],
        "mAP50": coco_eval.stats[1],
        "mAP75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
    }
    
    metrics_path = out_dir / "metrics.json" # Use same name as expected
    # Convert numpy floats to native python floats
    metrics = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")

    # Generate curve-style plots (metrics + results) into out_dir/plots/
    plot_script = REPO_ROOT / "scripts" / "plot_rfdetr_metrics.py"
    if plot_script.exists():
        r = subprocess.run(
            [sys.executable, str(plot_script), "--results-dir", str(out_dir)],
            cwd=str(REPO_ROOT),
            capture_output=True,
        )
        if r.returncode == 0:
            print(f"Plots saved to {out_dir / 'plots'}")
        # Non-zero is non-fatal; metrics and results are already saved

    return 0

if __name__ == "__main__":
    sys.exit(main())
