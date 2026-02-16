"""Inference engine: run YOLO26 or RF-DETR on image/video, return detections and annotated output."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── YOLO26 weights ─────────────────────────────────────────────────────────
YOLO_WEIGHTS = REPO_ROOT / "models" / "Yolo26" / "weights" / "best.pt"

# ── RF-DETR weights ────────────────────────────────────────────────────────
_RFDETR_DIR = REPO_ROOT / "models" / "rfdetr" / "weights"
_RFDETR_CANDIDATES = [
    "checkpoint_best_total.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_ema.pth",
]
RFDETR_WEIGHTS = _RFDETR_DIR / "checkpoint_best_total.pth"
for _name in _RFDETR_CANDIDATES:
    if (_RFDETR_DIR / _name).exists():
        RFDETR_WEIGHTS = _RFDETR_DIR / _name
        break

# Number of classes the fine-tuned VisDrone RF-DETR checkpoint was trained on
RFDETR_NUM_CLASSES = 10

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

print(f"[engine] Python executable : {sys.executable}")
print(f"[engine] Python version    : {sys.version}")
print(f"[engine] YOLO weights      : {YOLO_WEIGHTS}  (exists={YOLO_WEIGHTS.exists()})")
print(f"[engine] RF-DETR weights   : {RFDETR_WEIGHTS}  (exists={RFDETR_WEIGHTS.exists()})")


# ── RF-DETR model cache (singleton) ───────────────────────────────────────
_rfdetr_model = None  # cached RFDETRBase instance


def _get_rfdetr_model(weights_path: str = ""):
    """Return a cached RFDETRBase model, loading it only once."""
    global _rfdetr_model
    w = weights_path or str(RFDETR_WEIGHTS)

    if _rfdetr_model is not None:
        return _rfdetr_model

    from rfdetr import RFDETRBase
    print(f"[engine] Loading RF-DETR model from: {w}")
    print(f"[engine] num_classes={RFDETR_NUM_CLASSES}  (VisDrone: {VISDRONE_NAMES})")
    model = RFDETRBase(pretrain_weights=w, num_classes=RFDETR_NUM_CLASSES)
    print(f"[engine] RF-DETR model loaded successfully ✓")
    _rfdetr_model = model
    return model


# ── YOLO26 inference ───────────────────────────────────────────────────────

def run_yolo26(image_path: str, weights_path: str = "", device: str = "") -> list[dict]:
    from ultralytics import YOLO
    w = weights_path or str(YOLO_WEIGHTS)
    model = YOLO(w)
    kwargs = {"device": device} if device else {}
    results = model.predict(image_path, **kwargs)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            xyxy = r.boxes.xyxy[i].tolist()
            out.append({
                "class_id": cls_id,
                "class_name": VISDRONE_NAMES[cls_id] if cls_id < len(VISDRONE_NAMES) else str(cls_id),
                "confidence": round(conf, 4),
                "bbox": [round(x, 2) for x in xyxy],
            })
    return out


# ── RF-DETR inference ──────────────────────────────────────────────────────

def run_rfdetr(image_path: str, weights_path: str = "", device: str = "") -> list[dict]:
    """Run RF-DETR inference using the fine-tuned .pth checkpoint."""
    model = _get_rfdetr_model(weights_path)
    detections = model.predict(image_path)
    out = []

    # predict() returns a supervision.Detections object
    if hasattr(detections, "xyxy") and detections.xyxy is not None:
        for i in range(len(detections.xyxy)):
            xyxy = detections.xyxy[i].tolist()
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0

            # Map class IDs to 0-based VisDrone names
            if cls_id >= len(VISDRONE_NAMES):
                cls_id = len(VISDRONE_NAMES) - 1
            if cls_id < 0:
                cls_id = 0

            out.append({
                "class_id": cls_id,
                "class_name": VISDRONE_NAMES[cls_id],
                "confidence": round(conf, 4),
                "bbox": [round(x, 2) for x in xyxy],
            })
    return out


# ── Annotated Image Helpers ────────────────────────────────────────────────

def get_annotated_image_yolo(image_path: str, out_path: str, weights_path: str = "") -> None:
    from ultralytics import YOLO
    w = weights_path or str(YOLO_WEIGHTS)
    model = YOLO(w)
    results = model.predict(image_path, save=False)
    if not results:
        import shutil
        shutil.copy(image_path, out_path)
        return
    im = results[0].plot()
    import cv2
    cv2.imwrite(out_path, im)


def get_annotated_image_rfdetr(image_path: str, out_path: str, weights_path: str = "") -> None:
    import cv2
    dets = run_rfdetr(image_path, weights_path)
    img = cv2.imread(image_path)
    if img is None:
        import shutil
        shutil.copy(image_path, out_path)
        return
    for d in dets:
        bbox = d.get("bbox", [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.get('class_name', '')} {d.get('confidence', 0):.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, img)


# ── Annotated Video Helpers ────────────────────────────────────────────────

def get_annotated_video_yolo(video_path: str, out_path: str, weights_path: str = "") -> None:
    from ultralytics import YOLO
    import shutil
    w = weights_path or str(YOLO_WEIGHTS)
    model = YOLO(w)
    out_dir = Path(out_path).parent
    with tempfile.TemporaryDirectory(dir=str(out_dir)) as tmp:
        results = model.predict(video_path, save=True, project=tmp, name="out")
        run_dir = Path(tmp) / "out"
        if run_dir.exists():
            mp4s = list(run_dir.glob("*.mp4"))
            if mp4s:
                shutil.move(str(mp4s[0]), out_path)
                return
        run_dir = Path(tmp)
        mp4s = list(run_dir.rglob("*.mp4"))
        if mp4s:
            shutil.move(str(mp4s[0]), out_path)


def get_annotated_video_rfdetr(video_path: str, out_path: str, weights_path: str = "") -> None:
    import cv2
    model = _get_rfdetr_model(weights_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    with tempfile.TemporaryDirectory() as tmp:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = Path(tmp) / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            detections = model.predict(str(frame_path))
            if hasattr(detections, "xyxy") and detections.xyxy is not None:
                for i in range(len(detections.xyxy)):
                    xyxy = detections.xyxy[i].tolist()
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                    cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                    if cls_id >= len(VISDRONE_NAMES):
                        cls_id = len(VISDRONE_NAMES) - 1
                    if cls_id < 0:
                        cls_id = 0

                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{VISDRONE_NAMES[cls_id]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            writer.write(frame)
            frame_idx += 1
    cap.release()
    writer.release()
