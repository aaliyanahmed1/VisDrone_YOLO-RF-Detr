"""Inference engine: run YOLO26 or RF-DETR on image/video, return detections and annotated output."""

from __future__ import annotations

import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = REPO_ROOT / "models" / "Yolo26" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = REPO_ROOT / "runs" / "detect" / "runs" / "yolo26" / "visdrone" / "weights" / "best.pt"
RFDETR_WEIGHTS_DIR = REPO_ROOT / "models" / "rfdetr" / "weights"
if not RFDETR_WEIGHTS_DIR.exists() or not list(RFDETR_WEIGHTS_DIR.glob("checkpoint*.pth")):
    RFDETR_WEIGHTS_DIR = REPO_ROOT / "runs" / "rfdetr" / "visdrone_20260214_145610"

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def _resolve_rfdetr_ckpt(weights_path: str) -> str:
    p = Path(weights_path)
    if p.suffix == ".pth" and p.exists():
        return str(p)
    for name in ("checkpoint_best_regular.pth", "checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best.pth", "checkpoint_last.pth"):
        c = p / name
        if c.exists():
            return str(c)
    for c in sorted(p.glob("checkpoint_*.pth"), reverse=True):
        return str(c)
    return str(p / "checkpoint_best_total.pth")


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


def run_rfdetr(image_path: str, weights_path: str = "", device: str = "") -> list[dict]:
    from rfdetr import RFDETRBase
    w = weights_path or str(RFDETR_WEIGHTS_DIR)
    ckpt = _resolve_rfdetr_ckpt(w)
    model = RFDETRBase(pretrain_weights=ckpt)
    detections = model.predict(image_path)
    out = []
    if hasattr(detections, "__iter__"):
        for d in detections:
            if isinstance(d, dict):
                cls_id = d.get("category_id", d.get("class_id", 0)) - 1
                if cls_id < 0:
                    cls_id = 0
                out.append({
                    "class_id": cls_id,
                    "class_name": VISDRONE_NAMES[cls_id] if cls_id < len(VISDRONE_NAMES) else str(cls_id),
                    "confidence": round(float(d.get("score", d.get("confidence", 0))), 4),
                    "bbox": d.get("bbox", d.get("box", [])),
                })
    return out


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
    dets = run_rfdetr(image_path, weights_path or str(RFDETR_WEIGHTS_DIR))
    img = cv2.imread(image_path)
    if img is None:
        import shutil
        shutil.copy(image_path, out_path)
        return
    h, w = img.shape[:2]
    for d in dets:
        bbox = d.get("bbox", [])
        if len(bbox) == 4:
            x, y, bw, bh = bbox
            x2, y2 = x + bw, y + bh
        elif len(bbox) >= 4:
            x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            continue
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        label = f"{d.get('class_name', '')} {d.get('confidence', 0):.2f}"
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, img)


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
    from rfdetr import RFDETRBase
    w = weights_path or str(RFDETR_WEIGHTS_DIR)
    ckpt = _resolve_rfdetr_ckpt(w)
    model = RFDETRBase(pretrain_weights=ckpt)
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
            dets = run_rfdetr(str(frame_path), w)
            for d in dets:
                bbox = d.get("bbox", [])
                if len(bbox) >= 4:
                    x, y = int(bbox[0]), int(bbox[1])
                    if len(bbox) == 4:
                        x2, y2 = x + int(bbox[2]), y + int(bbox[3])
                    else:
                        x2, y2 = int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    label = f"{d.get('class_name', '')} {d.get('confidence', 0):.2f}"
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(frame)
            frame_idx += 1
    cap.release()
    writer.release()
