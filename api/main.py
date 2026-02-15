"""Inference API: switch model (YOLO26 / RF-DETR), upload image or video, get annotated result."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.engine import (
    get_annotated_image_rfdetr,
    get_annotated_image_yolo,
    get_annotated_video_rfdetr,
    get_annotated_video_yolo,
    run_rfdetr,
    run_yolo26,
)

app = FastAPI(title="VisDrone Inference API", version="1.0")

API_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = API_ROOT / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTS


@app.get("/", response_class=HTMLResponse)
def index():
    html = (API_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/infer")
async def infer(
    model: str = Form(..., description="yolo26 or rfdetr"),
    file: UploadFile = File(...),
):
    if model not in ("yolo26", "rfdetr"):
        raise HTTPException(400, "model must be yolo26 or rfdetr")

    ext = Path(file.filename or "x").suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        raise HTTPException(400, "Upload an image or video (jpg, png, mp4, etc.)")

    vid = is_video(file.filename or "")
    suffix = ".mp4" if vid else (ext or ".jpg")
    uid = str(uuid.uuid4())[:8]
    input_path = RESULTS_DIR / f"in_{uid}{suffix}"
    output_path = RESULTS_DIR / f"out_{uid}{suffix}"

    try:
        content = await file.read()
        input_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(500, f"Save upload failed: {e}")

    try:
        if vid:
            if model == "yolo26":
                get_annotated_video_yolo(str(input_path), str(output_path))
            else:
                get_annotated_video_rfdetr(str(input_path), str(output_path))
            detections = []
        else:
            if model == "yolo26":
                detections = run_yolo26(str(input_path))
                get_annotated_image_yolo(str(input_path), str(output_path))
            else:
                detections = run_rfdetr(str(input_path))
                get_annotated_image_rfdetr(str(input_path), str(output_path))
    except Exception as e:
        if input_path.exists():
            input_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Inference failed: {e}")

    input_path.unlink(missing_ok=True)
    result_filename = output_path.name

    return {
        "model": model,
        "detections": detections,
        "result_url": f"/results/{result_filename}",
        "result_filename": result_filename,
    }


app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
