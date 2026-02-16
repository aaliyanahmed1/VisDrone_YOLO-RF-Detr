# 🐳 VisDrone Inference API — Docker Setup

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Option 2: Docker CLI

```bash
# Build the image
docker build -t visdrone-api .

# Run (CPU)
docker run -p 8000:8000 visdrone-api

# Run (GPU — requires nvidia-container-toolkit)
docker run -p 8000:8000 --gpus all visdrone-api
```

---

## 🌐 Access the API

Once running, open your browser:
- **Web UI**: [http://localhost:8000](http://localhost:8000)
- **API docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Usage

### Inference Endpoint

```
POST /infer
```

**Form fields:**
| Field   | Type   | Values              | Description                    |
|---------|--------|---------------------|--------------------------------|
| `model` | string | `yolo26` or `rfdetr`| Which model to use             |
| `file`  | file   | image or video      | The file to run detection on   |

### Example: cURL

```bash
# Image inference with RF-DETR
curl -X POST http://localhost:8000/infer \
  -F "model=rfdetr" \
  -F "file=@my_image.jpg"

# Image inference with YOLO26
curl -X POST http://localhost:8000/infer \
  -F "model=yolo26" \
  -F "file=@my_image.jpg"
```

### Example: Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/infer",
    data={"model": "rfdetr"},
    files={"file": open("my_image.jpg", "rb")},
)
result = resp.json()
print(f"Detections: {len(result['detections'])}")
for det in result["detections"]:
    print(f"  {det['class_name']}: {det['confidence']:.1%} at {det['bbox']}")
```

### Response Format

```json
{
  "model": "rfdetr",
  "detections": [
    {
      "class_id": 3,
      "class_name": "car",
      "confidence": 0.9234,
      "bbox": [120.5, 230.1, 340.8, 410.3]
    }
  ],
  "result_url": "/results/out_abc12345.jpg",
  "result_filename": "out_abc12345.jpg"
}
```

---

## 🏷️ Supported Classes (VisDrone)

| ID | Class Name       |
|----|------------------|
| 0  | pedestrian       |
| 1  | people           |
| 2  | bicycle          |
| 3  | car              |
| 4  | van              |
| 5  | truck            |
| 6  | tricycle         |
| 7  | awning-tricycle  |
| 8  | bus              |
| 9  | motor            |

---

## 🖥️ Models Included

| Model   | Architecture    | Weights File                | Size   |
|---------|-----------------|-----------------------------|--------|
| YOLO26  | CNN (YOLO)      | `models/Yolo26/weights/best.pt` | ~57 MB |
| RF-DETR | Transformer (DETR) | `models/rfdetr/weights/checkpoint_best_total.pth` | ~122 MB |

Both models are fine-tuned on the VisDrone dataset for drone-view object detection.

---

## 🔧 GPU Support

By default, the Docker image uses **CPU-only PyTorch** to keep the image small.

For **GPU inference**, edit `requirements-docker.txt`:

```diff
- --extra-index-url https://download.pytorch.org/whl/cpu
+ --extra-index-url https://download.pytorch.org/whl/cu121
```

Then uncomment the GPU section in `docker-compose.yml` and rebuild:

```bash
docker compose up --build
```

> **Note**: GPU support requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## 📦 Sharing the Image

```bash
# Save image to a tar file
docker save visdrone-api:latest | gzip > visdrone-api.tar.gz

# Load on another machine
docker load < visdrone-api.tar.gz

# Run
docker run -p 8000:8000 visdrone-api:latest
```

Or push to a registry:

```bash
# Tag for your registry
docker tag visdrone-api:latest your-registry.com/visdrone-api:latest

# Push
docker push your-registry.com/visdrone-api:latest
```
