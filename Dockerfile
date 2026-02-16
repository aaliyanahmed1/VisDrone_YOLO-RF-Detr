# ============================================================================
# VisDrone Inference API — Docker Image
# Models: YOLO26 (CNN) + RF-DETR (Transformer)
# ============================================================================
# Build:   docker build -t visdrone-api .
# Run:     docker run -p 8000:8000 --gpus all visdrone-api
# Run CPU: docker run -p 8000:8000 visdrone-api
# ============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements-docker.txt .

RUN pip install --prefix=/install --no-cache-dir -r requirements-docker.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Allow PyTorch MPS fallback (no-op on Linux, safe to keep)
    PYTORCH_ENABLE_MPS_FALLBACK=1

WORKDIR /app

# System dependencies for OpenCV (headless) and general libs
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl; \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# ── Application code ──────────────────────────────────────────────────────
COPY api/ api/

# ── Model weights ─────────────────────────────────────────────────────────
# YOLO26 fine-tuned weights (~57 MB)
COPY models/Yolo26/weights/best.pt models/Yolo26/weights/best.pt

# RF-DETR fine-tuned weights (~122 MB)
COPY models/rfdetr/weights/checkpoint_best_total.pth models/rfdetr/weights/checkpoint_best_total.pth

# Create results directory for annotated outputs
RUN mkdir -p api/static/results

# ── Security: non-root user ──────────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
