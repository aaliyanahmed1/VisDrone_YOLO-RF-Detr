# VisDrone inference API: ONNX YOLO + web UI. Production image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Minimal system deps for OpenCV (headless)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0; \
    rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY api/ api/
COPY models/ models/
RUN mkdir -p api/static/results

# Non-root user (create after copy so ownership is correct)
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
