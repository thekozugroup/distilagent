# syntax=docker/dockerfile:1.7

# ----- Stage 1: build the React/shadcn frontend ------------------------------
FROM node:20-slim AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY web/ ./
RUN npm run build

# ----- Stage 2: API runtime with distilabel + scripts ------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISTILAGENT_DATA=/data \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install distilabel fork (needed so `AutoReasonedGeneration` imports work).
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir 'httpx<0.28' datasets openai huggingface_hub

# API deps.
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# App code.
COPY api/ ./api/
COPY scripts/ ./scripts/

# Frontend bundle.
COPY --from=web-builder /web/dist ./web/dist

# Data dir lives on a volume so runs persist across restarts.
RUN mkdir -p /data/input /data/output /data/checkpoints /data/logs
VOLUME ["/data"]

EXPOSE 8080

# Healthcheck hits the FastAPI health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
