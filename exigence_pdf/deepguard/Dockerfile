# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# DeepGuard platform — single image: React build + Django (API + SPA).
#
# Build context = exigence_pdf/ (the repository root of the project):
#   docker build -f deepguard/Dockerfile -t deepguard-backend .
# ---------------------------------------------------------------------------
FROM node:20-alpine AS frontend

WORKDIR /web
COPY deepguard/frontend/package.json deepguard/frontend/package-lock.json ./
RUN npm ci
COPY deepguard/frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/deepguard/backend:/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

# CPU-only PyTorch first (avoids the multi-GB CUDA wheels on Linux).
# Pinned to the project's training version; Django 6.1 needs Python >= 3.12.
COPY deepguard/backend/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0 \
    && pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Shared model / data package used by both the FastAPI and Django backends.
COPY src ./src
COPY params.yaml ./params.yaml
COPY deepguard/backend ./deepguard/backend
COPY --from=frontend /web/dist ./deepguard/frontend/dist

RUN mkdir -p /app/models /app/reports /app/monitoring_data \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["gunicorn", "deepguard.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", "--threads", "4", "--timeout", "180"]
