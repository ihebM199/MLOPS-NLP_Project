"""FastAPI application.

Endpoints (Swagger: ``/docs``):
    GET  /health              -> 200 ok / 503 degraded
    GET  /model/metadata      -> model card + metrics
    POST /predict             -> label, probabilities, fusion weights, latency
    POST /predict/visualize   -> PNG (Grad-CAM overlay) or JSON+base64

The same process also serves the built React frontend (``frontend/dist``) so
that a **single** URL works on Colab behind one ngrok tunnel.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.app.inference import InferenceRegistry, gpu_available
from backend.app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelMetadata,
    PredictionResponse,
    VisualizationResponse,
)
from backend.app.settings import Settings, get_settings
from backend.app.validation import APIError, decode_image
from src.utils.logging import get_logger

LOGGER = get_logger("backend.api")

DESCRIPTION = """
Détection de visages **réels (0)** vs **générés par IA / deepfake (1)**.

* Prétraitement : redimensionnement 224×224, normalisation ImageNet, RGB.
* Flux fréquentiel : `log1p(|fftshift(fft2(x))|)` calculé hors du graphe ONNX.
* Règle de décision : `P(fake) = sigmoid(logit) ≥ seuil` ⇒ **FAKE**.
* `/predict` utilise ONNX Runtime ; `/predict/visualize` utilise PyTorch pour Grad-CAM.

⚠️ Ce modèle est un outil de recherche : il ne doit jamais constituer la seule
preuve de l'authenticité d'un contenu.
"""


class SPAStaticFiles(StaticFiles):
    """Serve the Vite build and fall back to ``index.html`` for client routes."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except Exception:
            index = Path(self.directory) / "index.html"
            if index.exists():
                return FileResponse(index)
            raise


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (tests inject their own settings)."""
    settings = settings or get_settings()
    registry = InferenceRegistry(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOGGER.info("Starting %s v%s", settings.app_name, settings.app_version)
        registry.started_at = time.time()
        await run_in_threadpool(registry.load)
        yield
        registry.close()
        LOGGER.info("Shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        contact={"name": "MLOps project — Tek-Up ING-5 SDIA", "url": "https://github.com/"},
        license_info={"name": "MIT"},
    )
    app.state.settings = settings
    app.state.registry = registry

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-ms", "X-Prediction-Label", "X-Prob-Fake"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - logged by the exception handler
            duration = (time.perf_counter() - started) * 1000
            LOGGER.exception(
                json.dumps(
                    {
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(duration, 2),
                        "status": 500,
                    }
                )
            )
            raise
        duration = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-ms"] = f"{duration:.2f}"
        LOGGER.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration, 2),
                }
            )
        )
        return response

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #
    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):  # noqa: ARG001
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.code, detail=exc.detail, status_code=exc.status_code
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):  # noqa: ARG001
        LOGGER.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_ERROR", detail=f"{exc.__class__.__name__}: {exc}", status_code=500
            ).model_dump(),
        )

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["monitoring"],
        summary="Service and model health",
        responses={503: {"model": ErrorResponse}},
    )
    async def health(request: Request):
        state: InferenceRegistry = request.app.state.registry
        payload = HealthResponse(
            status="ok" if state.model_loaded else "degraded",
            model_loaded=state.model_loaded,
            model_version=settings.model_version,
            gpu_available=gpu_available(),
            onnxruntime_providers=state.onnxruntime_providers,
            gradcam_available=state.gradcam.available,
            uptime_s=state.uptime,
            detail=state.load_error,
        )
        if not state.model_loaded:
            return JSONResponse(status_code=503, content=payload.model_dump())
        return payload

    @app.get(
        "/model/metadata",
        response_model=ModelMetadata,
        tags=["model"],
        summary="Model card, metrics and decision threshold",
    )
    async def model_metadata(request: Request):
        state: InferenceRegistry = request.app.state.registry
        return state.metadata_payload()

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        summary="Classify a face image (REAL / FAKE)",
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict(request: Request, file: UploadFile = File(..., description="JPEG/PNG/WebP/BMP image")):
        state: InferenceRegistry = request.app.state.registry
        data = await file.read(settings.max_upload_bytes + 1)
        image = await run_in_threadpool(
            decode_image,
            data,
            filename=file.filename,
            content_type=file.content_type,
            max_bytes=settings.max_upload_bytes,
        )
        result = await run_in_threadpool(state.predict, image)
        return PredictionResponse(**result)

    @app.post(
        "/predict/visualize",
        tags=["inference"],
        summary="Grad-CAM overlay (PNG by default, JSON+base64 with format=json)",
        responses={
            200: {"content": {"image/png": {}}, "description": "PNG overlay (format=png)"},
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict_visualize(
        request: Request,
        file: UploadFile = File(..., description="JPEG/PNG/WebP/BMP image"),
        target: Literal["fake", "real"] = Query("fake", description="Class to explain"),
        alpha: float = Query(0.45, ge=0.0, le=1.0, description="Heatmap opacity"),
        format: Literal["png", "json"] = Query("png", description="Response format"),
    ):
        state: InferenceRegistry = request.app.state.registry
        started = time.perf_counter()
        data = await file.read(settings.max_upload_bytes + 1)
        image = await run_in_threadpool(
            decode_image,
            data,
            filename=file.filename,
            content_type=file.content_type,
            max_bytes=settings.max_upload_bytes,
        )
        if format == "json":
            encoded, meta = await run_in_threadpool(
                state.gradcam.visualize_base64, image, target=target, alpha=alpha
            )
            total_ms = (time.perf_counter() - started) * 1000
            payload = VisualizationResponse(
                label=meta["label"],
                prob_fake=meta["prob_fake"],
                prob_real=1.0 - meta["prob_fake"],
                target=target,
                alpha=alpha,
                image_base64=encoded,
                latency={
                    "preprocess_ms": 0.0,
                    "inference_ms": meta["latency_ms"],
                    "total_server_ms": round(total_ms, 2),
                },
            )
            return JSONResponse(content=payload.model_dump())
        png, meta = await run_in_threadpool(state.gradcam.visualize, image, target=target, alpha=alpha)
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Prediction-Label": meta["label"],
                "X-Prob-Fake": f"{meta['prob_fake']:.4f}",
                "X-Gradcam-Target": target,
                "Cache-Control": "no-store",
            },
        )

    # ------------------------------------------------------------------ #
    # Frontend (single URL on Colab)
    # ------------------------------------------------------------------ #
    dist = Path(settings.frontend_dist)
    if (dist / "index.html").exists():
        app.mount("/", SPAStaticFiles(directory=str(dist), html=True), name="frontend")
        LOGGER.info("Serving the frontend build from %s", dist)
    else:

        @app.get("/", include_in_schema=False)
        async def root() -> Any:
            return {
                "message": "Deepfake Adaptive Fusion API",
                "docs": "/docs",
                "health": "/health",
                "frontend": "not built — run `npm run build` in frontend/ (then restart the API)",
            }

    return app


app = create_app()
