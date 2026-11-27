"""HTTP endpoints — port of the FastAPI routes in ``backend/app/main.py``.

Contract (identical to the original service):

    GET  /health              -> 200 ok / 503 degraded
    GET  /model/metadata      -> model card + metrics
    POST /predict             -> label, probabilities, fusion weights, latency
    POST /predict/visualize   -> PNG (Grad-CAM overlay) or JSON+base64

The same process serves the platform front-end (``frontend/index.html``), so a
single URL works — exactly like the original single-URL Colab deployment.
"""

from __future__ import annotations

import time
from functools import wraps
from pathlib import Path

from django.http import FileResponse, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from api.config import get_settings
from api.inference import InferenceRegistry, gpu_available
from api.monitoring import record_prediction
from api.validation import APIError, decode_image
from src.utils.logging import get_logger

LOGGER = get_logger("backend.api")

# Module-level registry, loaded once at import time (the ONNX session is
# tiny); PyTorch stays lazily loaded for Grad-CAM, as in the original.
registry = InferenceRegistry(get_settings())
registry.load()


class MissingFileFieldError(APIError):
    """Missing multipart ``file`` field.

    FastAPI rejects this case with its *automatic* validation body (before any
    handler runs) — reproduced here byte-for-byte so the two services answer
    identically in every observable case.
    """

    FASTAPI_BODY = (
        '{"detail":[{"type":"missing","loc":["body","file"],' '"msg":"Field required","input":null}]}'
    )

    def __init__(self) -> None:
        super().__init__("Field required", code="VALIDATION_ERROR", status_code=422)


def api_guard(view):
    """Turn APIError into the homogeneous error body (as the FastAPI handler)."""

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except MissingFileFieldError:
            # Compact JSON, exactly like FastAPI's default renderer.
            return HttpResponse(
                MissingFileFieldError.FASTAPI_BODY, content_type="application/json", status=422
            )
        except APIError as exc:
            return JsonResponse(
                {"error": exc.code, "detail": exc.detail, "status_code": exc.status_code},
                status=exc.status_code,
            )
        except Exception as exc:  # noqa: BLE001 - homogeneous 500
            LOGGER.exception("Unhandled error")
            return JsonResponse(
                {"error": "INTERNAL_ERROR", "detail": f"{type(exc).__name__}: {exc}", "status_code": 500},
                status=500,
            )

    return wrapper


def _read_upload(request) -> tuple[bytes, str | None, str | None]:
    """Multipart field ``file`` (same name as the original ``UploadFile``)."""
    settings = get_settings()
    upload = request.FILES.get("file")
    if upload is None:
        raise MissingFileFieldError()
    data = upload.file.read(settings.max_upload_bytes + 1)
    return data, upload.name, upload.content_type


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@require_GET
@api_guard
def health(request):
    payload = {
        "status": "ok" if registry.model_loaded else "degraded",
        "model_loaded": registry.model_loaded,
        "model_version": registry.settings.model_version,
        "gpu_available": gpu_available(),
        "onnxruntime_providers": registry.onnxruntime_providers,
        "gradcam_available": registry.gradcam.available,
        "uptime_s": registry.uptime,
        "detail": registry.load_error,
    }
    return JsonResponse(payload, status=200 if registry.model_loaded else 503)


@require_GET
@api_guard
def model_metadata(request):
    return JsonResponse(registry.metadata_payload())


@csrf_exempt
@require_POST
@api_guard
def predict(request):
    settings = get_settings()
    data, filename, content_type = _read_upload(request)
    image = decode_image(
        data,
        filename=filename,
        content_type=content_type,
        max_bytes=settings.max_upload_bytes,
    )
    result = registry.predict(image)
    record_prediction(
        result,
        float((result.get("latency") or {}).get("total_server_ms", 0.0)),
        upload_bytes=len(data),
    )
    LOGGER.info(
        "POST /predict -> %s (p_fake=%.4f, %s)",
        result["label"],
        result["prob_fake"],
        request.META.get("REMOTE_ADDR", "-"),
    )
    return JsonResponse(result)


@csrf_exempt
@require_POST
@api_guard
def predict_visualize(request):
    settings = get_settings()
    query = request.GET
    target = query.get("target", "fake")
    if target not in ("fake", "real"):
        raise APIError(
            f"target must be 'fake' or 'real' (got {target!r}).",
            code="VALIDATION_ERROR",
            status_code=422,
        )
    try:
        alpha = float(query.get("alpha", "0.45"))
    except ValueError as exc:
        raise APIError(
            f"alpha must be a number (got {query.get('alpha')!r}).",
            code="VALIDATION_ERROR",
            status_code=422,
        ) from exc
    if not 0.0 <= alpha <= 1.0:
        raise APIError("alpha must be between 0.0 and 1.0.", code="VALIDATION_ERROR", status_code=422)
    fmt = query.get("format", "png")
    if fmt not in ("png", "json"):
        raise APIError(
            f"format must be 'png' or 'json' (got {fmt!r}).",
            code="VALIDATION_ERROR",
            status_code=422,
        )

    started = time.perf_counter()
    data, filename, content_type = _read_upload(request)
    image = decode_image(
        data,
        filename=filename,
        content_type=content_type,
        max_bytes=settings.max_upload_bytes,
    )

    if fmt == "json":
        encoded, meta = registry.gradcam.visualize_base64(image, target=target, alpha=alpha)
        total_ms = (time.perf_counter() - started) * 1000
        payload = {
            "label": meta["label"],
            "prob_fake": meta["prob_fake"],
            "prob_real": 1.0 - meta["prob_fake"],
            "target": target,
            "alpha": alpha,
            "image_base64": encoded,
            "latency": {
                "preprocess_ms": 0.0,
                "inference_ms": meta["latency_ms"],
                "total_server_ms": round(total_ms, 2),
            },
        }
        return JsonResponse(payload)

    png, meta = registry.gradcam.visualize(image, target=target, alpha=alpha)
    response = HttpResponse(png, content_type="image/png")
    response["X-Prediction-Label"] = meta["label"]
    response["X-Prob-Fake"] = f"{meta['prob_fake']:.4f}"
    response["X-Gradcam-Target"] = target
    response["Cache-Control"] = "no-store"
    return response


# --------------------------------------------------------------------------- #
# Front-end (built React app, single URL — like the original FastAPI mount)
# --------------------------------------------------------------------------- #
@require_GET
def home(request):
    index = Path(get_settings().frontend_path)
    if index.exists():
        return FileResponse(open(index, "rb"), content_type="text/html")
    return JsonResponse(
        {
            "message": "Deepfake Adaptive Fusion API",
            "health": "/health",
            "model": "/model/metadata",
            "predict": "POST /predict (multipart 'file')",
            "frontend": "not built — run `npm install && npm run dev` in frontend/ "
            "(dev server on http://localhost:5173, API proxied here), or "
            "`npm run build` to serve it from this port",
        }
    )
