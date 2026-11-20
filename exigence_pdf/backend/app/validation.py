"""Upload validation and typed HTTP errors.

The content type is only a hint: every image is decoded with PIL, which is the
only way to reject a text file renamed ``photo.jpg``.
"""

from __future__ import annotations

import io
from typing import Optional

from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}


class APIError(Exception):
    """Base class turned into an :class:`ErrorResponse` by the exception handler."""

    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, detail: str, *, code: Optional[str] = None, status_code: Optional[int] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class EmptyUploadError(APIError):
    status_code = 400
    code = "EMPTY_FILE"


class UnsupportedFormatError(APIError):
    status_code = 415
    code = "UNSUPPORTED_FORMAT"


class InvalidImageError(APIError):
    status_code = 400
    code = "INVALID_IMAGE"


class FileTooLargeError(APIError):
    status_code = 413
    code = "FILE_TOO_LARGE"


class ModelNotLoadedError(APIError):
    status_code = 503
    code = "MODEL_NOT_LOADED"


class GradCamUnavailableError(APIError):
    status_code = 503
    code = "GRADCAM_UNAVAILABLE"


def decode_image(
    data: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
) -> Image.Image:
    """Validate then decode an upload into an RGB PIL image.

    Raises :class:`EmptyUploadError`, :class:`FileTooLargeError`,
    :class:`UnsupportedFormatError` or :class:`InvalidImageError`.
    """
    if not data:
        raise EmptyUploadError(f"The file {filename or ''!r} is empty.".replace("  ", " "))
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"File is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB."
        )
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedFormatError(
            f"Content type {content_type!r} is not supported (use JPEG, PNG, WebP or BMP)."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode the image ({exc.__class__.__name__}).") from exc
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:  # truncated files pass verify()
        raise InvalidImageError("The image is truncated or corrupted.") from exc


def read_upload(upload, max_bytes: int) -> bytes:  # pragma: no cover - thin wrapper
    """Read at most ``max_bytes + 1`` bytes from an ``UploadFile`` (sync helper)."""
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds the {max_bytes / 1e6:.1f} MB limit.")
    return data
