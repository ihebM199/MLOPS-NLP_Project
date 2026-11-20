"""Upload validation unit tests."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from backend.app.validation import (
    EmptyUploadError,
    FileTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
    decode_image,
)


def _jpeg(size: tuple[int, int] = (120, 90), color: tuple[int, int, int] = (20, 130, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_decode_valid_image_returns_rgb():
    image = decode_image(_jpeg(), content_type="image/jpeg")
    assert image.mode == "RGB"
    assert image.size == (120, 90)


def test_decode_without_content_type():
    assert decode_image(_jpeg()).size == (120, 90)


def test_decode_empty():
    with pytest.raises(EmptyUploadError):
        decode_image(b"", filename="x.jpg")


def test_decode_too_large():
    with pytest.raises(FileTooLargeError):
        decode_image(b"0" * 1024, max_bytes=512)


def test_decode_unsupported_content_type():
    with pytest.raises(UnsupportedFormatError):
        decode_image(_jpeg(), content_type="application/pdf")


def test_decode_garbage():
    with pytest.raises(InvalidImageError):
        decode_image(b"garbage bytes", content_type="image/jpeg")


def test_decode_truncated():
    payload = _jpeg()
    with pytest.raises(InvalidImageError):
        decode_image(payload[: len(payload) // 3], content_type="image/jpeg")


def test_error_codes_and_statuses():
    assert EmptyUploadError("x").status_code == 400
    assert FileTooLargeError("x").status_code == 413
    assert UnsupportedFormatError("x").status_code == 415
    assert InvalidImageError("x").status_code == 400
