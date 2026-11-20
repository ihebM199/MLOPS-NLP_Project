"""Grad-CAM for the spatial stream (+ overlay rendering).

Grad-CAM needs gradients, therefore it cannot run in ONNX Runtime. The backend
uses ONNX for ``/predict`` (fast) and PyTorch for ``/predict/visualize``
(lazy-loaded) — decision documented in ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from src.data.transforms import denormalize
from src.utils.logging import get_logger

LOGGER = get_logger("utils.gradcam")


def _colormap(values: np.ndarray, name: str = "jet") -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return (cmap(np.clip(values, 0, 1))[..., :3] * 255).astype(np.uint8)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """Blend a ``[0,1]`` heatmap over an RGB image (both ``HxWx3`` / ``HxW``)."""
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    colour = _colormap(np.asarray(heatmap, dtype=np.float32), cmap).astype(np.float32) / 255.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = (1 - alpha) * image + alpha * colour
    return np.clip(blended, 0, 1)


def to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.max() <= 1.5:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(to_uint8(array)).save(buffer, format="PNG")
    return buffer.getvalue()


class GradCAM:
    """Gradient-weighted Class Activation Mapping on the spatial feature map."""

    def __init__(self, model: Any, device: Any = "cpu") -> None:
        self.model = model
        self.device = device

    def heatmap(self, tensor: Any, target: str = "fake") -> Tuple[np.ndarray, Dict[str, float]]:
        """Return ``(heatmap HxW in [0,1], {"logit":..., "prob_fake":...})``.

        ``tensor`` is a single preprocessed image ``(1, 3, H, W)``.
        ``target="fake"`` explains the FAKE logit, ``target="real"`` its negative.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        tensor = tensor.to(self.device)
        with torch.enable_grad():
            logits, _, feature_map = self.model.forward_detailed(tensor)
            score = logits[:, 0] if target.lower() == "fake" else -logits[:, 0]
            grads = torch.autograd.grad(score.sum(), feature_map)[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        prob_fake = float(torch.sigmoid(logits)[0, 0].detach().cpu())
        return cam.detach().cpu().numpy(), {
            "logit": float(logits.detach()[0, 0]),
            "prob_fake": prob_fake,
        }

    def explain(
        self,
        image: Image.Image,
        transform: Any,
        *,
        target: str = "fake",
        alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """Explain a PIL image: returns the RGB input, heatmap, overlay and PNG bytes."""
        import torch

        tensor = transform(image.convert("RGB")).unsqueeze(0)
        cam, prediction = self.heatmap(tensor, target=target)
        rgb = denormalize(tensor[0]).permute(1, 2, 0).detach().cpu().numpy()
        blended = overlay(rgb, cam, alpha=alpha)
        return {
            "image": rgb,
            "heatmap": cam,
            "overlay": blended,
            "png": to_png_bytes(blended),
            "png_heatmap": to_png_bytes(_colormap(cam)),
            "prediction": prediction,
        }


def gradcam_examples(
    model: Any,
    paths: list[str],
    transform: Any,
    *,
    device: Any = "cpu",
    target: str = "fake",
    alpha: float = 0.45,
    labels: Optional[list[int]] = None,
) -> list[Dict[str, Any]]:
    """Build the rows consumed by ``plots.plot_gradcam_grid`` and the API demo."""
    explainer = GradCAM(model, device=device)
    from src.data.dataset import ID2LABEL

    rows: list[Dict[str, Any]] = []
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except OSError as exc:
            LOGGER.warning("Skipping %s (%s)", path, exc)
            continue
        result = explainer.explain(image, transform, target=target, alpha=alpha)
        truth = f"true: {ID2LABEL[labels[index]].upper()} " if labels else ""
        rows.append(
            {
                "image": result["image"],
                "heatmap": result["heatmap"],
                "overlay": result["overlay"],
                "title": f"{truth}P(fake) = {result['prediction']['prob_fake']:.1%}",
                "path": path,
            }
        )
    return rows
