"""Model zoo: the final dual-stream detector + the ablation baselines.

Weight compatibility is contractual. The module names of ``fusion`` mode are
byte-for-byte the ones of the pre-trained checkpoint
(``AdaptiveFusion_best_model.pth``):

    spatial_net, freq_net, spatial_projection, frequency_projection,
    fusion_gate, classifier, residual_scale

Label convention: **real = 0, fake = 1**; decision rule: ``logit > 0 => FAKE``.

Ablations (a ``mode`` argument, same weights-loading code path):

===============  ==========================================================
``fusion``       the final model: ``z = p_s + α·(w_s·p_s + w_f·p_f)``
``spatial_only``  EfficientNetV2 stream only (baseline of the comparison)
``freq_only``     FFT stream only
``concat_no_gate`` fixed average of both projections (no learned gate)
===============  ==========================================================

All modes instantiate every sub-module so a ``fusion`` checkpoint can initialise
the baselines (fair comparison) and so ``state_dict`` keys stay identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.config import BACKBONE_CANDIDATES, MODES
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.logging import get_logger

LOGGER = get_logger("models.architecture")

#: Buffers injected by ``thop.profile`` in the source notebook (must be dropped).
_THOP_SUFFIXES = ("total_ops", "total_params")


class AdaptiveFusionDetector(nn.Module):
    """Spatial (EfficientNetV2) + frequency (FFT log-magnitude CNN) deepfake detector."""

    def __init__(
        self,
        backbone: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        projection_dim: Optional[int] = None,
        mode: str = "fusion",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r} (expected one of {MODES})")
        self.backbone_name = backbone
        self.mode = mode

        # 1. Spatial stream
        self.spatial_net = timm_create(backbone, pretrained)
        self.spatial_features_dim = self.spatial_net.num_features

        # 2. Frequency stream: log(1+|FFT|) -> small CNN
        self.freq_net = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        self.freq_features_dim = 64 * 7 * 7
        self.projection_dim = projection_dim or self.spatial_features_dim

        # 3. Projections into a shared space
        self.spatial_projection = nn.Sequential(
            nn.Linear(self.spatial_features_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
        )
        self.frequency_projection = nn.Sequential(
            nn.Linear(self.freq_features_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
        )

        # 4. Adaptive gate (starts at 0.5 / 0.5 because the last layer is zeroed)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.projection_dim * 2),
            nn.Linear(self.projection_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

        # 5. Classifier (one logit: > 0 means FAKE)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.projection_dim),
            nn.Linear(self.projection_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------ #
    # Frequency preprocessing (must stay identical to the ONNX path)
    # ------------------------------------------------------------------ #
    @staticmethod
    def frequency_input(x: torch.Tensor) -> torch.Tensor:
        """``log1p(|fftshift(fft2(x))|)`` in float32.

        FFT in float16 is only supported for power-of-two sizes (224 is not), so
        the transform is forced to float32 even under AMP. The exact same
        formula is reproduced with ``numpy.fft`` in the ONNX/API path.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fftshift(
                torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1)
            )
            return torch.log1p(torch.abs(spectrum))

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward_detailed(
        self, x: torch.Tensor, freq_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return ``(logits, (w_spatial, w_frequency), spatial_feature_map)``.

        ``freq_input`` lets the ONNX graph receive the pre-computed spectrum
        (``torch.fft`` does not export cleanly) — see ``docs/DECISIONS.md``.
        """
        fmap = self.spatial_net.forward_features(x)
        spatial_feat = self.spatial_net.forward_head(fmap)
        p_s = self.spatial_projection(spatial_feat)

        freq = freq_input if freq_input is not None else self.frequency_input(x)
        p_f = self.frequency_projection(self.freq_net(freq))

        gate = torch.softmax(self.fusion_gate(torch.cat([p_s, p_f], dim=1)), dim=1)
        w_s, w_f = gate[:, 0:1], gate[:, 1:2]

        if self.mode == "fusion":
            fused = p_s + self.residual_scale * (w_s * p_s + w_f * p_f)
        elif self.mode == "spatial_only":
            fused = p_s
            w_s, w_f = torch.ones_like(w_s), torch.zeros_like(w_f)
        elif self.mode == "freq_only":
            fused = p_f
            w_s, w_f = torch.zeros_like(w_s), torch.ones_like(w_f)
        else:  # concat_no_gate
            fused = 0.5 * (p_s + p_f)
            w_s, w_f = torch.full_like(w_s, 0.5), torch.full_like(w_f, 0.5)
        return self.classifier(fused), (w_s, w_f), fmap

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        logits, weights, _ = self.forward_detailed(x)
        return (logits, weights) if return_weights else logits

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def active_parameters(self) -> int:
        """Parameters actually used by the current mode (unused modules excluded)."""
        modules: Sequence[nn.Module] = [self.spatial_projection, self.classifier]
        if self.mode in ("fusion", "spatial_only", "concat_no_gate"):
            modules.append(self.spatial_net)
        if self.mode in ("fusion", "freq_only", "concat_no_gate"):
            modules += [self.freq_net, self.frequency_projection]
        if self.mode == "fusion":
            modules.append(self.fusion_gate)
        total = sum(p.numel() for module in modules for p in module.parameters())
        return int(total + (self.residual_scale.numel() if self.mode == "fusion" else 0))

    def describe(self) -> Dict[str, Any]:
        return {
            "architecture": "AdaptiveFusionDetector",
            "mode": self.mode,
            "backbone": self.backbone_name,
            "spatial_features_dim": self.spatial_features_dim,
            "freq_features_dim": self.freq_features_dim,
            "projection_dim": self.projection_dim,
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "active_parameters": self.active_parameters(),
            "residual_scale": float(self.residual_scale.detach().cpu()),
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            "label2id": {"real": 0, "fake": 1},
            "decision_rule": "logit > 0 -> fake (label 1); P(fake) = sigmoid(logit)",
        }


def timm_create(backbone: str, pretrained: bool) -> nn.Module:
    """``timm.create_model`` with a Colab-friendly offline fallback.

    Downloading ImageNet weights needs internet; on a restricted runtime we log
    a warning instead of crashing (training then starts from scratch).
    """
    import timm

    try:
        return timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    except Exception as exc:  # noqa: BLE001 - network / hub errors
        if not pretrained:
            raise
        LOGGER.warning(
            "Could not load ImageNet weights for %s (%s) — continuing with random init. "
            "Set train.pretrained=false to silence this message.",
            backbone,
            exc.__class__.__name__,
        )
        return timm.create_model(backbone, pretrained=False, num_classes=0)


def build_model(
    cfg=None,
    *,
    mode: Optional[str] = None,
    backbone: Optional[str] = None,
    pretrained: Optional[bool] = None,
    projection_dim: Optional[int] = None,
) -> AdaptiveFusionDetector:
    """Build a model from a :class:`src.config.TrainConfig` (or explicit args)."""
    mode = mode or (cfg.train.mode if cfg else "fusion")
    backbone = backbone or (cfg.train.backbone if cfg else "efficientnetv2_rw_s")
    pretrained = pretrained if pretrained is not None else (cfg.train.pretrained if cfg else True)
    return AdaptiveFusionDetector(
        backbone=backbone, pretrained=pretrained, projection_dim=projection_dim, mode=mode
    )


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` prefixes and the ``thop`` profiling buffers."""
    cleaned = {}
    for key, value in state.items():
        if key.endswith(_THOP_SUFFIXES):
            continue
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def read_checkpoint(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a raw ``state_dict`` or a full training checkpoint (``weights_only``)."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a state_dict")
    return clean_state_dict(obj)


def infer_architecture(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Read projection/spatial dimensions from the weights themselves."""
    key = "spatial_projection.0.weight"
    if key not in state:
        raise ValueError(
            "This file is not an AdaptiveFusionDetector checkpoint "
            f"(missing '{key}'). Keys start with: {list(state)[:5]}"
        )
    proj_dim, spatial_dim = state[key].shape
    return {"projection_dim": int(proj_dim), "spatial_features_dim": int(spatial_dim)}


def load_checkpoint(
    path: str | Path,
    *,
    backbone: Optional[str] = None,
    candidates: Sequence[str] = BACKBONE_CANDIDATES,
    mode: str = "fusion",
    device: str | torch.device = "cpu",
) -> Tuple[AdaptiveFusionDetector, Dict[str, Any]]:
    """Rebuild the exact architecture of a checkpoint and load it strictly.

    The backbone is auto-detected: ``model_config.json`` (if provided through
    ``backbone``) is tried first, then the configured backbone, then the known
    candidates, until ``load_state_dict(strict=True)`` succeeds.
    """
    state = read_checkpoint(path)
    dims = infer_architecture(state)
    names = [name for name in dict.fromkeys([backbone, *candidates]) if name]
    errors: list[str] = []
    for name in names:
        try:
            model = AdaptiveFusionDetector(
                name, pretrained=False, projection_dim=dims["projection_dim"], mode=mode
            )
            if model.spatial_features_dim != dims["spatial_features_dim"]:
                errors.append(
                    f"{name}: features {model.spatial_features_dim} != checkpoint "
                    f"{dims['spatial_features_dim']}"
                )
                continue
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            info = {
                "backbone": name,
                **dims,
                "path": str(path),
                "mode": mode,
                "parameters": int(sum(p.numel() for p in model.parameters())),
            }
            LOGGER.info(
                "Checkpoint loaded: %s (backbone=%s, projection_dim=%d)",
                Path(path).name,
                name,
                dims["projection_dim"],
            )
            return model, info
        except Exception as exc:  # noqa: BLE001 - try the next backbone
            errors.append(f"{name}: {str(exc).splitlines()[0][:120]}")
    raise RuntimeError(
        "No backbone matches the checkpoint "
        f"(spatial dim {dims['spatial_features_dim']}, projection dim {dims['projection_dim']}).\n"
        "Tried:\n  - " + "\n  - ".join(errors)
    )


def model_config(model: AdaptiveFusionDetector, *, image_size: int = 224, **extra: Any) -> Dict[str, Any]:
    """Metadata persisted next to the weights (used by train, API and ONNX export)."""
    payload = model.describe()
    payload.update({"image_size": image_size, "weights_file": "AdaptiveFusion_best_model.pth"})
    payload.update(extra)
    return payload
