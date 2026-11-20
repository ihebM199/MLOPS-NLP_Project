"""Typed configuration loaded from ``params.yaml``.

Every stage (prepare / train / evaluate / export) reads the same object, so a
single YAML file drives the whole pipeline — locally, in Docker and on Colab.

    from src.config import load_config
    cfg = load_config("params.yaml").with_quick_demo()
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from src.utils.paths import MODEL_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_dir

#: Backbones tried (in order) when a checkpoint does not match the configured one.
BACKBONE_CANDIDATES: Tuple[str, ...] = (
    "efficientnetv2_rw_s",
    "regnety_064",
    "regnety_040",
    "regnety_032",
    "convnext_tiny",
    "efficientnet_b3",
    "resnet50",
)

#: Model variants used for the mandatory baseline-vs-advanced comparison.
MODES: Tuple[str, ...] = ("fusion", "spatial_only", "freq_only", "concat_no_gate")


@dataclass
class DataParams:
    dataset: str = "xhlulu/140k-real-and-fake-faces"
    dataset_subdir: str = "real_vs_fake/real-vs-fake"
    image_size: int = 224
    train_per_class: Optional[int] = 10_000
    valid_per_class: Optional[int] = 2_000
    test_per_class: Optional[int] = 5_000
    num_workers: int = 2
    seed: int = 42
    # --- advanced augmentation (Axis 1) ---
    advanced_augmentation: bool = True
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 7
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5


@dataclass
class TrainParams:
    mode: str = "fusion"
    backbone: str = "efficientnetv2_rw_s"
    pretrained: bool = True
    epochs: int = 30
    batch_size: int = 32
    patience: int = 5
    weight_decay: float = 0.05
    lr_backbone: float = 1e-5
    lr_fusion: float = 1e-4
    lr_classifier: float = 2e-4
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    learn_residual_scale: bool = True
    use_amp: bool = True
    # --- losses / fine-tuning ablations ---
    focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    freeze_backbone_epochs: int = 0  # >0: linear-probe warm-up, then gradual unfreeze
    resume: bool = False
    seed: int = 42
    deterministic: bool = False
    # --- reproducibility of the comparison ---
    run_name: str = "R4_fusion"


@dataclass
class EvalParams:
    split: str = "test"
    threshold: float = 0.5  # P(fake) >= threshold -> FAKE
    calibration: bool = True
    robustness: bool = True
    jpeg_qualities: Tuple[int, ...] = (75, 50, 30)
    blur_sigmas: Tuple[float, ...] = (1.0, 2.0)
    resize_factors: Tuple[float, ...] = (0.5, 0.25)
    max_robustness_samples: int = 400
    cross_dataset: bool = False
    external_max_per_class: int = 500


@dataclass
class ExportParams:
    opset: int = 17
    path: str = "models/deepfake_fusion.onnx"
    fp16: bool = True
    int8: bool = True
    calibration_samples: int = 200
    parity_images: int = 64
    parity_tolerance: float = 1e-3


@dataclass
class TrackingParams:
    enabled: bool = True
    tracking_uri: str = ""  # empty -> sqlite:///mlruns/mlflow.db
    experiment: str = "deepfake-adaptive-fusion"
    registered_model: str = "deepfake-adaptive-fusion"
    register: bool = False
    alias: str = "staging"
    log_artifacts: bool = True
    log_system_metrics: bool = False


@dataclass
class TrainConfig:
    data: DataParams = field(default_factory=DataParams)
    train: TrainParams = field(default_factory=TrainParams)
    eval: EvalParams = field(default_factory=EvalParams)
    export: ExportParams = field(default_factory=ExportParams)
    tracking: TrackingParams = field(default_factory=TrackingParams)
    quick_demo: bool = False
    model_dir: str = str(MODEL_DIR)
    reports_dir: str = str(REPORTS_DIR)
    unknown_keys: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "TrainConfig":
        raw = dict(raw or {})
        unknown: list[str] = []

        def build(dc_type: Any, values: Mapping[str, Any]) -> Any:
            kwargs = {}
            for f in dataclasses.fields(dc_type):
                if f.name in values:
                    kwargs[f.name] = values[f.name]
            for key in values:
                if key not in {f.name for f in dataclasses.fields(dc_type)}:
                    unknown.append(f"{dc_type.__name__}.{key}")
            obj = dc_type(**kwargs)
            # tuples must stay tuples even if YAML gives a list
            for f in dataclasses.fields(dc_type):
                if isinstance(getattr(obj, f.name), list) and f.default is not dataclasses.MISSING:
                    if isinstance(f.default, tuple):
                        setattr(obj, f.name, tuple(getattr(obj, f.name)))
            return obj

        cfg = cls(
            data=build(DataParams, raw.get("data", {})),
            train=build(TrainParams, raw.get("train", {})),
            eval=build(EvalParams, raw.get("eval", {})),
            export=build(ExportParams, raw.get("export", {})),
            tracking=build(TrackingParams, raw.get("tracking", {})),
            quick_demo=bool(raw.get("quick_demo", False)),
            model_dir=str(raw.get("model_dir", MODEL_DIR)),
            reports_dir=str(raw.get("reports_dir", REPORTS_DIR)),
            unknown_keys=tuple(unknown),
        )
        if cfg.quick_demo:
            cfg.apply_quick_demo()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path = "params.yaml", *, required: bool = False) -> "TrainConfig":
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Configuration file not found: {path}")
            return cls()
        return cls.from_dict(yaml.safe_load(path.read_text("utf-8")) or {})

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def apply_quick_demo(self) -> "TrainConfig":
        """500 img/class, 2 epochs: validates the whole pipeline in minutes."""
        self.quick_demo = True
        self.data.train_per_class = 500
        self.data.valid_per_class = 200
        self.data.test_per_class = 200
        self.train.epochs = min(self.train.epochs, 2)
        self.train.patience = min(self.train.patience, 2)
        self.eval.max_robustness_samples = min(self.eval.max_robustness_samples, 100)
        self.export.int8 = False  # dynamic INT8 needs a few hundred calibration images
        return self

    def with_quick_demo(self, enabled: bool = True) -> "TrainConfig":
        return self.apply_quick_demo() if enabled else self

    def with_overrides(self, **overrides: Any) -> "TrainConfig":
        """``cfg.with_overrides(epochs=5, mode="spatial_only")`` — ``None`` ignored."""
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, _, name = dotted.partition("__")
            if name:
                if hasattr(self, section):
                    target = getattr(self, section)
                    if hasattr(target, name):
                        setattr(target, name, value)
                        continue
                raise AttributeError(f"Unknown parameter: {dotted}")
            if hasattr(self, section):
                setattr(self, section, value)
            else:
                raise AttributeError(f"Unknown parameter: {dotted}")
        return self

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def reports_path(self) -> Path:
        return Path(self.reports_dir)

    @property
    def weights_path(self) -> Path:
        return self.model_path / "AdaptiveFusion_best_model.pth"

    @property
    def onnx_path(self) -> Path:
        p = Path(self.export.path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def tracking_uri(self) -> str:
        """MLflow tracking URI: env > params > SQLite store (registry-compatible)."""
        env = os.environ.get("MLFLOW_TRACKING_URI")
        if env:
            return env
        if self.tracking.tracking_uri:
            return self.tracking.tracking_uri
        db = (self.model_path.parent / "mlruns" / "mlflow.db").resolve()
        return f"sqlite:///{db.as_posix()}"

    def prepare_dirs(self) -> "TrainConfig":
        ensure_dir(self.model_path)
        ensure_dir(self.reports_path)
        ensure_dir(self.reports_path / "figures")
        return self

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.pop("unknown_keys", None)
        return payload

    def save_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload.pop("quick_demo", None)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), "utf-8")
        return path

    def summary(self) -> str:
        t = self.train
        d = self.data
        return (
            f"mode={t.mode} backbone={t.backbone} epochs={t.epochs} batch={t.batch_size} "
            f"img={d.image_size} train/class={d.train_per_class} val/class={d.valid_per_class} "
            f"test/class={d.test_per_class} amp={t.use_amp} focal={t.focal_loss} "
            f"adv_aug={d.advanced_augmentation} seed={d.seed} quick_demo={self.quick_demo}"
        )


def load_config(path: str | Path = "params.yaml", *, quick_demo: bool = False) -> TrainConfig:
    """Load ``params.yaml`` (falling back to defaults) and optionally shrink it."""
    cfg = TrainConfig.from_yaml(path)
    return cfg.with_quick_demo(quick_demo)
