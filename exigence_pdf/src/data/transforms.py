"""Torchvision transforms — train (augmented) and eval (deterministic).

Rule enforced by the whole pipeline: **no augmentation ever touches valid/test**.
The eval transform is also the reference preprocessing used by the FastAPI
backend, so a prediction made through the API is bit-for-bit the one computed
in ``evaluate.py``.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

#: ImageNet statistics (EfficientNetV2 pretraining).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int = 224,
    *,
    advanced: bool = True,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 7,
) -> transforms.Compose:
    """Training transform.

    ``advanced=True`` adds **RandAugment** (Axis-1 requirement) on top of the
    reference recipe (RandomResizedCrop + flip + ColorJitter + RandomErasing).
    MixUp/CutMix are applied after batching, see ``src.models.augment``.
    """
    steps: list[Any] = [transforms.Resize((256, 256))]
    if advanced:
        steps.append(transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude))
    steps += [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(p=0.1)], p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ]
    return transforms.Compose(steps)


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing (valid / test / API / ONNX)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize(tensor: Any) -> Any:
    """Undo ImageNet normalization for display (keeps the tensor on its device)."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def preprocess_config(image_size: int = 224) -> dict[str, Any]:
    """Serialisable description shared with the backend and ``model_config.json``."""
    return {
        "image_size": image_size,
        "resize": [image_size, image_size],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "color_mode": "RGB",
        "value_range": "ImageNet-normalised float32",
    }
