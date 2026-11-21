"""Create a tiny ONNX model with the *same signature* as the real detector.

Used by CI / docker smoke tests so that no 38 MB checkpoint is required:

    python scripts/make_dummy_model.py --out models

It writes ``models/deepfake_fusion.onnx`` (2 inputs / 3 outputs),
``models/model_config.json`` and a deterministic ``models/sample.jpg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image


def build_dummy_onnx(path: Path) -> Path:
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])
    freq = helper.make_tensor_value_info("freq_input", TensorProto.FLOAT, ["batch", 3, 224, 224])
    logit = helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch", 1])
    w_spatial = helper.make_tensor_value_info("w_spatial", TensorProto.FLOAT, ["batch", 1])
    w_frequency = helper.make_tensor_value_info("w_frequency", TensorProto.FLOAT, ["batch", 1])

    nodes = [
        helper.make_node("ReduceMean", ["image"], ["image_mean"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Flatten", ["image_mean"], ["image_mean_flat"], axis=1),
        helper.make_node(
            "Constant",
            [],
            ["ten"],
            value=helper.make_tensor("ten_value", TensorProto.FLOAT, [1, 1], [10.0]),
        ),
        helper.make_node("Mul", ["image_mean_flat", "ten"], ["logit"]),
        helper.make_node("ReduceMean", ["freq_input"], ["freq_mean"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Flatten", ["freq_mean"], ["freq_mean_flat"], axis=1),
        helper.make_node("Sigmoid", ["freq_mean_flat"], ["w_spatial"]),
        helper.make_node(
            "Constant",
            [],
            ["one"],
            value=helper.make_tensor("one_value", TensorProto.FLOAT, [1, 1], [1.0]),
        ),
        helper.make_node("Sub", ["one", "w_spatial"], ["w_frequency"]),
    ]
    graph = helper.make_graph(nodes, "dummy_detector", [image, freq], [logit, w_spatial, w_frequency])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def make_sample_image(path: Path, size: tuple[int, int] = (320, 240)) -> Path:
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    array[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array[..., 2] = 128
    Image.fromarray(array).save(path, format="JPEG", quality=92)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a dummy ONNX model for smoke tests.")
    parser.add_argument("--out", default="models")
    args = parser.parse_args(argv)

    out = Path(args.out)
    onnx_path = build_dummy_onnx(out / "deepfake_fusion.onnx")
    sample = make_sample_image(out / "sample.jpg")
    config = {
        "architecture": "AdaptiveFusionDetector",
        "mode": "fusion",
        "backbone": "dummy",
        "image_size": 224,
        "parameters": 0,
        "note": "Dummy model for CI smoke tests only — not a trained detector.",
    }
    (out / "model_config.json").write_text(json.dumps(config, indent=2), "utf-8")
    print(f"Dummy ONNX: {onnx_path}\nSample image: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
