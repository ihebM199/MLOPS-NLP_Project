"""Shared fixtures: a fake ONNX model with the *exact same signature* as the real one.

The test suite never needs the 132 MB checkpoint nor the dataset: the fake graph
exposes the same two inputs (``image``, ``freq_input``) and three outputs
(``logit``, ``w_spatial``, ``w_frequency``), so every code path of the API is
exercised on CPU in a few seconds.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def build_fake_onnx(path: Path) -> Path:
    """Tiny graph: logit = 10 * mean(image), w_spatial = sigmoid(mean(freq))."""
    import onnx
    from onnx import TensorProto, helper

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
    graph = helper.make_graph(nodes, "fake_detector", [image, freq], [logit, w_spatial, w_frequency])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="session")
def fake_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("models")
    return build_fake_onnx(directory / "fake_deepfake.onnx")


@pytest.fixture()
def image_bytes() -> bytes:
    """A valid 320x240 JPEG (gradient, deterministic)."""
    array = np.zeros((240, 320, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
    array[..., 1] = np.linspace(0, 255, 240, dtype=np.uint8)[:, None]
    array[..., 2] = 128
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


@pytest.fixture()
def white_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def black_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()
