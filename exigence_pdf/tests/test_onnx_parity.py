"""ONNX contract tests (dummy graph) + optional real-model parity (slow)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.onnx_export import OnnxRunner, frequency_input_numpy, softmax_gate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_ONNX = PROJECT_ROOT / "models" / "deepfake_fusion.onnx"
REAL_CKPT = PROJECT_ROOT / "models" / "AdaptiveFusion_best_model.pth"


def _build_dummy(path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])
    freq = helper.make_tensor_value_info("freq_input", TensorProto.FLOAT, ["batch", 3, 224, 224])
    logit = helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch", 1])
    w_spatial = helper.make_tensor_value_info("w_spatial", TensorProto.FLOAT, ["batch", 1])
    w_frequency = helper.make_tensor_value_info("w_frequency", TensorProto.FLOAT, ["batch", 1])

    nodes = [
        helper.make_node("ReduceMean", ["image"], ["m"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Flatten", ["m"], ["mf"], axis=1),
        helper.make_node(
            "Constant", [], ["ten"], value=helper.make_tensor("t", TensorProto.FLOAT, [1, 1], [10.0])
        ),
        helper.make_node("Mul", ["mf", "ten"], ["logit"]),
        helper.make_node("ReduceMean", ["freq_input"], ["f"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Flatten", ["f"], ["ff"], axis=1),
        helper.make_node("Sigmoid", ["ff"], ["w_spatial"]),
        helper.make_node(
            "Constant", [], ["one"], value=helper.make_tensor("o", TensorProto.FLOAT, [1, 1], [1.0])
        ),
        helper.make_node("Sub", ["one", "w_spatial"], ["w_frequency"]),
    ]
    graph = helper.make_graph(nodes, "dummy", [image, freq], [logit, w_spatial, w_frequency])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="module")
def dummy_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_dummy(tmp_path_factory.mktemp("onnx") / "dummy.onnx")


def test_frequency_input_shape_and_finiteness() -> None:
    rng = np.random.default_rng(0)
    images = rng.random((4, 3, 224, 224), dtype=np.float32)
    freq = frequency_input_numpy(images)
    assert freq.shape == images.shape
    assert np.isfinite(freq).all()
    assert (freq >= 0).all()


def test_softmax_gate_is_a_distribution() -> None:
    weights = softmax_gate(np.array([[2.0, -1.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert (weights >= 0).all()


def test_dummy_onnx_signature_and_prediction(dummy_onnx: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(dummy_onnx), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["image", "freq_input"]
    assert [item.name for item in session.get_outputs()] == ["logit", "w_spatial", "w_frequency"]

    runner = OnnxRunner(dummy_onnx)
    prediction = runner.predict(np.full((3, 224, 224), -1.0, dtype=np.float32))
    assert prediction["label"] == "real"
    neutral = runner.predict(np.zeros((3, 224, 224), dtype=np.float32))
    assert neutral["prob_fake"] == pytest.approx(0.5)
    assert 0.0 <= prediction["prob_fake"] <= 1.0
    assert 0.0 <= prediction["w_spatial"] <= 1.0
    assert 0.0 <= prediction["w_frequency"] <= 1.0
    assert prediction["w_spatial"] + prediction["w_frequency"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_real_model_onnx_parity() -> None:
    if not (REAL_ONNX.exists() and REAL_CKPT.exists()):
        pytest.skip("real checkpoint/ONNX not available")
    torch = pytest.importorskip("torch")
    from src.config import load_config
    from src.models.evaluate import load_model_for_evaluation
    from src.models.onnx_export import check_parity

    cfg = load_config(PROJECT_ROOT / "params.yaml")
    model, _ = load_model_for_evaluation(cfg, REAL_CKPT, torch.device("cpu"))
    result = check_parity(model, REAL_ONNX, n_images=8)
    (PROJECT_ROOT / "reports" / "onnx_parity_test.json").write_text(json.dumps(result, indent=2), "utf-8")
    assert result["passed"], result
