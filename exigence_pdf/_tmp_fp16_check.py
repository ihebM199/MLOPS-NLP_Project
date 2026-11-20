import warnings

import numpy as np
import onnx
from onnxruntime import InferenceSession
from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_convert

warnings.filterwarnings("ignore")
p = "models/deepfake_fusion.onnx"
model = onnx.load_model(p)

candidates = {}
for name, kwargs in {
    "ort_keepio": {"keep_io_types": True},
    "ort_no_keepio": {"keep_io_types": False},
}.items():
    try:
        converted = ort_convert(model, **kwargs)
        path = f"models/_try_{name}.onnx"
        onnx.save_model(converted, path)
        candidates[name] = path
        print(name, "converted")
    except Exception as exc:
        print(name, "convert failed:", type(exc).__name__, str(exc)[:200])

x = np.random.randn(2, 3, 224, 224).astype(np.float32)
spec = np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))
f = np.log1p(np.abs(spec)).astype(np.float32)

for name, path in candidates.items():
    try:
        sess = InferenceSession(path, providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        feed = {}
        for i in inputs:
            arr = x if "freq" not in i.name else f
            feed[i.name] = arr.astype(np.float16) if i.type == "tensor(float16)" else arr
        names = [o.name for o in sess.get_outputs()]
        outs = sess.run(names, feed)
        print(name, "run OK", [np.asarray(o).dtype for o in outs], [np.asarray(o).ravel()[:2] for o in outs])
    except Exception as exc:
        print(name, "FAILED:", type(exc).__name__, str(exc)[:300])
