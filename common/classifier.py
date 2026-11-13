"""
Runs a udder/teat classifier from its Keras file or from the ONNX copy next to it.

    backend="keras"  models/<name>.keras with Keras/TensorFlow (training venv; the default)
    backend="onnx"   models/<name>.onnx with onnxruntime only: no TensorFlow, so the website fits in 512 MB

The ONNX files are made by web/deploy/convert_to_onnx.py and checked against Keras on the held-out test photos
by web/tools/check_onnx_parity.py (web/reports/onnx_parity.json).
"""

from pathlib import Path

import numpy as np


def load_classifier(keras_path, backend="keras"):
    """Returns a function: float32 array (1, 224, 224, 3), raw 0-255 -> the model's sigmoid output (float)."""
    keras_path = Path(keras_path)
    if backend == "keras":
        import keras
        model = keras.models.load_model(keras_path)
        return lambda x: float(model(x, training=False).numpy().ravel()[0])
    if backend == "onnx":
        from common.ort_session import ort_session
        session = ort_session(keras_path.with_suffix(".onnx"))
        name = session.get_inputs()[0].name
        return lambda x: float(np.asarray(session.run(None, {name: x})[0]).ravel()[0])
    raise ValueError(f"backend must be 'keras' or 'onnx', not {backend!r}")
