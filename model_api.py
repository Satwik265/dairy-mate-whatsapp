"""Lazy-loading wrappers around the supplied Dairy Mate model predictors."""
import importlib.util
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent
_instances = {}
_lock = Lock()


def _load_predictor(module_name, file_path, class_name, **kwargs):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load predictor from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)(**kwargs)


def get_screener(kind):
    with _lock:
        if kind in _instances:
            return _instances[kind]
        if kind == "udder":
            instance = _load_predictor(
                "dairymate_udder_predict", ROOT / "udder" / "src" / "predict.py", "UdderScreener",
                models_dir=ROOT / "udder" / "models", backend="onnx"
            )
        elif kind == "teat":
            instance = _load_predictor(
                "dairymate_teat_predict", ROOT / "teat" / "scripts" / "predict.py", "TeatScreener",
                models_dir=ROOT / "teat" / "models", backend="onnx"
            )
        elif kind == "tick":
            instance = _load_predictor(
                "dairymate_tick_predict", ROOT / "tick" / "scripts" / "predict.py", "TickScreener",
                models_dir=ROOT / "tick" / "models"
            )
        else:
            raise ValueError("Unknown model; expected udder, teat, or tick")
        _instances[kind] = instance
        return instance
