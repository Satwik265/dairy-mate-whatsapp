"""
Teat screening: one photo in, one plain result out. Same result shape as udder/src/predict.py.

    python scripts/predict.py path/to/photo.jpg [more photos...]

Flow:
    1. light quality check (only obvious problems: unreadable, tiny or almost black photos)
    2. shared photo gate (common/models/photo_gate.onnx): teat close-up -> score it;
       whole udder -> "use Udder Check"; nothing found -> retake
    3. score the teat crop AND the whole photo with models/teat_classifier_v2.keras, average,
       then calibrate (models/teat_model_card.json, fitted on the validation split)
    4. return a verdict with a calibrated probability and a confidence level

Honest status (reports/teat_v2_evaluation.json): ROC-AUC 0.95 on held-out test photos, 0.84 on SCTL photos
alone; rough rings (score 3) caught ~45%; healthy teats that don't look like the SCTL set are often flagged
(8/12 in test). The gate sends 100% of teat test photos here and keeps whole-udder photos out.
The crop+whole-photo average was carried over from the udder app; teat test photos are close-ups where
crop and whole photo are nearly the same, so it is untested on wider shots.

The UI should call a positive result "Teat problem detected", never "mastitis".
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAT.parent))
from common.classifier import load_classifier  # noqa: E402
from common.gate import PhotoGate, open_photo, quality_problem, crop  # noqa: E402

MODELS = TEAT / "models"
CLS_SIZE = 224
MIN_TEAT_PX = 32


class TeatScreener:
    """Load the gate and the teat model once, then call .predict() for every photo."""

    def __init__(self, models_dir=MODELS, gate=None, backend="keras"):
        # backend="onnx" runs the converted copy without TensorFlow (website server; see common/classifier.py)
        self.gate = gate or PhotoGate()
        with open(Path(models_dir) / "teat_model_card.json", encoding="utf-8") as f:
            card = json.load(f)
        self.model = load_classifier(Path(models_dir) / card["model"], backend)
        self.A, self.B = card["calibration"]["A"], card["calibration"]["B"]
        self.bands = [(name, lo) for name, lo in card["confidence_bands"]]

    def _raw(self, img):
        x = np.asarray(img.resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR), dtype=np.float32)[None]
        return self.model(x)

    def _calibrate(self, p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(1 / (1 + np.exp(-(self.A * np.log(p / (1 - p)) + self.B))))

    def predict(self, photo):
        result = dict(status="retake", reason_key=None, label=None, probability=None,
                      confidence_level=None, advice_key="advice.retake_photo",
                      disclaimer_key="disclaimer.screening_aid", box=None)
        try:
            img = open_photo(photo)
        except Exception:
            return {**result, "reason_key": "photo_unreadable"}
        problem = quality_problem(img)
        if problem:
            return {**result, "reason_key": problem}

        found = self.gate.detect(img)
        where = self.gate.route(found)
        if where == "udder":
            return {**result, "status": "use_udder_check", "advice_key": "advice.use_udder_check"}
        if where is None:
            return {**result, "reason_key": "no_teat_found"}
        box = found["teat_closeup"][1]
        teat = crop(img, box)
        if min(teat.size) < MIN_TEAT_PX:
            return {**result, "reason_key": "teat_too_far", "box": box}

        p = self._calibrate((self._raw(teat) + self._raw(img)) / 2)
        confidence = max(p, 1 - p)
        level = next(name for name, lo in self.bands if confidence >= lo)
        label = "affected" if p >= 0.5 else "healthy"
        advice = ("advice.see_vet" if label == "affected"
                  else "advice.keep_watching" if level == "high"
                  else "advice.check_again_or_vet")
        return dict(status="ok", reason_key=None, label=label, probability=round(p, 3),
                    confidence_level=level, advice_key=advice,
                    disclaimer_key="disclaimer.screening_aid", box=[round(v, 4) for v in box])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/predict.py photo.jpg [more photos...]")
    screener = TeatScreener()
    for path in sys.argv[1:]:
        print(path)
        print(json.dumps(screener.predict(path), indent=2))
