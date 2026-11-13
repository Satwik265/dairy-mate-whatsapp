"""
Udder screening: one photo in, one plain result out. This is the interface the website / API calls.

    python src/predict.py path/to/photo.jpg [more photos...]

Flow:
    1. light quality check (only obvious problems: unreadable, tiny or almost black photos)
    2. shared photo gate (common/models/photo_gate.onnx) finds the udder, or a teat close-up
    3. crop the udder (+5% margin); score the crop AND the whole photo with
       models/udder_classifier.keras and average the two
    4. return a verdict with a calibrated probability and a confidence level

End-to-end on held-out test photos with this gate (reports/udder_pipeline_evaluation_detector_v2.json):
53/55 udder photos scored, ROC-AUC 0.93, 12% of healthy flagged, 90% of mastitis caught.

Result fields (same shape for every outcome, so the app has one code path):
    status            "ok" | "retake" | "use_teat_check"
    reason_key        why we can't score ("photo_unreadable", "too_dark", "too_small", "no_udder_found",
                      "udder_too_far") or None
    label             "healthy" | "affected" | None
    probability       calibrated P(affected) in 0-1, or None
    confidence_level  "high" | "medium" | "low" | None
    advice_key        i18n key for the one-line "what to do next"
    disclaimer_key    always "disclaimer.screening_aid"
    box               udder box [x0, y0, x1, y1] as fractions of the photo, or None

The keys are looked up in the app's translation files, so no user-facing text lives here.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from common.classifier import load_classifier  # noqa: E402
from common.gate import PhotoGate, open_photo, quality_problem, crop  # noqa: E402

MODELS = ROOT / "models"
CLS_SIZE = 224             # classifier input (stretched square)
MIN_UDDER_PX = 32          # shorter side of the udder crop; only truly tiny udders are sent back.
                           # 96 px rejected 11/52 test photos that were all scored correctly.
BANDS = [("high", 0.85), ("medium", 0.65), ("low", 0.0)]


class UdderScreener:
    """Load both models once, then call .predict() for every photo."""

    def __init__(self, models_dir=MODELS, gate=None, backend="keras"):
        # backend="onnx" runs the converted copy without TensorFlow (website server; see common/classifier.py)
        self.gate = gate or PhotoGate()
        self.classifier = load_classifier(Path(models_dir) / "udder_classifier.keras", backend)

    def _classify(self, img):
        x = np.asarray(img.resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR), dtype=np.float32)[None]
        return self.classifier(x)

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
        if where == "teat":
            return {**result, "status": "use_teat_check", "advice_key": "advice.use_teat_check"}
        if where is None:
            return {**result, "reason_key": "no_udder_found"}
        box = found["udder"][1]
        udder = crop(img, box)
        if min(udder.size) < MIN_UDDER_PX:
            return {**result, "reason_key": "udder_too_far", "box": box}

        # average of the udder crop and the whole photo: two views of the same udder, so one bad
        # crop can't swing the verdict (end-to-end test: reports/udder_pipeline_evaluation*.json)
        p = (self._classify(udder) + self._classify(img)) / 2
        confidence = max(p, 1 - p)
        level = next(name for name, lo in BANDS if confidence >= lo)
        label = "affected" if p >= 0.5 else "healthy"
        advice = ("advice.see_vet" if label == "affected"
                  else "advice.keep_watching" if level == "high"
                  else "advice.check_again_or_vet")
        return dict(status="ok", reason_key=None, label=label, probability=round(p, 3),
                    confidence_level=level, advice_key=advice,
                    disclaimer_key="disclaimer.screening_aid", box=[round(v, 4) for v in box])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python src/predict.py photo.jpg [more photos...]")
    screener = UdderScreener()
    for path in sys.argv[1:]:
        print(path)
        print(json.dumps(screener.predict(path), indent=2))
