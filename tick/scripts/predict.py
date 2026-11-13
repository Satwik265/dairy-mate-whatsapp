"""
Tick screening: one photo in, one plain result out. Same result shape as teat/scripts/predict.py.

    python tick/scripts/predict.py path/to/photo.jpg [more photos...]

Flow:
    1. light quality check (common/gate.py: unreadable, tiny or almost black photos -> retake)
    2. the photo is stretched to 640x640, the same geometry as the Griffith training images
    3. lenient animal check: COCO YOLO11n (models/animal_detector.onnx, never trained on tick data) looks for an
       animal (dog...giraffe classes; cattle are often called horse/sheep/bear by COCO). Found -> crop to the most
       confident animal box + 10% padding. Not found -> score the whole photo anyway (close-ups of tick-covered
       skin usually have no recognisable animal in them) and say so in the result (animal_found=False).
    4. the crop is tiled into 96x96 patches (stride 48); the patch model named in the card (tick_patch_B.onnx)
       scores each patch ("is there a tick in this patch?")
    5. image score = aggregation rule from models/tick_model_card.json (picked on validation videos only),
       calibrated with Platt scaling fitted on the validation split -> P(ticks visible)
    6. verdict: affected ("Ticks detected") / healthy ("No ticks seen in this photo"). No "not sure" verdict: low
       confidence changes headline_key ("Possible ticks" / "no ticks seen, low confidence") and advice_key
       (check again by hand / see a vet). A healthy verdict is never "high" confidence.

The shared udder/teat photo gate is NOT used to reject or redirect: it would send 78% of the Griffith test tick photos
to "retake" and 34% of web tick photos to Teat Check (reports/tick_gate_routing.json). Only the quality check is used.

Honest status: see tick/README.md. Trained and validated only on one Roboflow set (a few Hereford calves in one
crush pen, ticks ~4x6 px) plus presumed tick-free web/CCTV cows. Not validated on field phone photos; on close-up
web photos of infested cattle it catches only 4 of 34.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TICK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TICK.parent))
from common.gate import open_photo, quality_problem  # noqa: E402
from common.ort_session import ort_session  # noqa: E402

MODELS = TICK / "models"
FRAME = 640            # Roboflow v1 stretched every image to 640x640; ticks are learned at that scale
PATCH = 96
STRIDE = 48
CROP_PAD = 0.10
DET_SIZE = 640
ANIMAL_CLASSES = range(16, 24)   # COCO: dog, horse, sheep, cow, elephant, bear, zebra, giraffe
MIN_ANIMAL_CONF = 0.25
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def to_frame(img):
    """Stretch to FRAME x FRAME like the training data (Roboflow 'Stretch to 640x640')."""
    return img.resize((FRAME, FRAME), Image.BILINEAR)


def pad_box(box, w, h, pad=CROP_PAD):
    """box in pixels (x0, y0, x1, y1) -> padded integer box clipped to the image."""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)), int(min(h, y1 + py)))


def grid(w, h, patch=PATCH, stride=STRIDE):
    """Sliding windows covering the whole (w, h) image; last row/column pulled inward, no padding."""
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    if xs[-1] != max(w - patch, 0):
        xs.append(max(w - patch, 0))
    if ys[-1] != max(h - patch, 0):
        ys.append(max(h - patch, 0))
    return [(x, y, x + patch, y + patch) for y in ys for x in xs]


def ensure_min_size(img, size=PATCH):
    """Crops smaller than one patch are padded with their mean colour (rare: tiny animal boxes)."""
    w, h = img.size
    if w >= size and h >= size:
        return img
    canvas = Image.new("RGB", (max(w, size), max(h, size)), tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(0)))
    canvas.paste(img, (0, 0))
    return canvas


def aggregate(probs, rule):
    """Image score from patch probabilities. rule = {"name": "topk_mean", "k": 3} | {"name": "count_above", "t": 0.9}."""
    p = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if rule["name"] == "max":
        return float(p[0])
    if rule["name"] == "topk_mean":
        return float(p[: rule["k"]].mean())
    if rule["name"] == "count_above":
        return float((p >= rule["t"]).sum())
    raise ValueError(rule)


class AnimalDetector:
    """COCO YOLO11n exported to ONNX; same letterbox as Ultralytics (see common/gate.py)."""

    def __init__(self, path=MODELS / "animal_detector.onnx"):
        self.session = ort_session(path)
        self.input_name = self.session.get_inputs()[0].name

    def best_animal(self, img):
        """-> (conf, coco_class, (x0, y0, x1, y1) in pixels of img) for the most confident animal, or None."""
        import cv2
        w, h = img.size
        scale = DET_SIZE / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        resized = cv2.resize(np.asarray(img), (nw, nh), interpolation=cv2.INTER_LINEAR)
        left, top = round((DET_SIZE - nw) / 2 - 0.1), round((DET_SIZE - nh) / 2 - 0.1)
        canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0]          # (4 + 80, anchors)
        scores = out[4:][list(ANIMAL_CLASSES)]
        cls_i, anchor = np.unravel_index(int(np.argmax(scores)), scores.shape)
        conf = float(scores[cls_i, anchor])
        if conf < MIN_ANIMAL_CONF:
            return None
        cx, cy, bw, bh = out[:4, anchor]
        box = ((cx - bw / 2 - left) / scale, (cy - bh / 2 - top) / scale,
               (cx + bw / 2 - left) / scale, (cy + bh / 2 - top) / scale)
        return conf, int(list(ANIMAL_CLASSES)[cls_i]), tuple(float(np.clip(v, 0, lim)) for v, lim in zip(box, (w, h, w, h)))


class PatchModel:
    def __init__(self, path=MODELS / "tick_patch_classifier.onnx", threads=None):
        self.session = ort_session(path, threads)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, patches, batch=None):
        """patches: uint8 array (N, PATCH, PATCH, 3) RGB -> P(tick) per patch. Smaller batches use less memory and give
        the same scores (DM_TICK_BATCH=16 on the 512 MB website server)."""
        batch = batch or int(os.environ.get("DM_TICK_BATCH", "64"))
        out = []
        for i in range(0, len(patches), batch):
            x = ((patches[i:i + batch].astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2)
            logits = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0].ravel()
            out.append(1 / (1 + np.exp(-logits)))
        return np.concatenate(out) if out else np.zeros(0)


def patch_map(img_frame, crop_box, patch_model):
    """Score every grid patch of the crop. Returns (windows in frame pixels, probabilities)."""
    x0, y0, x1, y1 = crop_box
    crop = ensure_min_size(img_frame.crop(crop_box))
    arr = np.asarray(crop)
    wins = grid(*crop.size)
    patches = np.stack([arr[b:d, a:c] for a, b, c, d in wins])
    probs = patch_model(patches)
    return [(a + x0, b + y0, c + x0, d + y0) for a, b, c, d in wins], probs


class TickScreener:
    """Load the models once, then call .predict() for every photo."""

    def __init__(self, models_dir=MODELS):
        models_dir = Path(models_dir)
        card = json.load(open(models_dir / "tick_model_card.json"))
        self.card = card
        self.detector = AnimalDetector(models_dir / card["animal_detector"])
        self.patches = PatchModel(models_dir / card["model"])
        self.rule = card["aggregation"]
        self.A, self.B = card["calibration"]["A"], card["calibration"]["B"]
        self.bands = [(name, lo) for name, lo in card["confidence_bands"]]
        self.release_status = card.get("release_status", "beta")

    def locate(self, img):
        """-> (frame image, crop box in frame pixels, animal info or None)."""
        frame = to_frame(img)
        found = self.detector.best_animal(frame)
        if found is None:
            return frame, (0, 0, FRAME, FRAME), None
        return frame, pad_box(found[2], FRAME, FRAME), found

    def raw_score(self, img):
        frame, box, found = self.locate(img)
        wins, probs = patch_map(frame, box, self.patches)
        return aggregate(probs, self.rule), wins, probs, box, found

    def calibrate(self, score):
        if self.card["score_transform"] == "log1p":
            s = float(np.log1p(score))
        else:
            q = min(max(score, 1e-6), 1 - 1e-6)
            s = float(np.log(q / (1 - q)))
        return float(1 / (1 + np.exp(-(self.A * s + self.B))))

    def predict(self, photo):
        result = dict(status="retake", release_status=self.release_status, reason_key=None, label=None, headline_key=None, probability=None,
                      confidence_level=None, advice_key="advice.retake_photo",
                      disclaimer_key="disclaimer.screening_aid", animal_found=None, box=None, tick_patches=None)
        try:
            img = open_photo(photo)
        except Exception:
            return {**result, "reason_key": "photo_unreadable"}
        problem = quality_problem(img)
        if problem:
            return {**result, "reason_key": problem}

        score, wins, probs, box, found = self.raw_score(img)
        p = self.calibrate(score)
        confidence = max(p, 1 - p)
        level = next(name for name, lo in self.bands if confidence >= lo)
        label = "affected" if p >= 0.5 else "healthy"
        if level == "high" and (found is None or label == "healthy"):
            # never claim high confidence without a recognised animal, nor for "no ticks": the model misses ticks
            # on close-up photos from other farms with p near 0 (see the model card)
            level = "medium"
        # Two verdicts only. Low confidence changes the wording and the advice, never the verdict.
        headline = (("tick.result.ticks_detected" if level != "low" else "tick.result.possible_ticks")
                    if label == "affected" else
                    ("tick.result.no_ticks_seen" if level != "low" else "tick.result.no_ticks_seen_low_confidence"))
        # "no ticks seen" always comes with "check by hand (ears, neck, dewlap, udder, tail base); vet if unsure"
        advice = "advice.see_vet" if label == "affected" else "advice.check_again_or_vet"
        hot = [[round(v / FRAME, 4) for v in w] for w, q in zip(wins, probs) if q >= self.card["patch_threshold"]]
        return dict(status="ok", release_status=self.release_status,
                    reason_key=None if found else "no_animal_found_scored_whole_photo",
                    label=label, headline_key=headline, probability=round(p, 3), confidence_level=level,
                    advice_key=advice,
                    disclaimer_key="disclaimer.screening_aid", animal_found=found is not None,
                    box=[round(v / FRAME, 4) for v in box], tick_patches=hot)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tick/scripts/predict.py photo.jpg [more photos...]")
    screener = TickScreener()
    for path in sys.argv[1:]:
        print(path)
        print(json.dumps(screener.predict(path), indent=2))
