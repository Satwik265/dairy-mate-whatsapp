"""
Shared photo gate for all DairyMate checks.

One YOLO11n detector (common/models/photo_gate.onnx, trained as udder/runs/udder_detector_v2) looks at the
whole photo and finds a whole udder and/or a teat close-up. Each app uses it to decide:
    score it            (the photo shows what this app checks)
    redirect            (it's the other app's photo: "try Udder Check" / "try Teat Check")
    retake              (unreadable, tiny, almost black, or nothing found)

Routing measured on held-out photos (udder/reports/gate_evaluation.json):
    teat close-ups -> teat 100% (298/298), whole udders -> udder 96%, tick/cow-skin photos -> retake only 46%.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

GATE_MODEL = Path(__file__).resolve().parent / "models" / "photo_gate.onnx"

DET_SIZE = 512
CLASSES = ["udder", "teat_closeup"]
MIN_CONF = 0.25            # below this the gate "didn't find" the object
MIN_PHOTO_PX = 160         # shorter side of the whole photo
MIN_BRIGHTNESS = 20        # mean grey level 0-255; only near-black photos are rejected
CROP_PAD = 0.05


def open_photo(photo):
    """Accept a path, raw bytes or a PIL image; apply EXIF rotation (phone photos)."""
    if isinstance(photo, (bytes, bytearray)):
        photo = io.BytesIO(photo)
    img = photo if isinstance(photo, Image.Image) else Image.open(photo)
    return ImageOps.exif_transpose(img).convert("RGB")


def quality_problem(img):
    """Only obvious problems; returns a reason key or None."""
    if min(img.size) < MIN_PHOTO_PX:
        return "too_small"
    if np.asarray(img.convert("L")).mean() < MIN_BRIGHTNESS:
        return "too_dark"
    return None


def crop(img, box, pad=CROP_PAD):
    w, h = img.size
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((int(max(0, (x0 - px) * w)), int(max(0, (y0 - py) * h)),
                     int(min(w, (x1 + px) * w)), int(min(h, (y1 + py) * h))))


class PhotoGate:
    def __init__(self, model_path=GATE_MODEL):
        from common.ort_session import ort_session
        self.session = ort_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, img):
        """Best confidence per class -> {class: (conf, [x0, y0, x1, y1] as fractions of the photo)}."""
        import cv2
        w, h = img.size
        scale = DET_SIZE / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        # same letterbox as Ultralytics: cv2 linear resize, grey (114) padding split around the image
        resized = cv2.resize(np.asarray(img), (nw, nh), interpolation=cv2.INTER_LINEAR)
        left, top = round((DET_SIZE - nw) / 2 - 0.1), round((DET_SIZE - nh) / 2 - 0.1)
        canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0]          # (4 + n_classes, anchors)
        best = {}
        for c, name in enumerate(CLASSES):
            scores = out[4 + c]
            i = int(np.argmax(scores))
            if scores[i] < MIN_CONF:
                continue
            cx, cy, bw, bh = out[:4, i]
            box = [(cx - bw / 2 - left) / scale / w, (cy - bh / 2 - top) / scale / h,
                   (cx + bw / 2 - left) / scale / w, (cy + bh / 2 - top) / scale / h]
            best[name] = (float(scores[i]), [float(np.clip(v, 0, 1)) for v in box])
        return best

    @staticmethod
    def route(found):
        """'udder' | 'teat' | None, using the more confident of the two detections."""
        udder, teat = found.get("udder"), found.get("teat_closeup")
        if teat and (not udder or teat[0] > udder[0]):
            return "teat"
        if udder:
            return "udder"
        return None
