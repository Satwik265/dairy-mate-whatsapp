"""
teat_end_crop.py

Crops a teat close-up to its teat end using the teat-end detector (models/teat_end_detector.onnx,
trained by scripts/teat_end_detector.py). The classifier then sees the ring instead of the whole teat.

The crop is a square around the most confident teat-end box, padded so some teat stays in view.
If nothing is found (confidence below MIN_CONF) the whole photo is used, so no photo is ever dropped.

    python scripts/teat_end_crop.py      # crop every processed_v2 photo -> data/processed_v2_teatend/
                                         # + curation/teat_end_crops_v2.csv (box, conf, fallback)
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TEAT = Path(__file__).resolve().parents[1]
MODEL = TEAT / "models" / "teat_end_detector.onnx"
SRC = TEAT / "data" / "processed_v2"
DST = TEAT / "data" / "processed_v2_teatend"
LOG = TEAT / "curation" / "teat_end_crops_v2.csv"

DET_SIZE = 512
MIN_CONF = 0.10       # below this: use the whole photo
PAD = 1.6             # crop side = PAD x the longer box side
MIN_SIDE = 0.25       # ...but at least this fraction of the photo's shorter side


class TeatEndCropper:
    def __init__(self, model_path=MODEL):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, img):
        """Most confident teat end -> (conf, [x0, y0, x1, y1] as fractions of the photo)."""
        import cv2
        w, h = img.size
        scale = DET_SIZE / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        # same letterbox as Ultralytics (and common/gate.py)
        resized = cv2.resize(np.asarray(img), (nw, nh), interpolation=cv2.INTER_LINEAR)
        left, top = round((DET_SIZE - nw) / 2 - 0.1), round((DET_SIZE - nh) / 2 - 0.1)
        canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0]          # (4 + 1, anchors)
        i = int(np.argmax(out[4]))
        cx, cy, bw, bh = out[:4, i]
        box = [(cx - bw / 2 - left) / scale / w, (cy - bh / 2 - top) / scale / h,
               (cx + bw / 2 - left) / scale / w, (cy + bh / 2 - top) / scale / h]
        return float(out[4, i]), [float(np.clip(v, 0, 1)) for v in box]

    def crop(self, img):
        """-> (cropped image, conf, box or None if the whole photo was kept)."""
        conf, box = self.detect(img)
        if conf < MIN_CONF:
            return img, conf, None
        w, h = img.size
        x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        side = min(max(PAD * max(x1 - x0, y1 - y0), MIN_SIDE * min(w, h)), min(w, h))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left = int(round(min(max(cx - side / 2, 0), w - side)))      # shift inside the photo
        top = int(round(min(max(cy - side / 2, 0), h - side)))
        side = int(round(side))
        return img.crop((left, top, left + side, top + side)), conf, box


def main():
    import shutil
    cropper = TeatEndCropper()
    if DST.exists():
        shutil.rmtree(DST)
    rows = []
    for path in sorted(SRC.glob("*/*/*.jpg")):
        split, label = path.parts[-3], path.parts[-2]
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        out, conf, box = cropper.crop(img)
        dst = DST / split / label / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, quality=95)
        rows.append(dict(split=split, label=label, file=path.name, conf=round(conf, 4),
                         fallback=int(box is None), crop_frac=round(out.size[0] * out.size[1] / (img.size[0] * img.size[1]), 3),
                         **({k: round(v, 4) for k, v in zip(("x0", "y0", "x1", "y1"), box)} if box else
                            dict(x0="", y0="", x1="", y1=""))))
    with open(LOG, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        fb = sum(r["fallback"] for r in rs)
        print(f"{split:5s} {len(rs):4d} photos | whole-photo fallback {fb} | "
              f"median crop {np.median([r['crop_frac'] for r in rs]):.2f} of the photo")
    print("saved ->", DST, "and", LOG)


if __name__ == "__main__":
    main()
