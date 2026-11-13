"""
Step 2 - Segmentation: isolate the teat/animal from background clutter
using GrabCut (classical, unsupervised).
"""
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "data" / "processed" / "segmented"

MAX_DIM = 320


def grabcut_segment(img, margin_frac=0.04, iters=3):
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else img
    sh, sw = small.shape[:2]

    mx, my = max(1, int(sw * margin_frac)), max(1, int(sh * margin_frac))
    rect = (mx, my, sw - 2 * mx, sh - 2 * my)

    mask = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return img, False

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    if mask2.sum() < 0.05 * sh * sw:
        return img, False

    full_mask = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)
    result = img * full_mask[:, :, np.newaxis]
    return result, True


def main():
    manifest_path = SPLITS_DIR / "annotations_full.csv"
    rows = list(csv.DictReader(open(manifest_path)))

    success, fallback, skipped = 0, 0, 0
    for row in tqdm(rows, desc="Segmenting"):
        src = Path(row["filepath"])
        cls = row["class_name"]
        out_dir = OUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = Path(src.stem + ".jpg")
        out_path = out_dir / out_name
        if out_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"Could not read {src}, skipping.")
            continue

        segmented, ok = grabcut_segment(img)
        success += ok
        fallback += (not ok)
        cv2.imwrite(str(out_path), segmented)

    print(f"\nDone. GrabCut succeeded on {success} images, fell back to original on {fallback}, "
          f"skipped {skipped} already done.")


if __name__ == "__main__":
    main()
