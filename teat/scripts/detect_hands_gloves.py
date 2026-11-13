"""
detect_hands_gloves.py

Scans a folder of teat/udder images and flags which ones contain a human
hand (bare or gloved), using MediaPipe's HandLandmarker — a shape-based
hand detector (not color-based), which is what actually works here.

WHY NOT COLOR-BASED DETECTION:
Tested against your real images first. Two dead ends:
  - Skin-tone detection fires on 70-90%+ of most teat images, because teat
    tissue and human skin are colorimetrically very similar. Can't tell hand
    from teat by color alone.
  - Glove color (blue/purple nitrile) is too inconsistent across lighting —
    some gloved shots read as pale/washed-out rather than clearly blue.
Shape-based hand detection (this script) sidesteps both problems.

*** IMPORTANT: RUN THIS IN A SEPARATE VIRTUAL ENVIRONMENT ***
MediaPipe's dependencies conflict with TensorFlow's `protobuf` version
requirement — installing mediapipe into your teat_pipeline venv will BREAK
your TensorFlow install. Use a second, separate venv just for this script:

    python -m venv venv_handdetect
    venv_handdetect\\Scripts\\Activate.ps1        (Windows)
    pip install mediapipe pillow opencv-python-headless numpy

Then run this script from that venv, pointing at your image folders. It does
NOT need TensorFlow at all, so keep it fully separate from teat_pipeline's venv.

Run:
    python detect_hands_gloves.py --input_dirs data/raw_healthy_sources data/raw_mastitis_sources

Output:
    reports/hand_detection_report.csv        -- per-image results
    reports/hand_detection_flagged_grid.png  -- contact sheet of flagged images
"""

import os
import csv
import argparse
import urllib.request
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_model(model_path):
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading hand detection model to {model_path} (one-time, ~10MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the hand detection model automatically ({e}).\n"
            f"Manually download it from:\n  {MODEL_URL}\n"
            f"and save it as: {model_path}\n"
            f"then re-run this script."
        )
    return model_path


def build_detector(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=4,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(detector, image_path):
    """Returns (num_hands, list_of_confidences, list_of_bbox_pixel_coords)."""
    import mediapipe as mp

    mp_image = mp.Image.create_from_file(image_path)
    result = detector.detect(mp_image)

    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
    confidences = []
    bboxes = []

    if num_hands > 0:
        w, h = mp_image.width, mp_image.height
        for i, landmarks in enumerate(result.hand_landmarks):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            bbox = (
                int(min(xs) * w), int(min(ys) * h),
                int(max(xs) * w), int(max(ys) * h),
            )
            bboxes.append(bbox)
            # handedness score doubles as a rough per-hand confidence
            if result.handedness and i < len(result.handedness):
                confidences.append(result.handedness[i][0].score)
            else:
                confidences.append(None)

    return num_hands, confidences, bboxes


def gather_images(input_dirs):
    records = []
    for d in input_dirs:
        if not os.path.isdir(d):
            print(f"Warning: {d} is not a directory, skipping.")
            continue
        for root, _, files in os.walk(d):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTS:
                    records.append(os.path.join(root, fname))
    return records


def draw_bbox_overlay(image_path, bboxes, out_path):
    im = Image.open(image_path).convert("RGB")
    arr = np.array(im)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr)
    for (x0, y0, x1, y1) in bboxes:
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_contact_sheet(flagged_paths, out_path, max_items=40):
    items = flagged_paths[:max_items]
    if not items:
        print("No flagged images to build a contact sheet from.")
        return
    cols = 5
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax, path in zip(axes, items):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(os.path.basename(path), fontsize=6)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True,
                         help="One or more folders to scan recursively for images.")
    parser.add_argument("--model_path", default="hand_landmarker.task")
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--overlay_dir", default="reports/hand_overlays",
                         help="Where to save individual bbox overlays for flagged images.")
    args = parser.parse_args()

    model_path = ensure_model(args.model_path)
    detector = build_detector(model_path)

    image_paths = gather_images(args.input_dirs)
    print(f"Scanning {len(image_paths)} images across: {args.input_dirs}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.overlay_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "hand_detection_report.csv")
    flagged_paths = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "num_hands_detected", "max_confidence", "flagged"])

        for i, path in enumerate(image_paths):
            try:
                num_hands, confidences, bboxes = detect_hands(detector, path)
            except Exception as e:
                print(f"  ERROR on {path}: {e}")
                writer.writerow([path, "ERROR", "", "FALSE"])
                continue

            max_conf = max([c for c in confidences if c is not None], default=0.0)
            flagged = num_hands > 0
            writer.writerow([path, num_hands, f"{max_conf:.3f}", flagged])

            if flagged:
                flagged_paths.append(path)
                overlay_name = f"{i:04d}_{os.path.basename(path)}.png"
                draw_bbox_overlay(path, bboxes, os.path.join(args.overlay_dir, overlay_name))

            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(image_paths)} scanned, {len(flagged_paths)} flagged so far")

    print(f"\nDone. {len(flagged_paths)} / {len(image_paths)} images flagged as containing a hand.")
    print(f"Full results: {csv_path}")
    print(f"Per-image overlays: {args.overlay_dir}/")

    grid_path = os.path.join(args.out_dir, "hand_detection_flagged_grid.png")
    build_contact_sheet(flagged_paths, grid_path)
    print(f"Summary grid (first 40 flagged): {grid_path}")


if __name__ == "__main__":
    main()
