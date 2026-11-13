"""
Teat-end detector (YOLO11n): finds the teat-end tip/ring in a teat close-up, so the teat classifier
can score a tighter crop than the shared gate box (v2 missed ~55% of rough rings on whole close-ups).

Trained on the 304-photo hand-labelled sample (curation/teat_end_sample.csv + teat_end_labels.csv).
The sample keeps the classifier's grouped split, so a photo in the classifier's test set is also
in the detector's test set. Photos labelled "unusable" are left out (they are not clean negatives).

Runs in the YOLO venv (PyTorch + Ultralytics), separate from the TensorFlow venv:
    ..\\venv_yolo\\Scripts\\python scripts/teat_end_detector.py --stage prepare   # YOLO dataset from the labels
    ..\\venv_yolo\\Scripts\\python scripts/teat_end_detector.py --stage train     # fine-tune YOLO11n, test mAP
    ..\\venv_yolo\\Scripts\\python scripts/teat_end_detector.py --stage export    # ONNX for deployment

Run one training job at a time on this machine (16 GB RAM, 4 GB GPU).
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

TEAT = Path(__file__).resolve().parents[1]
SAMPLE = TEAT / "curation" / "teat_end_sample.csv"
LABELS = TEAT / "curation" / "teat_end_labels.csv"
SRC = TEAT / "data" / "processed_v2"          # split/label/id.jpg, same files the labelling page showed
DET = TEAT / "data" / "teat_end_detector"
RUNS = TEAT / "runs"
RUN_NAME = "teat_end_detector"
NAMES = ["teat_end"]
IMGSZ = 512
SEED = 42
EPOCHS = 100


def stage_prepare():
    sample = {r["id"]: r for r in csv.DictReader(open(SAMPLE, newline=""))}
    labels = list(csv.DictReader(open(LABELS, newline="")))
    if DET.exists():
        shutil.rmtree(DET)                    # always rebuild from scratch, no stale files
    for split in ("train", "val", "test"):
        (DET / "images" / split).mkdir(parents=True)
        (DET / "labels" / split).mkdir(parents=True)

    counts, dropped = Counter(), Counter()
    for row in labels:
        meta = sample.get(row["id"])
        if meta is None:
            dropped["not in sample"] += 1
            continue
        if row["status"] != "box":
            dropped[row["status"]] += 1
            continue
        src = SRC / meta["split"] / meta["label"] / f"{row['id']}.jpg"
        if not src.exists():
            dropped["missing image"] += 1
            continue
        out = []
        for box in row["boxes"].split(";"):
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in box.split())
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            out.append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        if not out:
            dropped["empty boxes"] += 1
            continue
        shutil.copy2(src, DET / "images" / meta["split"] / src.name)
        (DET / "labels" / meta["split"] / f"{row['id']}.txt").write_text("\n".join(out) + "\n")
        counts[(meta["split"], "images")] += 1
        counts[(meta["split"], "boxes")] += len(out)

    (DET / "data.yaml").write_text(
        f"path: {DET.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print("dropped:", dict(dropped))
    for split in ("train", "val", "test"):
        print(f"  {split:5s} {counts[(split, 'images')]:4d} images | {counts[(split, 'boxes')]:4d} boxes")


def stage_train():
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")
    model.train(data=str(DET / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS, patience=25, batch=16,
                device=device, workers=2, seed=SEED, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, plots=True, verbose=False)
    metrics = YOLO(RUNS / RUN_NAME / "weights" / "best.pt").val(
        data=str(DET / "data.yaml"), split="test", imgsz=IMGSZ, device=device, verbose=False)
    print(f"TEST mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f} | "
          f"P {metrics.box.mp:.3f} | R {metrics.box.mr:.3f}")


def stage_export():
    from ultralytics import YOLO
    path = YOLO(RUNS / RUN_NAME / "weights" / "best.pt").export(format="onnx", imgsz=IMGSZ,
                                                                 simplify=False, dynamic=False)
    dst = TEAT / "models" / f"{RUN_NAME}.onnx"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(path, dst)
    print("exported", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "train", "export"], required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    EPOCHS = args.epochs
    {"prepare": stage_prepare, "train": stage_train, "export": stage_export}[args.stage]()
