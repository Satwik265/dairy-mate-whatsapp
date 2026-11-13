"""
Udder detector (YOLO11n): finds the udder in a whole photo so the classifier can score a crop.
Also detects teat close-ups, so the app can say "this looks like a teat photo, try Teat Check".

Runs in its own venv (PyTorch + Ultralytics), separate from the TensorFlow venv:
    ..\\venv_yolo\\Scripts\\python src/detector.py --stage prepare   # YOLO dataset from Roboflow v16 boxes
    ..\\venv_yolo\\Scripts\\python src/detector.py --stage train     # fine-tune YOLO11n
    ..\\venv_yolo\\Scripts\\python src/detector.py --stage detect    # detections on test photos -> CSV
    ..\\venv_yolo\\Scripts\\python src/detector.py --stage export    # ONNX for deployment

Classes:
    0 udder          whole-udder view (healthy or mastitis; the classifier decides that)
    1 teat_closeup   close-up of one or more teats (belongs to the teat model)

The split reuses the classifier's grouped split (rf16_clean_manifest.csv), so a source photo
in the classifier's test set is also in the detector's test set.
"""

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_roboflow_v16"
CUR = ROOT / "data" / "curation"
RUNS = ROOT / "runs"
TEAT = ROOT.parent / "teat"

# --version v1: udder + Roboflow teat close-ups (udder app only)
# --version v2: v1 + all teat v2 close-ups (SCTL teat ends, web lesions) so the same detector can
#               gate BOTH apps: udder photo -> Udder Check, teat close-up -> Teat Check, nothing -> retake
VERSION = "v1"
DET = ROOT / "data" / "detector"
RUN_NAME = "udder_detector"


def set_version(v):
    global VERSION, DET, RUN_NAME
    VERSION = v
    DET = ROOT / "data" / ("detector" if v == "v1" else f"detector_{v}")
    RUN_NAME = "udder_detector" if v == "v1" else f"udder_detector_{v}"

UDDER_CLASS_IDS = {4, 6}          # Healthy_udder, Mastitis_infected_udder in the Roboflow export
NAMES = ["udder", "teat_closeup"]
IMGSZ = 512
SEED = 42
EPOCHS = 80                 # v1 used 80; override with --epochs


def fallback_split(group):
    """Stable 70/15/15 split by hashing the group id (for images not in the classifier manifest)."""
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def yolo_box(parts):
    xs = list(map(float, parts[1:]))
    if len(xs) == 4:
        return xs
    x0, x1, y0, y1 = min(xs[0::2]), max(xs[0::2]), min(xs[1::2]), max(xs[1::2])
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def stage_prepare():
    boxes = list(csv.DictReader(open(CUR / "rf16_boxes.csv")))
    with open(CUR / "view_tags.csv") as f:
        view = {r["box_id"]: r["view"] for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    manifest = {r["box_id"]: r for r in csv.DictReader(open(CUR / "rf16_clean_manifest.csv"))}

    per_image = defaultdict(list)
    for b in boxes:
        per_image[(b["rf_split"], b["image"])].append(b)

    if DET.exists():
        shutil.rmtree(DET)
    counts, dropped = defaultdict(int), defaultdict(int)
    for (rf_split, image), bs in per_image.items():
        if any(b["thermal"] == "1" for b in bs):
            dropped["thermal"] += 1
            continue
        if any(view.get(b["box_id"]) == "reject" for b in bs):
            dropped["rejected_in_triage"] += 1
            continue
        in_manifest = [manifest[b["box_id"]] for b in bs if b["box_id"] in manifest]
        split = in_manifest[0]["split"] if in_manifest else fallback_split(bs[0]["group"])

        label_lines = (RAW / rf_split / "labels" / (Path(image).stem + ".txt")).read_text().splitlines()
        udder_lines = [l.split() for l in label_lines if l.split() and int(l.split()[0]) in UDDER_CLASS_IDS]
        out = []
        for b in bs:
            cls = 1 if view.get(b["box_id"]) == "teat_closeup" else 0
            out.append(f"{cls} " + " ".join(f"{v:.6f}" for v in yolo_box(udder_lines[int(b["box_idx"])])))
            counts[(split, NAMES[cls])] += 1

        (DET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DET / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAW / rf_split / "images" / image, DET / "images" / split / image)
        (DET / "labels" / split / (Path(image).stem + ".txt")).write_text("\n".join(out) + "\n")

    (DET / "data.yaml").write_text(
        f"path: {DET.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print("dropped images:", dict(dropped))
    for split in ("train", "val", "test"):
        n_img = len(list((DET / "images" / split).glob("*")))
        print(f"  {split:5s} {n_img:4d} images | " +
              " | ".join(f"{n} boxes {counts[(split, n)]}" for n in NAMES))


def stage_train():
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")
    model.train(data=str(DET / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS, patience=20, batch=16,
                device=device, workers=2, seed=SEED, project=str(RUNS), name=RUN_NAME,
                exist_ok=True, plots=True, verbose=False)
    metrics = YOLO(RUNS / RUN_NAME / "weights" / "best.pt").val(
        data=str(DET / "data.yaml"), split="test", imgsz=IMGSZ, device=device, verbose=False)
    print("TEST mAP50-95 per class:", {NAMES[i]: round(float(v), 3)
                                       for i, v in enumerate(metrics.box.maps)})
    print(f"TEST mAP50 (all) {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f}")


def stage_detect():
    """Detections on every detector test image -> CSV (consumed by evaluate_pipeline.py)."""
    import torch
    from ultralytics import YOLO
    model = YOLO(RUNS / RUN_NAME / "weights" / "best.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    # held-out test photos + optional outside photos in data/detector/external/<set name>/
    sources = [("test", p) for p in sorted((DET / "images" / "test").glob("*"))]
    for d in sorted((DET / "external").glob("*")) if (DET / "external").exists() else []:
        sources += [(f"external/{d.name}", p) for p in sorted(d.glob("*"))]
    rows = []
    from PIL import Image, ImageOps
    for source, img in sources:
        # open with PIL so .jfif and EXIF-rotated phone photos work (Ultralytics checks extensions)
        pil = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
        r = model.predict(pil, imgsz=IMGSZ, conf=0.05, device=device, verbose=False)[0]
        if len(r.boxes) == 0:
            rows.append(dict(source=source, image=img.name, cls="", conf=0, x0=0, y0=0, x1=0, y1=0))
        for b in r.boxes:
            x0, y0, x1, y1 = b.xyxyn[0].tolist()
            rows.append(dict(source=source, image=img.name, cls=NAMES[int(b.cls)],
                             conf=round(float(b.conf), 4), x0=x0, y0=y0, x1=x1, y1=y1))
    out = DET / "test_detections.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} detections -> {out}")


def stage_export():
    from ultralytics import YOLO
    path = YOLO(RUNS / RUN_NAME / "weights" / "best.pt").export(format="onnx", imgsz=IMGSZ,
                                                                         simplify=True, dynamic=False)
    dst = ROOT / "models" / f"{RUN_NAME}.onnx"
    shutil.copy2(path, dst)
    print(f"saved -> {dst} ({dst.stat().st_size / 2**20:.1f} MB)")


def stage_add_teat():
    """v2 only: add every teat v2 close-up (teat/curation/teat_v2_manifest.csv) as class 1.
    SCTL photos are already crops of one teat end, so the box is (almost) the whole photo.
    Images that came from the Roboflow v16 export are already in the set with real boxes -> skipped.
    The teat v2 split is kept, so teat test photos stay unseen by the detector too."""
    assert VERSION != "v1", "add_teat is for --version v2"
    rows = list(csv.DictReader(open(TEAT / "curation" / "teat_v2_manifest.csv")))
    added = defaultdict(int)
    for r in rows:
        if r["origin"].startswith("rf16:"):
            continue
        src = TEAT / "data" / "processed_v2" / r["split"] / r["label"] / f"{r['id']}.jpg"
        name = f"teat_{r['id']}.jpg"
        shutil.copy2(src, DET / "images" / r["split"] / name)
        (DET / "labels" / r["split"] / f"teat_{r['id']}.txt").write_text("1 0.5 0.5 0.96 0.96\n")
        added[(r["split"], r["source"])] += 1
    print("teat close-ups added:", dict(added))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "add_teat", "train", "detect", "export"], required=True)
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    set_version(args.version)
    EPOCHS = args.epochs
    dict(prepare=stage_prepare, add_teat=stage_add_teat, train=stage_train, detect=stage_detect,
         export=stage_export)[args.stage]()
