"""
build_dataset_v2.py

Teat dataset v2: teat close-ups only, honest labels, split by source photo / photo moment.

Why v2 (see ../../docs/audit_notes.md):
  - v1 split augmented copies of the same photo across train/test (inflated ~97% accuracy)
  - v1 healthy = one photo style (SCTL Score 1), problem = everything else -> the model learned photo style
  - ~40% of v1 "mastitis" images were whole-udder photos, which belong to the udder model

Labels:
  healthy  SCTL teat-end Score 1 (no ring) + Score 2 (smooth ring) + healthy teat close-ups from other sources
  problem  SCTL Score 3 (rough ring) + Score 4 (very rough) + teat-lesion close-ups (Roboflow / teat websites)

Stages:
  ..\\..\\venv_yolo\\Scripts\\python scripts/build_dataset_v2.py --stage views   # tag v1 positives as teat close-up / whole udder
  ..\\..\\venv_handdetect\\Scripts\\python scripts/build_dataset_v2.py --stage hands  # MediaPipe hand check on new images
  ..\\..\\venv\\Scripts\\python scripts/build_dataset_v2.py --stage build         # collect, group, dedupe, split, copy

Outputs:
  curation/view_auto.csv          udder-detector view tags for v1 positives
  curation/hand_flags.csv         images with a detected hand (excluded, like v1's quarantine)
  curation/udder_v2_candidates.csv  whole-udder photos moved out of the teat set (for a future udder v2)
  curation/teat_v2_manifest.csv   every kept image: id, label, source, score, group, split, origin
  data/processed_v2/{train,val,test}/{healthy,problem}/<id>.jpg
"""

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TEAT = Path(__file__).resolve().parents[1]
REPO = TEAT.parent
DATA = TEAT / "data"
CUR = TEAT / "curation"
OUT = DATA / "processed_v2"

UDDER_DETECTOR = REPO / "udder" / "runs" / "udder_detector" / "weights" / "best.pt"
RF16 = REPO / "udder" / "data" / "raw_roboflow_v16"
RF16_CUR = REPO / "udder" / "data" / "curation"
HAND_MODEL = TEAT / "hand_landmarker.task"

SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
CROP_PAD = 0.05
NEAR_DUP_BITS = 4          # dHash distance (of 64) at which two images count as the same photo


# ---------- grouping helpers ----------

def roboflow_stem(name):
    """'abc_jpg.rf.<hash>.jpg' -> 'abc' : augmented copies of one photo share a stem."""
    base = name.rsplit(".rf.", 1)[0]
    base = re.sub(r"_(jpe?g|png|webp|jfif)$", "", base, flags=re.I)
    base = Path(base).stem if "." in base[-6:] else base
    return re.sub(r"(\s*\(\d+\)|\s*-\s*copy(\s*\(\d+\))?)$", "", base.lower(), flags=re.I)


def sctl_moment(name):
    """SCTL names encode the photo moment; the four quarters of one photo share it.
    '20200724_093921_LH_C1_PP_M.jpg' -> '20200724_093921'   'R10_0815_43_crop_RH_C1_PN_m.jpg' -> 'R10_0815_43'"""
    parts = name.split("_")
    return "_".join(parts[:3]) if parts[0].startswith("R") else "_".join(parts[:2])


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---------- stage: views (venv_yolo) ----------

def stage_views():
    """Tag every v1 'mastitis' image as teat close-up / whole udder with the udder detector."""
    from ultralytics import YOLO
    model = YOLO(UDDER_DETECTOR)
    rows = []
    for folder in ["raw_mastitis_sources", "raw_healthy_sources"]:
        for f in sorted((DATA / folder).iterdir()):
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            r = model.predict(img, imgsz=512, conf=0.05, verbose=False)[0]
            best = {"udder": 0.0, "teat_closeup": 0.0}
            for b in r.boxes:
                name = r.names[int(b.cls)]
                best[name] = max(best[name], float(b.conf))
            view = ("teat_closeup" if best["teat_closeup"] >= max(best["udder"], 0.25)
                    else "udder" if best["udder"] >= 0.25 else "unclear")
            rows.append(dict(folder=folder, file=f.name, p_udder=round(best["udder"], 3),
                             p_teat=round(best["teat_closeup"], 3), view=view))
    CUR.mkdir(exist_ok=True)
    with open(CUR / "view_auto.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(Counter((r["folder"], r["view"]) for r in rows))


# ---------- collect candidates ----------

def collect():
    """Every candidate image as a dict; images are opened lazily."""
    items = []

    # 1) SCTL teat-end scores (one camera/farm style, both classes)
    for score in (1, 2, 3, 4):
        for f in sorted((DATA / "raw_sctl" / f"Score_{score}").iterdir()):
            items.append(dict(path=f, label="healthy" if score <= 2 else "problem", source="sctl",
                              score=score, group="sctl:" + sctl_moment(f.name)))

    # 2) v1 positives that are teat close-ups (whole-udder / unclear ones go to udder v2 candidates)
    views = {r["file"]: r for r in csv.DictReader(open(CUR / "view_auto.csv"))
             if r["folder"] == "raw_mastitis_sources"}
    udder_candidates = []
    for f in sorted((DATA / "raw_mastitis_sources").iterdir()):
        v = views[f.name]["view"]
        if v == "teat_closeup":
            items.append(dict(path=f, label="problem", source="web", score="",
                              group="web:" + roboflow_stem(f.name)))
        else:
            udder_candidates.append(dict(file=f.name, view_auto=v))
    with open(CUR / "udder_v2_candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "view_auto"])
        w.writeheader()
        w.writerows(udder_candidates)

    # 3) v1 healthy images that are NOT SCTL Score 1 (teat-scoring website "normal" photos)
    sctl1 = {md5(f) for f in (DATA / "raw_sctl" / "Score_1").iterdir()}
    for f in sorted((DATA / "raw_healthy_sources").iterdir()):
        if md5(f) not in sctl1:
            items.append(dict(path=f, label="healthy", source="web", score="", group="web:" + f.stem))

    # 4) Roboflow v16 boxes triaged as teat close-ups during udder curation (cropped to the box)
    with open(RF16_CUR / "view_tags.csv") as fh:
        tags = {r["box_id"]: r["view"] for r in csv.DictReader(l for l in fh if not l.startswith("#"))}
    for b in csv.DictReader(open(RF16_CUR / "rf16_boxes.csv")):
        if tags.get(b["box_id"]) == "teat_closeup":
            items.append(dict(path=RF16 / b["rf_split"] / "images" / b["image"], rf16_box=b,
                              label="problem" if b["label"] == "mastitis" else "healthy",
                              source="web", score="", group="web:" + roboflow_stem(b["image"])))
    return items, len(udder_candidates)


def load(item):
    img = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
    b = item.get("rf16_box")
    if b is None:
        return img
    lines = (RF16 / b["rf_split"] / "labels" / (Path(b["image"]).stem + ".txt")).read_text().splitlines()
    xs = [list(map(float, l.split()[1:])) for l in lines if l.split() and int(l.split()[0]) in (4, 6)]
    v = xs[int(b["box_idx"])]
    if len(v) == 4:
        x0, y0, x1, y1 = v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2
    else:
        x0, y0, x1, y1 = min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2])
    w, h = img.size
    px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
    return img.crop((int(max(0, (x0 - px) * w)), int(max(0, (y0 - py) * h)),
                     int(min(w, (x1 + px) * w)), int(min(h, (y1 + py) * h))))


def item_key(it):
    b = it.get("rf16_box")
    return f"rf16:{b['box_id']}" if b else f"{it['path'].parent.name}/{it['path'].name}"


# ---------- stage: hands (venv_handdetect) ----------

def stage_hands():
    """v1 raw folders were already hand-checked; check everything else."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(HAND_MODEL)), num_hands=4,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
    items, _ = collect()
    flagged = []
    for it in items:
        if it["path"].parent.name in ("raw_mastitis_sources", "raw_healthy_sources"):
            continue
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(load(it))))
        if res.hand_landmarks:
            flagged.append(dict(key=item_key(it), label=it["label"], source=it["source"]))
    with open(CUR / "hand_flags.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "label", "source"])
        w.writeheader()
        w.writerows(flagged)
    print(f"hands found in {len(flagged)} images:", Counter((f["label"], f["source"]) for f in flagged))


# ---------- stage: build (TF venv) ----------

def stage_build():
    items, n_udder = collect()
    hands = {r["key"] for r in csv.DictReader(open(CUR / "hand_flags.csv"))}
    dropped = Counter()

    # exact duplicates, hand images
    seen, kept = {}, []
    for it in items:
        if item_key(it) in hands:
            dropped["hand_in_photo"] += 1
            continue
        img = load(it)
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen:
            dropped["exact_duplicate"] += 1
            continue
        seen[h] = it
        it["dhash"] = dhash(img)
        kept.append(it)

    # near-duplicates (same photo under another name / re-encoded) -> merge their groups
    parent = {it["group"]: it["group"] for it in kept}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    # only among web images: SCTL is grouped exactly by photo moment, and dark low-contrast teat ends
    # give near-identical hashes that would chain unrelated SCTL and web photos into one group
    web = [it for it in kept if it["source"] == "web"]
    H = np.array([it["dhash"] for it in web])
    for i in range(len(web)):
        close = np.where((H[i + 1:] != H[i]).sum(1) <= NEAR_DUP_BITS)[0] + i + 1
        for j in close:
            parent[find(web[i]["group"])] = find(web[j]["group"])
    for it in kept:
        it["group"] = find(it["group"])

    # a group carrying both labels:
    #  - one SCTL photo moment: fine, the four quarters of one cow can score differently
    #  - the same web photo labelled both ways: the v1 "mastitis" folder contains the teat-scoring
    #    site's "normal..." photos (healthy, identical to v1 healthy images). If a group contains a
    #    trusted healthy image (v1 curated healthy folder or a 'normal...' file), the whole group is
    #    healthy. Note: 'sehat-N' files are NOT trusted - visually they are diseased teats.
    #  - any other contradiction -> drop the group
    def trusted_healthy(it):
        return (it["path"].parent.name == "raw_healthy_sources"
                or (it["path"].parent.name == "raw_mastitis_sources" and it["path"].name.lower().startswith("normal")))

    labels, trusted = defaultdict(set), set()
    for it in kept:
        labels[it["group"]].add(it["label"])
        if trusted_healthy(it):
            trusted.add(it["group"])
    mixed = {g for g, ls in labels.items() if len(ls) > 1 and not g.startswith("sctl:")}
    for it in kept:
        if it["group"] in mixed & trusted and it["label"] != "healthy":
            it["label"] = "healthy"
            dropped["relabelled_problem_to_healthy"] += 1
    conflict = mixed - trusted
    dropped["label_conflict"] = sum(it["group"] in conflict for it in kept)
    kept = [it for it in kept if it["group"] not in conflict]
    # 'normal...' files with no healthy twin are still healthy by name
    for it in kept:
        if trusted_healthy(it) and it["label"] != "healthy":
            it["label"] = "healthy"
            dropped["relabelled_problem_to_healthy"] += 1

    # grouped split, stratified by (label, source) so test has every kind of image
    rng = random.Random(SEED)
    strata = defaultdict(set)
    for it in kept:
        strata[(it["label"], it["source"])].add(it["group"])
    assign = {}
    for key in sorted(strata):
        groups = sorted(strata[key] - set(assign))      # a mixed-label SCTL moment is assigned once
        rng.shuffle(groups)
        n = len(groups)
        n_tr, n_va = round(n * SPLIT["train"]), round(n * SPLIT["val"])
        for i, g in enumerate(groups):
            assign[g] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for k, it in enumerate(sorted(kept, key=item_key)):
        split = assign[it["group"]]
        dst = OUT / split / it["label"] / f"{it['label'][0]}{k:05d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        load(it).save(dst, quality=95)
        rows.append(dict(id=dst.stem, label=it["label"], source=it["source"], score=it["score"],
                         group=it["group"], split=split, origin=item_key(it)))
    with open(CUR / "teat_v2_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"moved to udder v2 candidates: {n_udder} whole-udder/unclear v1 images")
    print("dropped:", dict(dropped))
    print(f"{'split':6s} {'label':8s} {'source':6s} images / distinct groups")
    for split in ("train", "val", "test"):
        for label in ("healthy", "problem"):
            for source in ("sctl", "web"):
                sub = [r for r in rows if (r["split"], r["label"], r["source"]) == (split, label, source)]
                print(f"{split:6s} {label:8s} {source:6s} {len(sub):5d} / {len({r['group'] for r in sub}):4d}")
    groups_multi_split = {g for g in {r["group"] for r in rows}
                          if len({r["split"] for r in rows if r["group"] == g}) > 1}
    print("groups spanning more than one split:", len(groups_multi_split))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["views", "hands", "build"], required=True)
    stage = ap.parse_args().stage
    dict(views=stage_views, hands=stage_hands, build=stage_build)[stage]()
