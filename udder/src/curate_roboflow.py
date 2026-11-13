"""
Curate udder crops from the Roboflow "cow and mastitis detection" v16 export.

Stages (run in order):
    python src/curate_roboflow.py --stage flags   # crop every udder box, compute auto flags, draw review sheets
    python src/curate_roboflow.py --stage build   # apply flags + view tags, group sources, grouped split, write crops

Inputs:
    data/raw_roboflow_v16/{train,valid,test}/{images,labels}   (YOLO export, no augmentation)
    data/curation/view_tags.csv   (written by hand after looking at the review sheets;
                                   one row per box_id: view = udder | teat_closeup | reject)
Outputs:
    data/curation/rf16_boxes.csv          every udder box with its auto flags
    data/curation/review_sheets/*.jpg     numbered contact sheets used for the view triage
    data/curation/rf16_clean_manifest.csv final kept crops, group id, split
    data/processed/rf16_crops/{healthy,mastitis}/*.jpg

Only PIL + numpy are used so this runs in the existing TF venv.
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_roboflow_v16"
CUR = ROOT / "data" / "curation"
OUT = ROOT / "data" / "processed" / "rf16_crops"

CLASS_IDS = {4: "healthy", 6: "mastitis"}   # from data.yaml: Healthy_udder=4, Mastitis_infected_udder=6
MIN_CROP_SIDE = 96      # crops smaller than this (px, short side) are too blurry to compare fairly
CROP_PAD = 0.05         # pad each box by 5% so the udder edge isn't clipped
SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------- source grouping ----------

def source_stem(filename):
    """Strip Roboflow's '.rf.<hash>' suffix and '_jpg'-style extension tags."""
    base = filename.rsplit(".rf.", 1)[0]
    base = re.sub(r"_(jpe?g|png|webp|jfif)$", "", base, flags=re.I)
    return base


def source_group(stem):
    """Frames of one video share a group; ' (1)' / ' - Copy' duplicates collapse too."""
    s = stem.lower()
    m = re.match(r"(.*?(mp4|mov|avi))-\d+$", s)
    if m:
        return "video:" + m.group(1)
    s = re.sub(r"(\s*\(\d+\)|\s*-\s*copy(\s*\(\d+\))?)$", "", s)
    return re.sub(r"[\s_\-]+", "", s)


# ---------- automatic image flags ----------

def is_thermal(rgb):
    """Thermal palettes are dominated by saturated purple/magenta + yellow/orange."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    h, s = hsv[..., 0] * (360 / 255), hsv[..., 1] / 255
    sat = s > 0.55
    purple = sat & (h > 250) & (h < 320)
    warm = sat & (h > 20) & (h < 60)
    return purple.mean() > 0.08 and (purple.mean() + warm.mean()) > 0.45


def is_grayscale(rgb):
    c = rgb.astype(np.int16)
    return np.abs(c[..., 0] - c[..., 1]).mean() < 4 and np.abs(c[..., 1] - c[..., 2]).mean() < 4


def black_corners(rgb, k=10):
    """Roboflow rotation augments leave pure-black triangles in the corners."""
    h, w, _ = rgb.shape
    patches = [rgb[:k, :k], rgb[:k, w - k:], rgb[h - k:, :k], rgb[h - k:, w - k:]]
    return sum(p.max() < 12 for p in patches) >= 2


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


# ---------- helpers ----------

def iter_udder_boxes():
    for split in ["train", "valid", "test"]:
        for lab in sorted((RAW / split / "labels").glob("*.txt")):
            img_path = next((RAW / split / "images" / (lab.stem + e) for e in (".jpg", ".jpeg", ".png")
                             if (RAW / split / "images" / (lab.stem + e)).exists()), None)
            if img_path is None:
                continue
            k = 0
            for line in lab.read_text().splitlines():
                parts = line.split()
                if not parts or int(parts[0]) not in CLASS_IDS:
                    continue
                xs = list(map(float, parts[1:]))
                if len(xs) == 4:
                    cx, cy, bw, bh = xs
                    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                else:  # polygon label
                    x0, x1, y0, y1 = min(xs[0::2]), max(xs[0::2]), min(xs[1::2]), max(xs[1::2])
                yield dict(split_rf=split, image=img_path, label=CLASS_IDS[int(parts[0])],
                           box=(x0, y0, x1, y1), box_idx=k)
                k += 1


def crop(img, box):
    w, h = img.size
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
    return img.crop((int(max(0, (x0 - px) * w)), int(max(0, (y0 - py) * h)),
                     int(min(w, (x1 + px) * w)), int(min(h, (y1 + py) * h))))


# ---------- stage: flags ----------

def stage_flags():
    CUR.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in iter_udder_boxes():
        img = Image.open(b["image"]).convert("RGB")
        full = np.asarray(img)
        c = crop(img, b["box"])
        stem = source_stem(b["image"].stem)
        x0, y0, x1, y1 = b["box"]
        rows.append(dict(
            box_id=f"{b['label'][0]}{len(rows):04d}",
            label=b["label"], rf_split=b["split_rf"], image=b["image"].name, box_idx=b["box_idx"],
            source_stem=stem, group=source_group(stem),
            box_area=round((x1 - x0) * (y1 - y0), 4), crop_w=c.width, crop_h=c.height,
            thermal=int(is_thermal(full)), grayscale=int(is_grayscale(full)),
            black_corners=int(black_corners(full)),
            small=int(min(c.size) < MIN_CROP_SIDE),
            dhash="".join("1" if v else "0" for v in dhash(c)),
        ))
        rows[-1]["_crop"] = c

    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(CUR / "rf16_boxes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # numbered review sheets, 48 crops each, auto flags printed on the tile
    sheet_dir = CUR / "review_sheets"
    sheet_dir.mkdir(exist_ok=True)
    tile, cols = 160, 8
    for label in ["healthy", "mastitis"]:
        sub = [r for r in rows if r["label"] == label]
        for s in range(0, len(sub), 48):
            chunk = sub[s:s + 48]
            sheet = Image.new("RGB", (cols * tile, 6 * tile), "white")
            draw = ImageDraw.Draw(sheet)
            for i, r in enumerate(chunk):
                t = r["_crop"].copy()
                t.thumbnail((tile - 4, tile - 18))
                x, y = (i % cols) * tile, (i // cols) * tile
                sheet.paste(t, (x + 2, y + 16))
                flags = "".join(ch for ch, k in [("T", "thermal"), ("G", "grayscale"),
                                                 ("B", "black_corners"), ("S", "small")] if r[k])
                draw.rectangle([x, y, x + tile, y + 14], fill="black")
                draw.text((x + 3, y + 2), f"{r['box_id']} {flags}", fill="yellow")
            sheet.save(sheet_dir / f"{label}_{s // 48:02d}.jpg", quality=85)

    n = defaultdict(int)
    for r in rows:
        n[r["label"]] += 1
        for k in ["thermal", "grayscale", "black_corners", "small"]:
            n[(r["label"], k)] += r[k]
    print(f"udder boxes: {len(rows)}")
    for label in ["healthy", "mastitis"]:
        print(f"  {label}: {n[label]} boxes | thermal {n[(label, 'thermal')]} | grayscale "
              f"{n[(label, 'grayscale')]} | black corners {n[(label, 'black_corners')]} | "
              f"small (<{MIN_CROP_SIDE}px) {n[(label, 'small')]}")
    print(f"review sheets -> {sheet_dir}")


# ---------- stage: build ----------

def merge_near_duplicates(rows, max_dist=6):
    """Union groups whose crops are near-identical (same photo re-uploaded under another name)."""
    parent = {r["group"]: r["group"] for r in rows}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    hashes = [(r["group"], np.array([c == "1" for c in r["dhash"]])) for r in rows]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if (hashes[i][1] != hashes[j][1]).sum() <= max_dist:
                parent[find(hashes[i][0])] = find(hashes[j][0])
    for r in rows:
        r["group"] = find(r["group"])


def grouped_split(rows):
    """Stratified by label, grouped by source: a source photo lands in exactly one split."""
    rng = random.Random(SEED)
    by_label = defaultdict(set)
    for r in rows:
        by_label[r["label"]].add(r["group"])
    assign = {}
    for label, groups in by_label.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_tr, n_va = round(n * SPLIT["train"]), round(n * SPLIT["val"])
        for i, g in enumerate(groups):
            assign[g] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return assign


def stage_build():
    rows = list(csv.DictReader(open(CUR / "rf16_boxes.csv")))
    with open(CUR / "view_tags.csv") as f:
        tags = {r["box_id"]: r for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    drop = defaultdict(int)
    kept = []
    for r in rows:
        view = tags.get(r["box_id"], {}).get("view", "udder")
        reason = ("thermal" if r["thermal"] == "1" else
                  "small_crop" if r["small"] == "1" else
                  "teat_closeup" if view == "teat_closeup" else
                  "rejected_in_triage" if view == "reject" else None)
        if reason:
            drop[(r["label"], reason)] += 1
            continue
        kept.append(r)

    # a group whose crops carry both labels is contradictory -> drop it entirely
    labels_per_group = defaultdict(set)
    merge_near_duplicates(kept)
    for r in kept:
        labels_per_group[r["group"]].add(r["label"])
    conflicted = {g for g, ls in labels_per_group.items() if len(ls) > 1}
    for r in kept:
        if r["group"] in conflicted:
            drop[(r["label"], "label_conflict")] += 1
    kept = [r for r in kept if r["group"] not in conflicted]

    assign = grouped_split(kept)
    for d in ["train", "val", "test"]:
        for label in ["healthy", "mastitis"]:
            (OUT / d / label).mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in kept:
        split = assign[r["group"]]
        src_split = r["rf_split"]
        img = Image.open(RAW / src_split / "images" / r["image"]).convert("RGB")
        lab = (RAW / src_split / "labels" / (Path(r["image"]).stem + ".txt")).read_text().splitlines()
        boxes = [list(map(float, l.split()[1:])) for l in lab
                 if l.split() and int(l.split()[0]) in CLASS_IDS]
        xs = boxes[int(r["box_idx"])]
        if len(xs) == 4:
            cx, cy, bw, bh = xs
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        else:
            box = (min(xs[0::2]), min(xs[1::2]), max(xs[0::2]), max(xs[1::2]))
        dst = OUT / split / r["label"] / f"{r['box_id']}.jpg"
        crop(img, box).save(dst, quality=95)
        out_rows.append(dict(box_id=r["box_id"], label=r["label"], group=r["group"], split=split,
                             crop_path=str(dst.relative_to(ROOT)).replace("\\", "/"),
                             source_image=r["image"], grayscale=r["grayscale"],
                             black_corners=r["black_corners"]))

    with open(CUR / "rf16_clean_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print("dropped:")
    for (label, reason), n in sorted(drop.items()):
        print(f"  {label:9s} {reason:20s} {n}")
    print("kept (crops / distinct source groups):")
    for split in ["train", "val", "test"]:
        for label in ["healthy", "mastitis"]:
            sub = [r for r in out_rows if r["split"] == split and r["label"] == label]
            print(f"  {split:5s} {label:9s} {len(sub):4d} / {len({r['group'] for r in sub}):4d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["flags", "build"], required=True)
    args = ap.parse_args()
    stage_flags() if args.stage == "flags" else stage_build()
