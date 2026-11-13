"""
Build the grouped train/val/test split for the tick model -> tick/curation/split_manifest.csv
and copies of the images under tick/data/split/{train,val,test}/ (wiped before every write).

    python tick/scripts/build_split.py

Two sources:

1. Griffith University "Ticks" (Roboflow v1, tick/data/raw/, CC BY 4.0): the positives.
   - Roboflow v1 has 855 files but only 455 distinct images: each train image is exported twice
     (2 noise-augmented outputs). Copies share the original filename before ".rf.<hash>", so one copy is kept.
   - Every image shows the same few Hereford calves in one crush pen. Source videos (IMG_1428_frame380 ...)
     are grouped by video. The "standalone" photos (IMG_1339 ... IMG_1495) are consecutive shots of the same
     calves seconds apart, so treating each as its own group would leak near-duplicates across splits.
     They are grouped by run of 10 IMG numbers (IMG_134x, IMG_135x ...). Neighbouring runs can still look alike;
     there is no way to get animal-independent splits from this set.
   - Only 20 of the 455 images have no tick box. Most are frames of videos IMG_1386 and IMG_1485, so those two
     videos are pinned to val and test so both get some image-level negatives. IMG_1428 (121 frames, 27% of the
     set) is pinned to train. Everything else is assigned by a seeded greedy fill toward the target ratios.
   - The 8 "suggested-*" images are other cows on other farms, each its own group.

2. Tick-free cows from the Roboflow "cow and mastitis detection" v16 export already used by the udder model
   (udder/data/raw_roboflow_v16/, CC BY 4.0): images whose only boxes are Cow_Feeding / Cow_drinking_water /
   Cow_lying / Cow_standing. They are NOT labelled for ticks; they are presumed tick-free (dairy / CCTV / stock
   photos at a distance). Used as image-level negatives and as extra negative patches.
   Deduplicated by original filename, grouped by series (CCTV "walk"/"lie" frames, video names, numeric prefixes).

Rule groups linked by any near-duplicate pair (dHash) are merged before splitting.
Checks before writing: no group in two splits, no md5 duplicate across splits; near-duplicate pairs across splits are
reported (0 by construction).
"""

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TICK = ROOT / "tick"
GRIFFITH_RAW = TICK / "data" / "raw"
RF16_RAW = ROOT / "udder" / "data" / "raw_roboflow_v16"
COW_ONLY_CLASSES = {0, 1, 2, 3}          # rf16: Cow_Feeding, Cow_drinking_water, Cow_lying, Cow_standing
NEAR_DUP = 6                              # dHash Hamming distance treated as "same scene"
PINNED = {"vid_IMG_1428": "train", "vid_IMG_1386": "val", "vid_IMG_1485": "test"}


def original_stem(name):
    stem = Path(name).stem.split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png|webp)$", "", stem, flags=re.IGNORECASE)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def griffith_group(stem):
    m = re.match(r"(IMG_\d+)_frame\d+$", stem)
    if m:
        return "vid_" + m.group(1)
    m = re.match(r"IMG_(\d+)$", stem)
    if m:
        return f"photos_IMG_{int(m.group(1)) // 10}x"
    return "other_" + stem


def rf16_group(stem):
    """Conservative: anything that could be frames/bursts from one camera or session shares a group."""
    s = stem.lower()
    m = re.match(r"img_(\d{8})_", s)
    if m:
        return "rf16_img_" + m.group(1)                    # phone photos: one group per day
    if re.match(r"img_\d+$", s):
        return "rf16_img_seq"
    if re.match(r"\d+$", s):
        return "rf16_num8" if len(s) >= 6 else "rf16_num"  # bare frame numbers from unknown sources
    m = re.match(r"(\d+)[-_ ]", s)
    if m:
        return "rf16_n" + m.group(1)                       # "1-958-", "33_mp4-0005" -> leading number series
    m = re.match(r"([a-z]+)", s)
    return "rf16_" + (m.group(1) if m else s)              # walk*, lie*, adobestock*, frame*, grz*, youtube* ...


def dhash(path, size=8):
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(im.tobytes())
    bits = [px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] for r in range(size) for c in range(size)]
    return int("".join("1" if b else "0" for b in bits), 2)


def read_labels(path):
    if not path.exists():
        return []
    return [int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]


def collect(raw_dir, source, keep):
    rows = {}
    for sp in ("train", "valid", "test"):
        img_dir = raw_dir / sp / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            stem = original_stem(p.name)
            if stem in rows:
                continue                                    # Roboflow augmentation copy of an image already kept
            lbl = raw_dir / sp / "labels" / (p.stem + ".txt")
            cls = read_labels(lbl)
            if not keep(cls):
                continue
            rows[stem] = dict(source=source, stem=stem, image=p.relative_to(ROOT).as_posix(),
                              label_file=lbl.relative_to(ROOT).as_posix() if lbl.exists() else "",
                              classes=cls)
    return list(rows.values())


def assign(groups, ratios, seed, pinned):
    total = sum(len(v) for v in groups.values())
    target = {s: total * r for s, r in ratios.items()}
    count = Counter()
    out = {}
    for g, s in pinned.items():
        if g in groups:
            out[g] = s
            count[s] += len(groups[g])
    rest = [g for g in groups if g not in out]
    random.Random(seed).shuffle(rest)
    rest.sort(key=lambda g: -len(groups[g]))           # place big groups first, ties in seeded random order
    for g in rest:
        s = max(ratios, key=lambda k: target[k] - count[k])
        out[g] = s
        count[s] += len(groups[g])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(TICK / "data" / "split"))
    ap.add_argument("--manifest", default=str(TICK / "curation" / "split_manifest.csv"))
    args = ap.parse_args()

    grif = collect(GRIFFITH_RAW, "griffith", keep=lambda c: True)
    rf16 = collect(RF16_RAW, "rf16_cows", keep=lambda c: bool(c) and set(c) <= COW_ONLY_CLASSES)
    for r in grif:
        r["group"] = griffith_group(r["stem"])
        r["n_tick"], r["n_cluster"], r["n_cattle"] = r["classes"].count(1), r["classes"].count(2), r["classes"].count(0)
    for r in rf16:
        r["group"] = rf16_group(r["stem"])
        r["n_tick"] = r["n_cluster"] = 0
        r["n_cattle"] = len(r["classes"])
    print(f"griffith: {len(grif)} distinct images; rf16 tick-free cows: {len(rf16)} distinct images")

    # Merge rule groups that share any near-duplicate pair (64-bit dHash, Hamming <= NEAR_DUP), so photos of one
    # scene taken around a video, or consecutive CCTV clips, never straddle two splits.
    for r in grif + rf16:
        r["dhash"] = dhash(ROOT / r["image"])
        r["rule_group"] = r["group"]
    for rows in (grif, rf16):
        parent = {r["group"]: r["group"] for r in rows}

        def find(g):
            while parent[g] != g:
                parent[g] = parent[parent[g]]
                g = parent[g]
            return g
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["group"] != b["group"] and bin(a["dhash"] ^ b["dhash"]).count("1") <= NEAR_DUP:
                    ra, rb = find(a["group"]), find(b["group"])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        members = defaultdict(set)
        for g in parent:
            members[find(g)].add(g)
        for r in rows:
            m = sorted(members[find(r["rule_group"])])
            r["group"] = m[0] if len(m) == 1 else m[0] + f"+{len(m) - 1}"
        merged = {k: v for k, v in members.items() if len(v) > 1}
        print(f"  {rows[0]['source']}: {len(parent)} rule groups -> {len(members)} after near-duplicate merge;"
              f" merged: {[sorted(v) for v in merged.values()]}")

    split_of = {}
    for rows, ratios, pins in ((grif, {"train": 0.65, "val": 0.20, "test": 0.15}, PINNED),
                               (rf16, {"train": 0.50, "val": 0.25, "test": 0.25}, {})):
        groups = defaultdict(list)
        for r in rows:
            groups[r["group"]].append(r)
        pinned = {}
        for r in rows:
            if r["rule_group"] in pins:
                if pinned.get(r["group"], pins[r["rule_group"]]) != pins[r["rule_group"]]:
                    raise SystemExit(f"pinned groups with different splits were merged: {r['group']}")
                pinned[r["group"]] = pins[r["rule_group"]]
        split_of.update(assign(groups, ratios, args.seed, pinned))

    rows = grif + rf16
    for r in rows:
        r["split"] = split_of[r["group"]]
        r["has_ticks"] = int(r["n_tick"] + r["n_cluster"] > 0)
        r["md5"] = md5(ROOT / r["image"])

    # leakage checks
    g_split = defaultdict(set)
    m_split = defaultdict(set)
    for r in rows:
        g_split[r["group"]].add(r["split"])
        m_split[r["md5"]].add(r["split"])
    bad_g = [g for g, s in g_split.items() if len(s) > 1]
    bad_m = [m for m, s in m_split.items() if len(s) > 1]
    dup_m = sum(1 for m in m_split if sum(1 for r in rows if r["md5"] == m) > 1)
    if bad_g or bad_m:
        raise SystemExit(f"LEAKAGE: groups in >1 split {bad_g[:5]}, md5 in >1 split {len(bad_m)}; nothing written")
    print(f"[OK] {len(g_split)} groups, each in exactly one split; md5 duplicates across splits: 0 "
          f"(md5 duplicates within a split: {dup_m})")

    # near-duplicate check between splits (should be 0 after the merge above)
    near = Counter()
    for src in ("griffith", "rf16_cows"):
        sel = [r for r in rows if r["source"] == src]
        for i, a in enumerate(sel):
            for b in sel[i + 1:]:
                if a["split"] != b["split"] and bin(a["dhash"] ^ b["dhash"]).count("1") <= NEAR_DUP:
                    near[(src, *sorted((a["split"], b["split"])))] += 1
    print(f"near-duplicate pairs across splits (dHash Hamming <= {NEAR_DUP}):", dict(near) or "none")

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    for r in rows:
        d = out / r["split"] / r["source"]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / (r["stem"] + Path(r["image"]).suffix.lower())
        shutil.copy2(ROOT / r["image"], dst)
        if r["label_file"]:
            shutil.copy2(ROOT / r["label_file"], dst.with_suffix(".txt"))
        r["split_image"] = dst.relative_to(ROOT).as_posix()

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    cols = ["source", "split", "group", "rule_group", "stem", "has_ticks", "n_tick", "n_cluster", "n_cattle", "md5", "image", "split_image"]
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["source"], r["split"], r["group"], r["stem"])):
            w.writerow(r)

    print(f"\n{'source':10s} {'split':5s} {'images':>6s} {'groups':>6s} {'with ticks':>10s} {'no ticks':>8s}")
    for src in ("griffith", "rf16_cows"):
        for sp in ("train", "val", "test"):
            sel = [r for r in rows if r["source"] == src and r["split"] == sp]
            print(f"{src:10s} {sp:5s} {len(sel):6d} {len({r['group'] for r in sel}):6d} "
                  f"{sum(r['has_ticks'] for r in sel):10d} {sum(1 - r['has_ticks'] for r in sel):8d}")
    for sp in ("train", "val", "test"):
        vids = sorted({r["group"] for r in grif if r["split"] == sp and r["group"].startswith("vid_")})
        print(f"  griffith {sp} videos: {vids}")
    print("wrote", args.manifest, "and", out)


if __name__ == "__main__":
    main()
