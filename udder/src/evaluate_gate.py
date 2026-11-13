"""
Where does the photo gate send each kind of photo?  (detector v1 vs v2)

    python src/evaluate_gate.py

Gate rule (same as predict.py): best 'udder' and best 'teat_closeup' box with conf >= 0.25;
teat wins if its confidence is higher -> "teat"; else udder -> "udder"; nothing -> "retake".

Expected routes:
    whole-udder test photos           -> udder
    teat close-up test photos (SCTL)  -> teat
    teat close-up test photos (web)   -> teat
    outside healthy udders            -> udder
    teat photos with a hand (v1 quarantine) -> teat
    tick / cow-skin photos            -> retake (neither app should score them)
Results -> reports/gate_evaluation.json
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = 0.25


def route(dets):
    best = {"udder": 0.0, "teat_closeup": 0.0}
    for d in dets:
        if d["cls"] in best:
            best[d["cls"]] = max(best[d["cls"]], float(d["conf"]))
    if best["teat_closeup"] >= CONF and best["teat_closeup"] > best["udder"]:
        return "teat"
    if best["udder"] >= CONF:
        return "udder"
    return "retake"


def load(version):
    det_dir = ROOT / "data" / ("detector" if version == "v1" else f"detector_{version}")
    dets = defaultdict(list)
    for d in csv.DictReader(open(det_dir / "test_detections.csv")):
        dets[(d["source"], d["image"])].append(d)
    return det_dir, dets


def main():
    # which v1 test images are whole-udder photos (all boxes udder view, from udder curation)
    with open(ROOT / "data" / "curation" / "view_tags.csv") as f:
        view = {r["box_id"]: r["view"] for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    boxes = defaultdict(list)
    for b in csv.DictReader(open(ROOT / "data" / "curation" / "rf16_boxes.csv")):
        boxes[b["image"]].append(b)
    teat_rows = {f"teat_{r['id']}.jpg": r for r in
                 csv.DictReader(open(ROOT.parent / "teat" / "curation" / "teat_v2_manifest.csv"))}

    results = {}
    for version in ("v1", "v2"):
        det_dir, dets = load(version)
        cats = defaultdict(list)
        for (source, image), ds in dets.items():
            if source == "test":
                if image in teat_rows:
                    cats[f"teat_test_{teat_rows[image]['source']}"].append(route(ds))
                elif image in boxes and all(view.get(b["box_id"], "udder") == "udder" for b in boxes[image]):
                    cats["udder_test"].append(route(ds))
                elif image in boxes and all(view.get(b["box_id"]) == "teat_closeup" for b in boxes[image]):
                    cats["teat_test_roboflow"].append(route(ds))
            else:
                cats[source.replace("external/", "outside_")].append(route(ds))
        results[version] = {k: dict(n=len(v), **{r: round(c / len(v), 3) for r, c in Counter(v).items()})
                            for k, v in sorted(cats.items())}

    out = ROOT / "reports" / "gate_evaluation.json"
    json.dump(results, open(out, "w"), indent=2)
    names = sorted(set(results["v1"]) | set(results["v2"]))
    print(f"{'photo kind':30s} {'detector v1':>34s} {'detector v2':>34s}")
    fmt = lambda r: "" if r is None else f"n={r['n']:3d} " + " ".join(f"{k}={r.get(k, 0):.2f}" for k in ("udder", "teat", "retake"))
    for n in names:
        print(f"{n:30s} {fmt(results['v1'].get(n)):>34s} {fmt(results['v2'].get(n)):>34s}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
