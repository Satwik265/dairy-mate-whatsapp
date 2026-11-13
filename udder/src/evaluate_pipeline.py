"""
End-to-end check of the farmer flow: whole photo -> udder detector -> crop -> udder classifier.
Runs in the TensorFlow venv after `detector.py --stage detect` has written test_detections.csv.

    python src/evaluate_pipeline.py

Compares, on the held-out test photos:
    full_photo      classifier on the whole photo (what happens with no detector)
    true_crop       classifier on the hand-labelled Roboflow box (best case)
    detector_crop   classifier on the detector's best udder box (the real app flow)

The classifier used here is fitted on train+val only (never on test), with the calibration from
reports/udder_final_evaluation.json, so the test numbers are honest.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from build_udder_model import (load_rows, load_features, arrays, fit_classifier, logit_of,
                               feature_model, to_input, sigmoid, band_of)

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "data" / "detector"
CUR = ROOT / "data" / "curation"
REPORTS = ROOT / "reports"

UDDER_CONF = 0.25        # below this the app says "we couldn't find the udder, move closer"
CROP_PAD = 0.05          # same padding as curate_roboflow.crop()


def crop_box(img, x0, y0, x1, y1):
    w, h = img.size
    px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
    return img.crop((int(max(0, (x0 - px) * w)), int(max(0, (y0 - py) * h)),
                     int(min(w, (x1 + px) * w)), int(min(h, (y1 + py) * h))))


def decide(dets):
    """App logic for one photo: returns ('score', box) | ('teat', None) | ('no_udder', None)."""
    udders = [d for d in dets if d["cls"] == "udder" and float(d["conf"]) >= UDDER_CONF]
    teats = [d for d in dets if d["cls"] == "teat_closeup" and float(d["conf"]) >= UDDER_CONF]
    best_u = max(udders, key=lambda d: float(d["conf"]), default=None)
    best_t = max(teats, key=lambda d: float(d["conf"]), default=None)
    if best_t and (not best_u or float(best_t["conf"]) > float(best_u["conf"])):
        return "teat", None
    if best_u:
        return "score", tuple(float(best_u[k]) for k in ("x0", "y0", "x1", "y1"))
    return "no_udder", None


def main():
    # classifier fitted on train+val only
    rows = load_rows()
    X = load_features(rows)
    y, groups, w = arrays(rows)
    dev = np.array([r["split"] != "test" for r in rows])
    clf = fit_classifier(X[dev], y[dev], w[dev])
    cal = json.load(open(REPORTS / "udder_final_evaluation.json"))["calibration"]
    fm = feature_model()

    def score(images):
        feats = fm.predict(np.stack([to_input(im) for im in images]), verbose=0)
        return sigmoid(cal["A"] * logit_of(clf, feats) + cal["B"])

    dets = defaultdict(list)
    for d in csv.DictReader(open(DET / "test_detections.csv")):
        dets[(d["source"], d["image"])].append(d)

    # ground truth for detector test images: whole-udder boxes from the curation tables
    boxes = {r["box_id"]: r for r in csv.DictReader(open(CUR / "rf16_boxes.csv"))}
    with open(CUR / "view_tags.csv") as f:
        view = {r["box_id"]: r["view"] for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    truth = defaultdict(list)
    for b in boxes.values():
        truth[b["image"]].append(b)

    test_imgs = sorted(p.name for p in (DET / "images" / "test").glob("*"))
    udder_imgs, teat_imgs = [], []
    for name in test_imgs:
        bs = truth[name]
        if all(view.get(b["box_id"]) == "teat_closeup" for b in bs):
            teat_imgs.append(name)
        elif all(view.get(b["box_id"], "udder") == "udder" for b in bs) and len({b["label"] for b in bs}) == 1:
            udder_imgs.append(name)

    lab = np.array([truth[n][0]["label"] == "mastitis" for n in udder_imgs])
    full, true_crop, det_crop, actions = [], [], [], []
    for n in udder_imgs:
        img = Image.open(DET / "images" / "test" / n).convert("RGB")
        full.append(img)
        # true box from the Roboflow label file
        lines = (DET / "labels" / "test" / (Path(n).stem + ".txt")).read_text().split("\n")
        cx, cy, bw, bh = map(float, lines[0].split()[1:])
        true_crop.append(crop_box(img, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        action, box = decide(dets[("test", n)])
        actions.append(action)
        det_crop.append(crop_box(img, *box) if action == "score" else img)

    p_full, p_true, p_det = score(full), score(true_crop), score(det_crop)
    acts = np.array(actions)
    scored = acts == "score"

    def summary(p, mask):
        yy, pp = lab[mask], p[mask]
        return dict(n=int(mask.sum()), roc_auc=float(roc_auc_score(yy, pp)),
                    healthy_flagged=float(np.mean(pp[~yy] >= 0.5)),
                    mastitis_caught=float(np.mean(pp[yy] >= 0.5)))

    all_mask = np.ones(len(lab), bool)
    p_avg = (p_full + p_det) / 2
    with open(REPORTS / ("udder_pipeline_per_photo.csv" if DET.name == "detector" else f"udder_pipeline_per_photo_{DET.name}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "label", "action", "p_full", "p_true_crop", "p_detector_crop", "p_avg"])
        for row in zip(udder_imgs, np.where(lab, "mastitis", "healthy"), actions,
                       *[np.round(v, 4) for v in (p_full, p_true, p_det, p_avg)]):
            wr.writerow(row)
    results = dict(
        udder_test_photos=dict(n=len(lab), mastitis=int(lab.sum()), healthy=int((~lab).sum())),
        full_photo=summary(p_full, all_mask),
        true_crop=summary(p_true, all_mask),
        full_photo_scored_only=summary(p_full, scored),
        detector_crop_scored_only=summary(p_det, scored),
        avg_full_and_detector_crop_scored_only=summary(p_avg, scored),
        detector_actions_on_udder_photos={a: int((acts == a).sum()) for a in ("score", "teat", "no_udder")},
        confidence_bands_detector_crop={b: int((band_of(p_det[scored]) == b).sum()) for b in ("high", "medium", "low")},
    )

    # teat close-up test photos: how often does the app redirect them to Teat Check?
    t_acts = [decide(dets[("test", n)])[0] for n in teat_imgs]
    results["teat_closeup_photos"] = {a: t_acts.count(a) for a in ("teat", "score", "no_udder")}

    # outside photos (never used in training)
    for src in sorted({s for s, _ in dets if s.startswith("external/")}):
        names = sorted({n for s, n in dets if s == src})
        acts_e = [decide(dets[(src, n)]) for n in names]
        entry = {a: sum(1 for x, _ in acts_e if x == a) for a in ("score", "teat", "no_udder")}
        to_score = [(n, b) for n, (a, b) in zip(names, acts_e) if a == "score"]
        if to_score:
            imgs = [crop_box(Image.open(DET / src / n).convert("RGB"), *b) for n, b in to_score]
            p = score(imgs)
            entry["flagged_mastitis"] = int((p >= 0.5).sum())
        results[src] = entry

    REPORTS.mkdir(exist_ok=True)
    name = "udder_pipeline_evaluation.json" if DET.name == "detector" else f"udder_pipeline_evaluation_{DET.name}.json"
    json.dump(results, open(REPORTS / name, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1",
                    help="which detector's test_detections.csv to use")
    if ap.parse_args().version == "v2":
        DET = ROOT / "data" / "detector_v2"
    main()
