"""
evaluate_v2.py

Honest evaluation of the teat v2 model (models/teat_classifier_v2.keras) on the grouped test split.

    python scripts/evaluate_v2.py

Reports (reports/teat_v2_evaluation.json):
  - test metrics with 95% intervals from a bootstrap over source groups (photo moments / photos)
  - SCTL-only metrics: both classes share one camera/farm style there, so the photo-style shortcut
    can't help; this is the most trustworthy number
  - same-cow pairs: SCTL photo moments that contain both a healthy (score 1-2) and a problem (3-4) teat
  - per-source / per-score breakdown (e.g. are smooth rings, score 2, flagged?)
  - calibration (Platt on the validation split, equal prior) + confidence bands, like the udder model
  - stress test that v1 failed: whole udders, tick (cow-skin) photos, hand photos
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TEAT = Path(__file__).resolve().parents[1]
REPO = TEAT.parent
MODEL = TEAT / "models" / "teat_classifier_v2.keras"
MANIFEST = TEAT / "curation" / "teat_v2_manifest.csv"
DATA = TEAT / "data" / "processed_v2"
REPORTS = TEAT / "reports"
IMG = 224                  # replaced by the model's own input size in main()
CROPPER = None             # --teat_end: crop every photo to its teat end first (scripts/teat_end_crop.py)
BANDS = [("high", 0.85), ("medium", 0.65), ("low", 0.0)]

STRESS = {
    "whole_udder_healthy": REPO / "udder_images" / "healthy",
    "tick_cow_skin": REPO / "tick_images" / "infested",
    "hand_in_photo_v1_quarantine": TEAT / "data" / "quarantined_hand_images",
    "whole_udder_mastitis_v1_moved_out": None,      # filled from udder_v2_candidates.csv
}


def load_input(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if CROPPER is not None:
        img = CROPPER.crop(img)[0]
    return np.asarray(img.resize((IMG, IMG), Image.BILINEAR), dtype=np.float32)


def predict(model, paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        out.append(model.predict(np.stack([load_input(p) for p in paths[i:i + batch]]), verbose=0).ravel())
    return np.concatenate(out)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(y, p):
    from sklearn.metrics import roc_auc_score
    pred = p >= 0.5
    sens = float(pred[y == 1].mean()) if (y == 1).any() else None
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else None
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    bal = (sens + spec) / 2 if sens is not None and spec is not None else None
    return dict(n=int(len(y)), roc_auc=auc, sensitivity=sens, specificity=spec, balanced_accuracy=bal)


def bootstrap_ci(y, p, groups, key, n=2000, seed=0):
    """95% interval for one metric, resampling whole source groups."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        v = metrics(y[idx], p[idx])[key]
        if v is not None:
            vals.append(v)
    return [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])]


def main(model_path=MODEL, out_name="teat_v2_evaluation.json", teat_end=False):
    import keras
    global IMG, CROPPER
    if teat_end:
        from teat_end_crop import TeatEndCropper
        CROPPER = TeatEndCropper()
    model = keras.models.load_model(model_path)
    IMG = model.input_shape[1]
    rows = list(csv.DictReader(open(MANIFEST)))
    for r in rows:
        r["path"] = DATA / r["split"] / r["label"] / f"{r['id']}.jpg"

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    raw = {s: predict(model, [r["path"] for r in by_split[s]]) for s in ("val", "test")}

    # calibration on val (balanced prior), applied to test
    from sklearn.linear_model import LogisticRegression
    yv = np.array([r["label"] == "problem" for r in by_split["val"]], dtype=int)
    wv = np.where(yv == 1, 0.5 / yv.sum(), 0.5 / (1 - yv).sum())
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw["val"]).reshape(-1, 1), yv, sample_weight=wv)
    A, B = float(cal.coef_[0, 0]), float(cal.intercept_[0])
    calib = lambda p: 1 / (1 + np.exp(-(A * logit(p) + B)))

    test = by_split["test"]
    y = np.array([r["label"] == "problem" for r in test], dtype=int)
    g = np.array([r["group"] for r in test])
    src = np.array([r["source"] for r in test])
    score = np.array([r["score"] for r in test])
    p_raw, p = raw["test"], calib(raw["test"])

    res = dict(model=Path(model_path).name, calibration=dict(A=A, B=B, fitted_on="val split, equal prior"))
    for name, mask in [("all_test", np.ones(len(y), bool)), ("sctl_only", src == "sctl"), ("web_only", src == "web")]:
        m = metrics(y[mask], p[mask])
        for k in ("roc_auc", "sensitivity", "specificity", "balanced_accuracy"):
            if m[k] is not None:
                m[k + "_ci95"] = bootstrap_ci(y[mask], p[mask], g[mask], k)
        m["distinct_groups"] = dict(healthy=int(len(set(g[mask & (y == 0)]))), problem=int(len(set(g[mask & (y == 1)]))))
        res[name] = m

    # same-cow pairs: SCTL moments with both labels
    mixed = [grp for grp in set(g[src == "sctl"]) if len(set(y[g == grp])) == 2]
    mm = np.isin(g, mixed)
    res["sctl_same_cow_moments"] = dict(moments=len(mixed), **metrics(y[mm], p[mm]))

    res["flagged_rate_by_kind"] = {}
    for s in ("1", "2", "3", "4"):
        m = score == s
        res["flagged_rate_by_kind"][f"sctl_score_{s}"] = dict(n=int(m.sum()), flagged=round(float((p[m] >= 0.5).mean()), 3))
    for lab in ("healthy", "problem"):
        m = (src == "web") & (y == (lab == "problem"))
        res["flagged_rate_by_kind"][f"web_{lab}"] = dict(n=int(m.sum()), flagged=round(float((p[m] >= 0.5).mean()), 3))

    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    band = np.array([next(name for name, lo in BANDS if c >= lo) for c in conf])
    res["confidence_bands_test"] = {
        name: dict(share=round(float((band == name).mean()), 3),
                   right=round(float(correct[band == name].mean()), 3) if (band == name).any() else None)
        for name, _ in BANDS}
    res["raw_vs_calibrated_extreme_share"] = dict(
        raw=round(float(((p_raw < 0.02) | (p_raw > 0.98)).mean()), 3),
        calibrated=round(float(((p < 0.02) | (p > 0.98)).mean()), 3))

    # stress test (images the teat model should not be confidently wrong on)
    cand = [TEAT / "data" / "raw_mastitis_sources" / r["file"]
            for r in csv.DictReader(open(TEAT / "curation" / "udder_v2_candidates.csv"))]
    res["stress_test"] = {}
    for name, folder in STRESS.items():
        paths = cand if folder is None else sorted(f for f in Path(folder).rglob("*") if f.is_file())
        if not paths:
            continue
        ps = calib(predict(model, paths))
        res["stress_test"][name] = dict(n=len(paths), flagged_problem=round(float((ps >= 0.5).mean()), 3),
                                        very_confident=round(float(((ps < 0.05) | (ps > 0.95)).mean()), 3))

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / out_name
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--out", default="teat_v2_evaluation.json")
    ap.add_argument("--teat_end", action="store_true", help="crop each photo to its teat end (models trained on crops)")
    a = ap.parse_args()
    main(a.model, a.out, a.teat_end)
