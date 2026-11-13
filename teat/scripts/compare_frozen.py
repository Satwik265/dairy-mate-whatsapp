"""
compare_frozen.py

Compare the fine-tuned teat v2 CNN with the udder recipe (frozen ImageNet MobileNetV2 + logistic regression)
on the same grouped teat v2 split.

    python scripts/compare_frozen.py --stage features
    python scripts/compare_frozen.py --stage compare

Frozen-feature variants:
    frozen_logreg           class-balanced weights, each source group counts once
    frozen_logreg_cellbal   each (label x source) cell gets equal total weight, so the ~15 non-SCTL healthy
                            photos are not drowned out by ~200 web problem photos (targets the
                            "not SCTL style = problem" shortcut)

Selection is done on train+val with grouped CV out-of-fold predictions (never on test). The headline
checks are the ones the fine-tuned model was weak on: SCTL-only ROC-AUC, Score 3 caught, and non-SCTL
healthy teats wrongly flagged. Results -> reports/teat_model_comparison.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TEAT = Path(__file__).resolve().parents[1]
REPO = TEAT.parent
MANIFEST = TEAT / "curation" / "teat_v2_manifest.csv"
DATA = TEAT / "data" / "processed_v2"
CACHE = TEAT / "data" / "cache" / "frozen_features.npz"
REPORTS = TEAT / "reports"
IMG = 224
C_GRID = [0.01, 0.03, 0.1, 0.3]
N_REPEATS, N_FOLDS = 5, 5


def rows_and_paths():
    rows = list(csv.DictReader(open(MANIFEST)))
    for r in rows:
        r["path"] = DATA / r["split"] / r["label"] / f"{r['id']}.jpg"
    stress = {
        "whole_udder_healthy": sorted((REPO / "udder_images" / "healthy").glob("*")),
        "tick_cow_skin": sorted((REPO / "tick_images" / "infested").glob("*")),
        "hand_in_photo": sorted(p for p in (TEAT / "data" / "quarantined_hand_images").rglob("*") if p.is_file()),
    }
    return rows, stress


def to_input(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img.resize((IMG, IMG), Image.BILINEAR), dtype=np.float32)


def stage_features():
    import keras
    backbone = keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg",
                                              input_shape=(IMG, IMG, 3))
    prep = keras.applications.mobilenet_v2.preprocess_input

    def embed(paths):
        out = []
        for i in range(0, len(paths), 32):
            x = prep(np.stack([to_input(p) for p in paths[i:i + 32]]))
            out.append(backbone.predict(x, verbose=0))
        return np.concatenate(out)

    rows, stress = rows_and_paths()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=embed([r["path"] for r in rows]), ids=np.array([r["id"] for r in rows]),
             **{f"stress_{k}": embed(v) for k, v in stress.items()})
    print(f"cached {len(rows)} teat images + stress sets -> {CACHE}")


def weights(rows, mode):
    """mode 'class': classes equal, each group counts once. 'cell': every (label, source) cell equal."""
    groups = Counter(r["group"] for r in rows)
    w = np.array([1.0 / groups[r["group"]] for r in rows])
    keys = [r["label"] if mode == "class" else (r["label"], r["source"]) for r in rows]
    total = Counter()
    for k, wi in zip(keys, w):
        total[k] += wi
    return np.array([wi / total[k] for k, wi in zip(keys, w)])


def make_model(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit(X, y, w, C):
    m = make_model(C)
    m.fit(X, y, logisticregression__sample_weight=w * len(w))
    return m


def report(y, p, rows):
    from sklearn.metrics import roc_auc_score
    src = np.array([r["source"] for r in rows])
    score = np.array([r["score"] for r in rows])
    out = dict(roc_auc_all=float(roc_auc_score(y, p)))
    s = src == "sctl"
    out["roc_auc_sctl_only"] = float(roc_auc_score(y[s], p[s]))
    for k in ("1", "2", "3", "4"):
        m = score == k
        out[f"flagged_sctl_score_{k}"] = float((p[m] >= 0.5).mean())
    for lab, val in (("healthy", 0), ("problem", 1)):
        m = (src == "web") & (y == val)
        out[f"flagged_web_{lab}"] = float((p[m] >= 0.5).mean())
    out["sensitivity"] = float((p[y == 1] >= 0.5).mean())
    out["specificity"] = float((p[y == 0] < 0.5).mean())
    return {k: round(v, 3) for k, v in out.items()}


def platt(z_oof, y, w):
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(z_oof.reshape(-1, 1), y, sample_weight=w * len(w))
    return float(cal.coef_[0, 0]), float(cal.intercept_[0])


def stage_compare():
    from sklearn.model_selection import StratifiedGroupKFold
    rows, stress = rows_and_paths()
    d = np.load(CACHE)
    assert list(d["ids"]) == [r["id"] for r in rows], "feature cache is stale; rerun --stage features"
    X = d["X"]
    y = np.array([r["label"] == "problem" for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])
    split = np.array([r["split"] for r in rows])
    strat = np.array([f"{r['label']}|{r['source']}" for r in rows])
    dev, test = split != "test", split == "test"
    dev_rows = [r for r, m in zip(rows, dev) if m]
    test_rows = [r for r, m in zip(rows, test) if m]

    results = {}
    for name, mode in (("frozen_logreg", "class"), ("frozen_logreg_cellbal", "cell")):
        w_dev = weights(dev_rows, mode)
        # choose C by grouped CV on train+val (out-of-fold ROC-AUC on SCTL photos + all photos)
        best = None
        for C in C_GRID:
            z = np.zeros((N_REPEATS, dev.sum()))
            for rep in range(N_REPEATS):
                cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=rep)
                for tr, va in cv.split(X[dev], strat[dev], groups[dev]):
                    w_tr = w_dev[tr] / w_dev[tr].sum()
                    z[rep, va] = fit(X[dev][tr], y[dev][tr], w_tr, C).decision_function(X[dev][va])
            oof = report(y[dev], 1 / (1 + np.exp(-z.mean(0))), dev_rows)
            key = oof["roc_auc_sctl_only"] + oof["roc_auc_all"]
            if best is None or key > best[0]:
                best = (key, C, z, oof)
        _, C, z, oof = best
        A, B = platt(z.reshape(-1), np.tile(y[dev], N_REPEATS), np.tile(w_dev, N_REPEATS) / N_REPEATS)
        model = fit(X[dev], y[dev], w_dev, C)
        calib = lambda zz: 1 / (1 + np.exp(-(A * zz + B)))
        p_test = calib(model.decision_function(X[test]))
        res = dict(C=C, calibration=dict(A=A, B=B), cv_oof=report(y[dev], calib(z.mean(0)), dev_rows),
                   test=report(y[test], p_test, test_rows))
        res["stress_flagged_problem"] = {k: round(float((calib(model.decision_function(d[f"stress_{k}"])) >= 0.5).mean()), 3)
                                         for k in stress}
        results[name] = res

    # fine-tuned v2 numbers from evaluate_v2.py, same test split
    ft = json.load(open(REPORTS / "teat_v2_evaluation.json"))
    results["finetuned_v2"] = dict(test=dict(
        roc_auc_all=round(ft["all_test"]["roc_auc"], 3), roc_auc_sctl_only=round(ft["sctl_only"]["roc_auc"], 3),
        **{f"flagged_sctl_score_{k}": ft["flagged_rate_by_kind"][f"sctl_score_{k}"]["flagged"] for k in "1234"},
        flagged_web_healthy=ft["flagged_rate_by_kind"]["web_healthy"]["flagged"],
        flagged_web_problem=ft["flagged_rate_by_kind"]["web_problem"]["flagged"],
        sensitivity=round(ft["all_test"]["sensitivity"], 3), specificity=round(ft["all_test"]["specificity"], 3)),
        stress_flagged_problem={k: ft["stress_test"][v]["flagged_problem"] for k, v in
                                (("whole_udder_healthy", "whole_udder_healthy"), ("tick_cow_skin", "tick_cow_skin"),
                                 ("hand_in_photo", "hand_in_photo_v1_quarantine"))})

    REPORTS.mkdir(exist_ok=True)
    json.dump(results, open(REPORTS / "teat_model_comparison.json", "w"), indent=2)
    cols = ["roc_auc_all", "roc_auc_sctl_only", "flagged_sctl_score_2", "flagged_sctl_score_3",
            "flagged_web_healthy", "flagged_web_problem", "sensitivity", "specificity"]
    print(f"{'TEST':24s}" + "".join(f"{c.replace('flagged_', 'flag ').replace('roc_auc_', 'auc ')[:14]:>15s}" for c in cols))
    for name, r in results.items():
        print(f"{name:24s}" + "".join(f"{r['test'][c]:15.3f}" for c in cols))
    print("\nnon-SCTL healthy flagged in grouped CV on train+val (12 photos, more signal than test's 3):")
    for name in ("frozen_logreg", "frozen_logreg_cellbal"):
        print(f"  {name:24s} {results[name]['cv_oof']['flagged_web_healthy']:.3f}   (C={results[name]['C']})")
    print(f"saved -> {REPORTS / 'teat_model_comparison.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "compare"], required=True)
    stage = ap.parse_args().stage
    stage_features() if stage == "features" else stage_compare()
