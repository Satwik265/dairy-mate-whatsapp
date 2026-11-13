"""
Compare udder classifiers on the curated Roboflow v16 crops (see curate_roboflow.py).

    python src/compare_models.py --stage features   # cache features for every crop
    python src/compare_models.py --stage compare    # repeated grouped CV on train+val, one final test check

Models compared (same folds, same preprocessing):
    shortcut_only       logistic regression on crop size / aspect / greyscale / black corners.
                        Not a real model: it shows how much can be scored WITHOUT looking at the udder.
    handcrafted_svm     the teammate's 223 handcrafted features (features.py) + RBF SVM
    mobilenet_logreg    frozen ImageNet MobileNetV2 embeddings (1280-d) + logistic regression
    mobilenet_svm       frozen ImageNet MobileNetV2 embeddings + RBF SVM

Fairness controls:
    - Resolution matching: every crop is first downscaled so its short side is at most DETAIL_PX,
      because mastitis crops are ~2x sharper than healthy ones in this dataset.
    - Greyscale and black-corner crops are dropped (both are far more common in the mastitis class).
    - Samples are weighted 1 / (crops in their source group) for training AND metrics, so one stock
      video with 59 frames counts like one photo.
    - Folds are grouped by source photo (StratifiedGroupKFold), repeated with different seeds.
"""

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# sklearn 1.9 deprecates SVC(probability=True); it still works and keeps this comparison simple
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "curation" / "rf16_clean_manifest.csv"
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"

DETAIL_PX = 112          # just above the 10th percentile of healthy crop size (106 px)
N_REPEATS = 10
N_FOLDS = 5
TARGET_SPECIFICITY = 0.90


def load_rows():
    rows = list(csv.DictReader(open(MANIFEST)))
    kept = [r for r in rows if r["grayscale"] == "0" and r["black_corners"] == "0"]
    print(f"crops: {len(rows)} -> {len(kept)} after dropping greyscale/black-corner crops")
    return kept


def detail_matched(path):
    """Open a crop and cap its short side at DETAIL_PX so both classes carry similar detail."""
    img = Image.open(ROOT / path).convert("RGB")
    s = min(img.size)
    if s > DETAIL_PX:
        img = img.resize((round(img.width * DETAIL_PX / s), round(img.height * DETAIL_PX / s)),
                         Image.LANCZOS)
    return img


# ---------- stage: features ----------

def stage_features():
    import cv2
    import tensorflow as tf
    from features import extract_features   # teammate's handcrafted features

    rows = load_rows()
    CACHE.mkdir(parents=True, exist_ok=True)
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                                 pooling="avg", input_shape=(224, 224, 3))
    prep = tf.keras.applications.mobilenet_v2.preprocess_input

    hand, cnn, shortcut = [], [], []
    for i, r in enumerate(rows):
        orig = Image.open(ROOT / r["crop_path"])
        w, h = orig.size
        shortcut.append([np.log(w), np.log(h), np.log(w / h),
                         float(r["grayscale"]), float(r["black_corners"])])
        img = detail_matched(r["crop_path"])
        hand.append(extract_features(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)))
        x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32)[None]
        cnn.append(backbone(prep(x), training=False).numpy()[0])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}")

    np.savez(CACHE / "features.npz", hand=np.array(hand), cnn=np.array(cnn),
             shortcut=np.array(shortcut), box_id=np.array([r["box_id"] for r in rows]))
    print(f"cached features for {len(rows)} crops -> {CACHE / 'features.npz'}")


# ---------- stage: compare ----------

def build(name):
    if name == "shortcut_only":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "handcrafted_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                   class_weight="balanced", probability=True))
    if name == "mobilenet_logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced",
                                                                  max_iter=5000))
    if name == "mobilenet_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                   class_weight="balanced", probability=True))
    raise ValueError(name)


FEATURE_OF = {"shortcut_only": "shortcut", "handcrafted_svm": "hand",
              "mobilenet_logreg": "cnn", "mobilenet_svm": "cnn"}


def metrics(y, p, w):
    fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
    sens_at_spec = float(np.interp(1 - TARGET_SPECIFICITY, fpr, tpr))
    pred = (p >= 0.5).astype(int)
    sens = np.sum(w * (pred == 1) * (y == 1)) / np.sum(w * (y == 1))
    spec = np.sum(w * (pred == 0) * (y == 0)) / np.sum(w * (y == 0))
    return dict(roc_auc=roc_auc_score(y, p, sample_weight=w),
                pr_auc=average_precision_score(y, p, sample_weight=w),
                bal_acc=(sens + spec) / 2, sensitivity=sens, specificity=spec,
                sens_at_90spec=sens_at_spec)


def fit(model, X, y, w):
    model.fit(X, y, **{model.steps[-1][0] + "__sample_weight": w})
    return model


def group_bootstrap(y, p, w, groups, n=2000, seed=0):
    """95% interval for ROC-AUC, resampling whole source groups."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx], sample_weight=w[idx]))
    return np.percentile(vals, [2.5, 97.5])


def stage_compare():
    rows = load_rows()
    data = np.load(CACHE / "features.npz")
    assert list(data["box_id"]) == [r["box_id"] for r in rows], "feature cache is stale; rerun --stage features"

    y = np.array([r["label"] == "mastitis" for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])
    split = np.array([r["split"] for r in rows])
    gsize = Counter(groups)
    w = np.array([1.0 / gsize[g] for g in groups])

    dev = split != "test"            # train + val used for cross-validation
    test = split == "test"
    print(f"dev: {dev.sum()} crops, {len(set(groups[dev & (y == 1)]))} mastitis / "
          f"{len(set(groups[dev & (y == 0)]))} healthy source groups")
    print(f"test: {test.sum()} crops, {len(set(groups[test & (y == 1)]))} mastitis / "
          f"{len(set(groups[test & (y == 0)]))} healthy source groups\n")

    results = {}
    for name, feat in FEATURE_OF.items():
        X = data[feat]
        per_repeat = []
        for rep in range(N_REPEATS):
            oof = np.zeros(dev.sum())
            Xd, yd, gd, wd = X[dev], y[dev], groups[dev], w[dev]
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
            for tr, va in cv.split(Xd, yd, gd):
                oof[va] = fit(build(name), Xd[tr], yd[tr], wd[tr]).predict_proba(Xd[va])[:, 1]
            per_repeat.append(metrics(yd, oof, wd))
        summary = {k: (np.mean([m[k] for m in per_repeat]),
                       np.min([m[k] for m in per_repeat]),
                       np.max([m[k] for m in per_repeat])) for k in per_repeat[0]}

        # one final look at the held-out test split, trained on all of train+val
        model = fit(build(name), X[dev], y[dev], w[dev])
        pt = model.predict_proba(X[test])[:, 1]
        tm = metrics(y[test], pt, w[test])
        tm["roc_auc_ci95"] = list(group_bootstrap(y[test], pt, w[test], groups[test]))
        results[name] = dict(cv=summary, test=tm)

        print(f"{name:17s} CV ROC-AUC {summary['roc_auc'][0]:.3f} [{summary['roc_auc'][1]:.3f}-"
              f"{summary['roc_auc'][2]:.3f}]  PR-AUC {summary['pr_auc'][0]:.3f}  "
              f"bal-acc {summary['bal_acc'][0]:.3f}  sens@90%spec {summary['sens_at_90spec'][0]:.3f}  |  "
              f"TEST ROC-AUC {tm['roc_auc']:.3f} (95% CI {tm['roc_auc_ci95'][0]:.2f}-"
              f"{tm['roc_auc_ci95'][1]:.2f})")

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "rf16_model_comparison.json"
    json.dump(dict(settings=dict(detail_px=DETAIL_PX, repeats=N_REPEATS, folds=N_FOLDS,
                                 weighting="1/group size", dropped="greyscale, black corners"),
                   results=results), open(out, "w"), indent=2, default=float)
    print(f"\nCV numbers are mean [min-max] over {N_REPEATS} repeats of {N_FOLDS}-fold grouped CV.")
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "compare"], required=True)
    args = ap.parse_args()
    stage_features() if args.stage == "features" else stage_compare()
