"""
Udder pipeline - Model development.

Only 95 confidently-labeled images total (66 train), a much smaller and
much more balanced dataset than teats (1.36:1 vs teats' 28.5:1). Given the
small size, this compares standard approaches (not the severe-imbalance
tools used for teats, which aren't needed here) with 3-fold group CV.
"""
import csv
import numpy as np
from pathlib import Path

import cv2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.pipeline import Pipeline as ImbPipeline

from features import extract_features, load_manifest

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 7
N_SPLITS = 3


def load_features_for_csv(csv_path):
    rows = load_manifest(csv_path)
    X, y, groups = [], [], []
    for row in rows:
        img = cv2.imread(row["filepath"])
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(int(row["label"]))
        groups.append(row["group_id"])
    return np.vstack(X), np.array(y), np.array(groups)


def cv_evaluate(X, y, groups, build_pipeline_fn, use_sample_weight=False):
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, precs, recs, f1s, bal_accs = [], [], [], [], []
    for train_idx, val_idx in skf.split(X, y, groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pipe = build_pipeline_fn()
        if use_sample_weight:
            sw = compute_sample_weight("balanced", y_tr)
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        accs.append(accuracy_score(y_val, preds))
        precs.append(precision_score(y_val, preds, zero_division=0))
        recs.append(recall_score(y_val, preds, zero_division=0))
        f1s.append(f1_score(y_val, preds, zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_val, preds))
    return {
        "acc_mean": np.mean(accs), "recall_mean": np.mean(recs),
        "prec_mean": np.mean(precs), "f1_mean": np.mean(f1s),
        "bal_acc_mean": np.mean(bal_accs), "recall_std": np.std(recs),
    }


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    print("Loading features for train split...")
    X_train, y_train, g_train = load_features_for_csv(SPLITS_DIR / "train.csv")
    print(f"Train: {len(y_train)} images ({(y_train==0).sum()} healthy, {(y_train==1).sum()} mastitis), "
          f"{len(set(g_train))} unique groups")

    candidates = {
        "RandomForest_class_weight": (lambda: ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)),
        ]), False),
        "GradientBoosting_sample_weight": (lambda: ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE)),
        ]), True),
        "SVM_rbf_class_weight": (lambda: ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        ]), False),
    }

    results = []
    print("\n=== Model comparison (95 images, 1.36:1 imbalance) ===")
    for name, (build_fn, use_sw) in candidates.items():
        metrics = cv_evaluate(X_train, y_train, g_train, build_fn, use_sample_weight=use_sw)
        metrics["strategy"] = name
        results.append(metrics)
        print(f"{name:32s} acc={metrics['acc_mean']:.3f} bal_acc={metrics['bal_acc_mean']:.3f} "
              f"recall={metrics['recall_mean']:.3f} f1={metrics['f1_mean']:.3f}")

    results.sort(key=lambda r: (r["bal_acc_mean"], r["f1_mean"]), reverse=True)
    print(f"\nBest model (by balanced accuracy): {results[0]['strategy']}")

    with open(REPORTS_DIR / "model_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "acc_mean", "bal_acc_mean", "prec_mean", "recall_mean", "recall_std", "f1_mean"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved -> {REPORTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
