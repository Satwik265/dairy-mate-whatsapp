"""
Udder pipeline - Hyperparameter tuning for SVM (winning model from
model_development.py).
"""
import csv
import json
from itertools import product
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline

from model_development import load_features_for_csv, cv_evaluate

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 7


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X_train, y_train, g_train = load_features_for_csv(SPLITS_DIR / "train.csv")

    param_grid = {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "kernel": ["rbf", "linear"],
    }
    combos = list(product(param_grid["C"], param_grid["gamma"], param_grid["kernel"]))
    print(f"Tuning {len(combos)} combinations...")

    log = []
    for C, gamma, kernel in combos:
        if kernel == "linear" and gamma != "scale":
            continue  # gamma irrelevant for linear kernel, skip redundant combos

        def build_fn(C=C, gamma=gamma, kernel=kernel):
            return ImbPipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=C, gamma=gamma, kernel=kernel, class_weight="balanced",
                             probability=True, random_state=RANDOM_STATE)),
            ])
        metrics = cv_evaluate(X_train, y_train, g_train, build_fn, use_sample_weight=False)
        metrics.update({"C": C, "gamma": str(gamma), "kernel": kernel})
        log.append(metrics)
        print(f"C={C:4.1f} gamma={str(gamma):6s} kernel={kernel:6s} | "
              f"bal_acc={metrics['bal_acc_mean']:.3f} recall={metrics['recall_mean']:.3f} f1={metrics['f1_mean']:.3f}")

    log.sort(key=lambda r: (r["bal_acc_mean"], r["f1_mean"]), reverse=True)
    best = log[0]
    print(f"\nBest: C={best['C']}, gamma={best['gamma']}, kernel={best['kernel']} (bal_acc={best['bal_acc_mean']:.3f})")

    with open(REPORTS_DIR / "hyperparameter_tuning_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "C", "gamma", "kernel", "acc_mean", "bal_acc_mean", "prec_mean", "recall_mean", "recall_std", "f1_mean",
        ])
        writer.writeheader()
        writer.writerows(log)

    with open(MODELS_DIR / "best_config.json", "w") as f:
        json.dump({"C": best["C"], "gamma": best["gamma"], "kernel": best["kernel"]}, f, indent=2)
    print(f"Saved -> {REPORTS_DIR / 'hyperparameter_tuning_log.csv'}")
    print(f"Saved -> {MODELS_DIR / 'best_config.json'}")


if __name__ == "__main__":
    main()
