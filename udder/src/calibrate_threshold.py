"""
Teats pipeline - Threshold calibration on validation set only.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

from model_development import load_features_for_csv

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"


def main():
    pipeline = joblib.load(MODELS_DIR / "udder_svm_model.joblib")
    X_val, y_val, _ = load_features_for_csv(SPLITS_DIR / "val.csv")
    probs = pipeline.predict_proba(X_val)[:, 1]

    results = []
    for t in np.arange(0.20, 0.85, 0.02):
        preds = (probs >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_val, preds)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        results.append((t, bal_acc, rec, f1))

    max_score = max(r[1] for r in results)
    tied = [r for r in results if abs(r[1] - max_score) < 1e-9]
    best_t, best_score, _, _ = min(tied, key=lambda r: abs(r[0] - 0.5))

    print("threshold | balanced_acc | recall | f1")
    for t, bal_acc, rec, f1 in results:
        marker = "  <-- selected" if abs(t - best_t) < 1e-9 else ""
        print(f"{t:.2f}      | {bal_acc:.3f}        | {rec:.3f}  | {f1:.3f}{marker}")

    print(f"\nSelected threshold: {best_t:.2f} (balanced_acc={best_score:.3f})")
    with open(MODELS_DIR / "decision_threshold.json", "w") as f:
        json.dump({"threshold": float(best_t)}, f)
    print(f"Saved -> {MODELS_DIR / 'decision_threshold.json'}")


if __name__ == "__main__":
    main()
