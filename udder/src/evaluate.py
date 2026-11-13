"""
Teats pipeline - Evaluate final model on held-out test set.
"""
import json
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score,
)

from model_development import load_features_for_csv

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    pipeline = joblib.load(MODELS_DIR / "udder_svm_model.joblib")
    threshold = json.load(open(MODELS_DIR / "decision_threshold.json"))["threshold"]

    X_test, y_test, _ = load_features_for_csv(SPLITS_DIR / "test.csv")
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_test, preds)
    report_text = classification_report(y_test, preds, target_names=["healthy", "mastitis"])

    summary = (
        f"UDDER MODEL - TEST SET RESULTS (n={len(y_test)}, threshold={threshold:.2f})\n\n"
        f"Accuracy:          {acc:.3f}\n"
        f"Balanced Accuracy: {bal_acc:.3f}  <- the metric that matters given 1.36:1 imbalance\n"
        f"Precision:         {precision:.3f}\n"
        f"Recall:            {recall:.3f}\n"
        f"F1 score:          {f1:.3f}\n"
        f"ROC-AUC:           {auc:.3f}\n\n"
        f"{report_text}\n"
        f"Confusion matrix (rows=actual, cols=predicted, order=[healthy, mastitis]):\n{cm}\n"
    )
    print(summary)

    with open(REPORTS_DIR / "test_metrics.txt", "w") as f:
        f.write(summary)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["healthy", "mastitis"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["healthy", "mastitis"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Udder Model - Test Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved -> {REPORTS_DIR / 'test_metrics.txt'} and confusion_matrix.png")


if __name__ == "__main__":
    main()
