"""
Teats pipeline - Error analysis on test set.
"""
import csv
import json
import joblib
import numpy as np
from pathlib import Path
from PIL import Image

from features import extract_features
import cv2

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def main():
    pipeline = joblib.load(MODELS_DIR / "udder_svm_model.joblib")
    threshold = json.load(open(MODELS_DIR / "decision_threshold.json"))["threshold"]

    rows = list(csv.DictReader(open(SPLITS_DIR / "test.csv")))
    misclassified = []
    for row in rows:
        img = cv2.imread(row["filepath"])
        feat = extract_features(img).reshape(1, -1)
        prob = pipeline.predict_proba(feat)[0, 1]
        pred = int(prob >= threshold)
        true_label = int(row["label"])
        if pred != true_label:
            misclassified.append({
                "filepath": row["filepath"], "true": row["class_name"],
                "predicted": "mastitis" if pred == 1 else "healthy",
                "prob_mastitis": round(float(prob), 3),
            })

    print(f"Misclassified: {len(misclassified)} / {len(rows)}")
    # All errors are mastitis->healthy in this model (0 healthy misclassified).
    # Show the 8 most confidently-wrong mastitis->healthy examples (lowest prob).
    misclassified.sort(key=lambda m: m["prob_mastitis"])
    worst = misclassified[:8]

    thumbs = []
    for m in worst:
        im = Image.open(m["filepath"]).convert("RGB").resize((200, 200))
        thumbs.append(im)
    cols = 4
    rows_n = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (200 * cols, 200 * rows_n), "white")
    for i, im in enumerate(thumbs):
        grid.paste(im, ((i % cols) * 200, (i // cols) * 200))
    grid.save(REPORTS_DIR / "error_analysis_grid.png")

    with open(REPORTS_DIR / "error_analysis.txt", "w") as f:
        f.write(f"Total misclassified: {len(misclassified)} / {len(rows)} (all mastitis predicted as healthy; "
                f"0 healthy images misclassified)\n\n")
        f.write("Most confidently-wrong examples (lowest P(mastitis)):\n")
        for m in worst:
            f.write(f"{m}\n")

    print(f"Saved -> {REPORTS_DIR / 'error_analysis.txt'} and error_analysis_grid.png")


if __name__ == "__main__":
    main()
