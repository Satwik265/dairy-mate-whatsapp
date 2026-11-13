"""
evaluate.py

Loads the trained teat classifier and the held-out test set, then produces:
    - per-class precision / recall / F1
    - confusion matrix
    - a saved PNG of the confusion matrix
    - reports/teat_evaluation_report.json

Run:
    python scripts/evaluate.py
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_path", default="models/teat_classifier.keras")
    parser.add_argument("--out_report", default="reports/teat_evaluation_report.json")
    parser.add_argument("--out_confmat_png", default="reports/teat_confusion_matrix.png")
    args = parser.parse_args()

    test_dir = os.path.join(args.data_dir, "test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False,
    )
    class_names = test_ds.class_names  # e.g. ['healthy', 'mastitis']

    model = tf.keras.models.load_model(args.model_path)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).astype(int).flatten()
    y_prob = model.predict(test_ds).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("=== Confusion matrix (rows=true, cols=predicted) ===")
    print(class_names)
    print(cm)

    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    with open(args.out_report, "w") as f:
        json.dump({
            "class_names": class_names,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Teat Classifier — Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out_confmat_png, dpi=150)
    print(f"\nSaved report to {args.out_report}")
    print(f"Saved confusion matrix plot to {args.out_confmat_png}")


if __name__ == "__main__":
    main()
