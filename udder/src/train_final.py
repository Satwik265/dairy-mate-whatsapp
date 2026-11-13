"""
Udder pipeline - Train final model: SVM with tuned hyperparameters.
"""
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

from model_development import load_features_for_csv

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 7


def main():
    config = json.load(open(MODELS_DIR / "best_config.json"))
    gamma = config["gamma"]
    if gamma not in ("scale", "auto"):
        gamma = float(gamma)
    print(f"Using config: C={config['C']}, gamma={gamma}, kernel={config['kernel']}")

    X_train, y_train, _ = load_features_for_csv(SPLITS_DIR / "train.csv")
    X_val, y_val, _ = load_features_for_csv(SPLITS_DIR / "val.csv")
    print(f"Train: {len(y_train)} ({(y_train==0).sum()} healthy, {(y_train==1).sum()} mastitis)")
    print(f"Val:   {len(y_val)} ({(y_val==0).sum()} healthy, {(y_val==1).sum()} mastitis)")

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=config["C"], gamma=gamma, kernel=config["kernel"],
                     class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    val_preds = pipeline.predict(X_val)
    print("\nValidation sanity check:")
    print(f"  accuracy={accuracy_score(y_val, val_preds):.3f} "
          f"balanced_acc={balanced_accuracy_score(y_val, val_preds):.3f} "
          f"precision={precision_score(y_val, val_preds, zero_division=0):.3f} "
          f"recall={recall_score(y_val, val_preds, zero_division=0):.3f} "
          f"f1={f1_score(y_val, val_preds, zero_division=0):.3f}")

    model_path = MODELS_DIR / "udder_svm_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved -> {model_path}")


if __name__ == "__main__":
    main()
