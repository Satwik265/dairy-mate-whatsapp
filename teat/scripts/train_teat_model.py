"""
train_teat_model.py

Trains a healthy-vs-mastitis teat classifier using transfer learning
(MobileNetV2 backbone, pretrained on ImageNet, fine-tuned on our data).

Assumes data/processed/{train,val,test}/{healthy,mastitis} already exists
(run scripts/prepare_dataset.py first).

Run:
    python scripts/train_teat_model.py
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


def build_datasets(data_dir):
    print("Image size:", IMG_SIZE)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=True, seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False,
    )

    # class_names[0] is alphabetically first -> "healthy", [1] -> "mastitis"
    class_names = train_ds.class_names
    print("Class order (0/1):", class_names)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_dir):
    """Simple count-based class weighting: weight_i = n_total / (n_classes * n_i)."""
    counts = {}
    for cls in sorted(os.listdir(train_dir)):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))

    classes_sorted = sorted(counts.keys())  # matches class_names order
    n_total = sum(counts.values())
    n_classes = len(counts)
    weights = {
        i: n_total / (n_classes * counts[cls])
        for i, cls in enumerate(classes_sorted)
    }
    print("Class counts:", counts)
    print("Class weights:", weights)
    return weights


def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="augmentation")


def build_model():
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # freeze for stage 1

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = build_augmentation()(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs_head", type=int, default=15)
    parser.add_argument("--epochs_finetune", type=int, default=10)
    parser.add_argument("--out_model", default="models/teat_classifier.keras")
    parser.add_argument("--history_out", default="reports/teat_training_history.json")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Square input size; larger keeps more teat-end detail (photos are ~450-600px)")
    args = parser.parse_args()

    global IMG_SIZE
    IMG_SIZE = (args.img_size, args.img_size)

    train_ds, val_ds, test_ds, class_names = build_datasets(args.data_dir)
    class_weights = compute_class_weights(os.path.join(args.data_dir, "train"))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model, base = build_model()

    # ---- Stage 1: train the classification head only ----
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            args.out_model, monitor="val_loss", save_best_only=True),
    ]

    print("\n=== Stage 1: training classification head (backbone frozen) ===")
    history_head = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs_head, class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---- Stage 2: unfreeze top layers of backbone and fine-tune at low LR ----
    base.trainable = True
    for layer in base.layers[:-30]:  # keep most of the backbone frozen, unfreeze last ~30 layers
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    print("\n=== Stage 2: fine-tuning top backbone layers ===")
    history_finetune = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs_finetune, class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---- Final test evaluation ----
    print("\n=== Test set evaluation ===")
    test_results = model.evaluate(test_ds, return_dict=True)
    print(test_results)

    os.makedirs("reports", exist_ok=True)
    combined_history = {
        "head": {k: [float(v) for v in vals] for k, vals in history_head.history.items()},
        "finetune": {k: [float(v) for v in vals] for k, vals in history_finetune.history.items()},
        "test_results": {k: float(v) for k, v in test_results.items()},
        "class_names": class_names,
    }
    with open(args.history_out, "w") as f:
        json.dump(combined_history, f, indent=2)

    model.save(args.out_model)
    print(f"\nModel saved to {args.out_model}")
    print(f"Training history + test metrics saved to {args.history_out}")


if __name__ == "__main__":
    main()
