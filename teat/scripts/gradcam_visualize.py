"""
gradcam_visualize.py

Generates Grad-CAM heatmaps for the trained teat classifier, so you can see
WHICH PIXELS the model is actually using to make its predictions. This is the
key sanity check for whether the model learned real teat/udder features vs.
a shortcut (e.g. background, lighting, or source-dataset artifacts).

For each sampled image, produces:
    - a heatmap overlay showing which regions push the prediction toward "mastitis"
    - saved individually under reports/gradcam/<true_label>/
    - plus one combined contact-sheet grid for a fast visual scan

WHAT TO LOOK FOR:
    - Good sign: hot regions concentrated on the teat/teat-end itself.
    - Red flag: hot regions on background, cow's leg, lighting glare, image
      borders, or anything that isn't the teat — that means the model may be
      keying off something other than actual teat health.

Run (after training):
    python scripts/gradcam_visualize.py

Optional flags:
    python scripts/gradcam_visualize.py --num_per_class 8 --split test
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMG_SIZE = (224, 224)


# --------------------------------------------------------------------------
# Model surgery: split the trained model into
#   (1) feature_extractor: raw-preprocessed image -> last conv feature map
#   (2) head_model:        last conv feature map -> final prediction
# Splitting this way sidesteps any ambiguity around gradients through a
# nested submodel (the MobileNetV2 backbone is stored as a single layer
# inside the outer model) and works regardless of Keras version quirks.
# --------------------------------------------------------------------------

def find_backbone(model):
    """Find the nested CNN backbone (e.g. MobileNetV2) inside the outer model.

    Note: the augmentation block (built as a Sequential) is ALSO a tf.keras.Model
    instance, and it sits earlier in model.layers than the real backbone. So we
    can't just grab the first nested Model — instead pick the one with the most
    sublayers, since a real CNN backbone has 100+ layers vs. ~5 for augmentation.
    """
    candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if not candidates:
        raise ValueError(
            "Could not find a nested backbone model inside the loaded model. "
            "This script expects the architecture produced by train_teat_model.py."
        )
    backbone = max(candidates, key=lambda l: len(l.layers))
    return backbone


def find_last_conv_layer_name(backbone):
    """Auto-detect the last layer with a 4D (spatial) output — works across backbones."""
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("Could not find a 4D conv-output layer in the backbone.")


def build_feature_extractor_and_head(model):
    backbone = find_backbone(model)
    last_conv_name = find_last_conv_layer_name(backbone)
    print(f"Backbone: {backbone.name}  ({len(backbone.layers)} layers)   Last conv layer: {last_conv_name}")

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer(last_conv_name).output,
        name="feature_extractor",
    )

    # Reuse the *trained* head layers (GAP -> Dropout -> Dense) in a small
    # standalone model that takes the conv feature map as input.
    backbone_index = model.layers.index(backbone)
    head_layers = model.layers[backbone_index + 1:]

    feat_shape = feature_extractor.output.shape[1:]
    feat_input = tf.keras.Input(shape=feat_shape, name="feat_input")
    x = feat_input
    used_layers = []
    for layer in head_layers:
        # GlobalAveragePooling2D / Dropout / Dense are all safe to reapply here;
        # skip anything unexpected rather than risk an incorrect graph.
        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D,
                              tf.keras.layers.GlobalMaxPooling2D)):
            x = layer(x)
            used_layers.append(layer.name)
        elif isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)
            used_layers.append(layer.name)
        elif isinstance(layer, tf.keras.layers.Dense):
            x = layer(x)
            used_layers.append(layer.name)

    head_model = tf.keras.Model(inputs=feat_input, outputs=x, name="head_model")
    print("Head layers reused (in order):", used_layers)
    return feature_extractor, head_model


# --------------------------------------------------------------------------
# Grad-CAM core
# --------------------------------------------------------------------------

def compute_gradcam(raw_img_224, feature_extractor, head_model):
    """raw_img_224: float32 array, shape (224,224,3), pixel values 0-255 (unpreprocessed)."""
    batch = np.expand_dims(raw_img_224, axis=0).astype("float32")
    preprocessed = preprocess_input(batch)

    conv_output = feature_extractor(preprocessed)

    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        prediction = head_model(conv_output, training=False)
        # Single sigmoid output: class 1 = mastitis. Using the raw score as
        # the Grad-CAM target shows what drives the prediction toward mastitis,
        # for both healthy and mastitis images alike.
        target_score = prediction[:, 0]

    grads = tape.gradient(target_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy(), float(prediction.numpy()[0, 0])


def overlay_heatmap(raw_img_224, heatmap, alpha=0.45):
    """raw_img_224: uint8 array (224,224,3). heatmap: float array (h,w) in [0,1]."""
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype("uint8")).resize(IMG_SIZE, Image.BILINEAR)
    )
    try:
        jet = matplotlib.colormaps["jet"]  # matplotlib >= 3.9
    except AttributeError:
        jet = cm.get_cmap("jet")  # older matplotlib fallback
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = (jet_heatmap * 255).astype("uint8")

    base = raw_img_224.astype("uint8")
    overlay = (base * (1 - alpha) + jet_heatmap * alpha).astype("uint8")
    return overlay


# --------------------------------------------------------------------------
# Dataset walking + orchestration
# --------------------------------------------------------------------------

def load_image_224(path):
    im = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.array(im).astype("float32")


def gather_samples(split_dir):
    """Return list of (path, true_label_str) across class subfolders."""
    samples = []
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            samples.append((os.path.join(cls_dir, fname), cls))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/teat_classifier.keras")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_per_class", type=int, default=6,
                         help="How many CORRECT examples per class to visualize (all misclassified ones are always included).")
    parser.add_argument("--out_dir", default="reports/gradcam")
    args = parser.parse_args()

    global IMG_SIZE
    model = tf.keras.models.load_model(args.model_path)
    IMG_SIZE = tuple(model.input_shape[1:3])
    feature_extractor, head_model = build_feature_extractor_and_head(model)

    split_dir = os.path.join(args.data_dir, args.split)
    samples = gather_samples(split_dir)
    if not samples:
        print(f"No images found under {split_dir}. Did you run prepare_dataset.py?")
        return

    class_names = sorted({label for _, label in samples})  # alphabetical, matches training
    print(f"Classes found: {class_names}  |  {len(samples)} images in '{args.split}' split")

    correct_by_class = {c: [] for c in class_names}
    misclassified = []

    for path, true_label in samples:
        raw_img = load_image_224(path)
        batch = np.expand_dims(raw_img, axis=0)
        prob = float(model.predict(batch, verbose=0)[0, 0])
        pred_label = class_names[1] if prob >= 0.5 else class_names[0]
        record = {"path": path, "true_label": true_label, "pred_label": pred_label, "prob": prob}
        if pred_label == true_label:
            correct_by_class[true_label].append(record)
        else:
            misclassified.append(record)

    print(f"Misclassified in '{args.split}' split: {len(misclassified)} / {len(samples)}")

    to_visualize = []
    for c in class_names:
        to_visualize.extend(correct_by_class[c][:args.num_per_class])
    to_visualize.extend(misclassified)  # always include ALL misclassified examples

    os.makedirs(args.out_dir, exist_ok=True)
    grid_thumbs = []

    for i, rec in enumerate(to_visualize):
        raw_img = load_image_224(rec["path"])
        heatmap, prob = compute_gradcam(raw_img, feature_extractor, head_model)
        overlay = overlay_heatmap(raw_img, heatmap)

        status = "correct" if rec["pred_label"] == rec["true_label"] else "MISCLASSIFIED"
        fname = f"{status}_{rec['true_label']}_pred-{rec['pred_label']}_{prob:.2f}_{i:03d}.png"
        out_subdir = os.path.join(args.out_dir, rec["true_label"])
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, fname)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        axes[0].imshow(raw_img.astype("uint8"))
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM (mastitis score={prob:.2f})", fontsize=9)
        axes[1].axis("off")
        fig.suptitle(f"True: {rec['true_label']}  |  Predicted: {rec['pred_label']}  |  {status}",
                     fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        grid_thumbs.append((out_path, rec, status))
        print(f"  saved {out_path}")

    build_contact_sheet(grid_thumbs, os.path.join(args.out_dir, "gradcam_summary_grid.png"))
    print(f"\nDone. {len(to_visualize)} Grad-CAM visualizations saved under {args.out_dir}/")
    print(f"Summary grid: {os.path.join(args.out_dir, 'gradcam_summary_grid.png')}")


def build_contact_sheet(grid_thumbs, out_path, max_items=24):
    items = grid_thumbs[:max_items]
    if not items:
        return
    n = len(items)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for ax, (path, rec, status) in zip(axes, items):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis("off")
        color = "red" if status == "MISCLASSIFIED" else "black"
        ax.set_title(f"{rec['true_label']} -> {rec['pred_label']}", fontsize=8, color=color)

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
