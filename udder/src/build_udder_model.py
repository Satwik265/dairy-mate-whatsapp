"""
Build the deployable udder classifier: frozen MobileNetV2 features + logistic regression,
calibrated, folded into ONE Keras file.

    python src/build_udder_model.py --stage features   # embed every curated crop with the serving pipeline
    python src/build_udder_model.py --stage evaluate   # grouped CV on train+val, calibration, confidence bands, test check
    python src/build_udder_model.py --stage export     # fit on ALL data, fold into models/udder_classifier.keras, parity + timing
    python src/build_udder_model.py --stage gradcam    # heatmaps on test crops -> reports/udder_gradcam.jpg

Serving contract (what the app must do before calling the model):
    open photo -> RGB -> stretch-resize to 224x224 (bilinear) -> float32 array of raw 0-255 values.
    Everything else (detail matching, MobileNet preprocessing, classifier, calibration) is inside the model.
    Output: one number = calibrated P(mastitis), assuming equal prior odds of healthy vs mastitis.
"""

import argparse
import csv
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "curation" / "rf16_clean_manifest.csv"
CACHE = ROOT / "data" / "cache" / "serving_features.npz"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

INPUT = 224
DETAIL_PX = 112          # crops are squeezed to this detail level inside the model (resolution matching)
LR_C = 0.05
N_REPEATS, N_FOLDS = 10, 5
BANDS = [("high", 0.85), ("medium", 0.65), ("low", 0.0)]   # on confidence = max(p, 1 - p)


# ---------- data ----------

def load_rows():
    rows = list(csv.DictReader(open(MANIFEST)))
    return [r for r in rows if r["grayscale"] == "0" and r["black_corners"] == "0"]


def to_input(img):
    """The serving contract: RGB, stretched to 224x224, raw 0-255 float32."""
    return np.asarray(img.convert("RGB").resize((INPUT, INPUT), Image.BILINEAR), dtype=np.float32)


def arrays(rows):
    y = np.array([r["label"] == "mastitis" for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])
    size = Counter(groups)
    w = np.array([1.0 / size[g] for g in groups])     # each source photo counts once
    return y, groups, w


# ---------- model pieces ----------

def feature_model():
    """224x224 raw RGB -> 1280-d MobileNetV2 embedding, with resolution matching built in."""
    import keras
    from keras import layers
    inp = keras.Input((INPUT, INPUT, 3), name="image_0_255")
    x = layers.Resizing(DETAIL_PX, DETAIL_PX, interpolation="area", name="detail_match_down")(inp)
    x = layers.Resizing(INPUT, INPUT, interpolation="bilinear", name="detail_match_up")(x)
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    backbone = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              pooling="avg", input_shape=(INPUT, INPUT, 3))
    backbone.trainable = False
    out = backbone(x, training=False)
    return keras.Model(inp, out, name="udder_features")


def make_classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, class_weight="balanced", max_iter=5000))


def fit_classifier(X, y, w):
    clf = make_classifier()
    clf.fit(X, y, logisticregression__sample_weight=w)
    return clf


def logit_of(clf, X):
    return clf.decision_function(X)


def fit_platt(logits, y, w):
    """1-D logistic calibration on out-of-fold logits; balanced so the prior is 50/50."""
    from sklearn.linear_model import LogisticRegression
    bal = np.where(y == 1, 0.5 / w[y == 1].sum(), 0.5 / w[y == 0].sum()) * w
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(logits.reshape(-1, 1), y, sample_weight=bal)
    return float(cal.coef_[0, 0]), float(cal.intercept_[0])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def band_of(p):
    conf = np.maximum(p, 1 - p)
    return np.array([next(name for name, lo in BANDS if c >= lo) for c in conf])


def band_table(y, p, w):
    """For each confidence band: share of photos and how often the verdict was right."""
    bands, correct = band_of(p), ((p >= 0.5).astype(int) == y)
    out = {}
    for name, _ in BANDS:
        m = bands == name
        out[name] = dict(share=float(w[m].sum() / w.sum()),
                         accuracy=float((w[m] * correct[m]).sum() / w[m].sum()) if m.any() else None,
                         n_crops=int(m.sum()))
    return out


def metrics(y, p, w):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    pred = (p >= 0.5).astype(int)
    sens = (w * (pred == 1) * (y == 1)).sum() / (w * (y == 1)).sum()
    spec = (w * (pred == 0) * (y == 0)).sum() / (w * (y == 0)).sum()
    return dict(roc_auc=float(roc_auc_score(y, p, sample_weight=w)), sensitivity=float(sens),
                specificity=float(spec), balanced_accuracy=float((sens + spec) / 2),
                brier=float(brier_score_loss(y, p, sample_weight=w)))


def oof_logits(X, y, groups, w, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    out = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
        out[va] = logit_of(fit_classifier(X[tr], y[tr], w[tr]), X[va])
    return out


# ---------- stages ----------

def stage_features():
    rows = load_rows()
    fm = feature_model()
    X = []
    for i in range(0, len(rows), 32):
        batch = np.stack([to_input(Image.open(ROOT / r["crop_path"])) for r in rows[i:i + 32]])
        X.append(fm.predict(batch, verbose=0))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=np.concatenate(X), box_id=np.array([r["box_id"] for r in rows]))
    print(f"embedded {len(rows)} crops -> {CACHE}")


def load_features(rows):
    d = np.load(CACHE)
    assert list(d["box_id"]) == [r["box_id"] for r in rows], "feature cache is stale; rerun --stage features"
    return d["X"]


def stage_evaluate():
    rows = load_rows()
    X = load_features(rows)
    y, groups, w = arrays(rows)
    split = np.array([r["split"] for r in rows])
    dev, test = split != "test", split == "test"

    # 1) repeated grouped CV on train+val -> raw out-of-fold logits
    reps = [oof_logits(X[dev], y[dev], groups[dev], w[dev], seed) for seed in range(N_REPEATS)]
    # 2) calibration fitted on the pooled out-of-fold logits
    A, B = fit_platt(np.concatenate(reps), np.tile(y[dev], N_REPEATS), np.tile(w[dev], N_REPEATS))
    cv = [metrics(y[dev], sigmoid(A * z + B), w[dev]) for z in reps]
    cv_summary = {k: (float(np.mean([m[k] for m in cv])), float(np.min([m[k] for m in cv])),
                      float(np.max([m[k] for m in cv]))) for k in cv[0]}
    cv_bands = band_table(np.tile(y[dev], N_REPEATS), sigmoid(A * np.concatenate(reps) + B),
                          np.tile(w[dev], N_REPEATS))

    # 3) one look at the held-out test split (model fit on train+val, calibration from step 2)
    clf = fit_classifier(X[dev], y[dev], w[dev])
    pt = sigmoid(A * logit_of(clf, X[test]) + B)
    test_metrics, test_bands = metrics(y[test], pt, w[test]), band_table(y[test], pt, w[test])

    print(f"calibration: p = sigmoid({A:.3f} * logit + {B:.3f})")
    print("\nCV on train+val (mean [min-max] over repeats):")
    for k, (m, lo, hi) in cv_summary.items():
        print(f"  {k:18s} {m:.3f} [{lo:.3f}-{hi:.3f}]")
    print("\nconfidence bands (CV, pooled):")
    for name, b in cv_bands.items():
        print(f"  {name:6s} {b['share']:6.1%} of photos, verdict right {b['accuracy']:.1%}")
    print(f"\nheld-out test ({test.sum()} crops, {len(set(groups[test & (y == 1)]))} mastitis / "
          f"{len(set(groups[test & (y == 0)]))} healthy source photos):")
    for k, v in test_metrics.items():
        print(f"  {k:18s} {v:.3f}")
    for name, b in test_bands.items():
        acc = "n/a" if b["accuracy"] is None else f"{b['accuracy']:.1%}"
        print(f"  band {name:6s} {b['share']:6.1%} of photos, verdict right {acc}")

    REPORTS.mkdir(exist_ok=True)
    json.dump(dict(calibration=dict(A=A, B=B), bands={n: lo for n, lo in BANDS},
                   cv=cv_summary, cv_bands=cv_bands, test=test_metrics, test_bands=test_bands,
                   settings=dict(lr_C=LR_C, detail_px=DETAIL_PX, repeats=N_REPEATS, folds=N_FOLDS)),
              open(REPORTS / "udder_final_evaluation.json", "w"), indent=2)
    print(f"\nsaved -> {REPORTS / 'udder_final_evaluation.json'}")


def stage_export():
    import keras
    from keras import layers
    rows = load_rows()
    X = load_features(rows)
    y, groups, w = arrays(rows)

    # deployment model uses ALL curated data; calibration from grouped CV on all data
    reps = [oof_logits(X, y, groups, w, seed) for seed in range(N_REPEATS)]
    A, B = fit_platt(np.concatenate(reps), np.tile(y, N_REPEATS), np.tile(w, N_REPEATS))
    clf = fit_classifier(X, y, w)
    scaler, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]

    # fold scaler + logistic regression + calibration into one Dense layer:
    # p = sigmoid(A * (coef . (x - mean) / scale + b0) + B)
    coef = lr.coef_[0] / scaler.scale_
    kernel = (A * coef).reshape(-1, 1).astype(np.float32)
    bias = np.array([A * (lr.intercept_[0] - coef @ scaler.mean_) + B], dtype=np.float32)

    fm = feature_model()
    head = layers.Dense(1, activation="sigmoid", name="p_mastitis")
    model = keras.Model(fm.input, head(fm.output), name="udder_classifier")
    head.set_weights([kernel, bias])

    MODELS.mkdir(exist_ok=True)
    path = MODELS / "udder_classifier.keras"
    model.save(path)

    # parity: exported file must reproduce the sklearn pipeline
    reloaded = keras.models.load_model(path)
    sample = rows[:: max(1, len(rows) // 40)]
    batch = np.stack([to_input(Image.open(ROOT / r["crop_path"])) for r in sample])
    p_file = reloaded.predict(batch, verbose=0).ravel()
    idx = [rows.index(r) for r in sample]
    p_ref = sigmoid(A * logit_of(clf, X[idx]) + B)
    print(f"parity: max |file - pipeline| = {np.abs(p_file - p_ref).max():.2e} over {len(sample)} crops")

    # timing on this CPU
    t0 = time.time()
    keras.models.load_model(path)
    load_s = time.time() - t0
    one = batch[:1]
    reloaded.predict(one, verbose=0)
    times = []
    for _ in range(20):
        t0 = time.time()
        reloaded(one, training=False)
        times.append(time.time() - t0)
    size_mb = path.stat().st_size / 2**20
    print(f"file {path.name}: {size_mb:.1f} MB | load {load_s:.1f} s | {np.median(times) * 1000:.0f} ms per photo (median, CPU)")

    card = dict(
        model="udder_classifier.keras", task="whole-udder photo -> P(mastitis)",
        input="RGB photo stretched to 224x224, float32 raw 0-255, shape (1,224,224,3)",
        output="calibrated P(mastitis) assuming equal prior odds; verdict = mastitis if p >= 0.5",
        confidence="confidence = max(p, 1-p); bands: " + ", ".join(f"{n} >= {lo}" for n, lo in BANDS),
        architecture="ImageNet MobileNetV2 (frozen) -> GAP -> logistic regression (folded with Platt calibration)",
        training_data="Roboflow 'cow and mastitis detection' v16 (CC BY 4.0), curated whole-udder crops; "
                      f"{len(set(groups[y == 1]))} mastitis / {len(set(groups[y == 0]))} healthy source photos",
        evaluation="see reports/udder_final_evaluation.json (grouped CV + held-out test)",
        limitations=["validated only on web/Roboflow images, not field photos",
                     "trained on udder crops; whole-cow photos with a small udder are out of scope",
                     "cannot detect subclinical mastitis"],
        calibration=dict(A=A, B=B), size_mb=round(size_mb, 1),
        cpu_ms_per_photo=round(float(np.median(times) * 1000)),
    )
    json.dump(card, open(MODELS / "udder_model_card.json", "w"), indent=2)
    print(f"saved -> {path} and udder_model_card.json")


def stage_gradcam():
    """Grad-CAM on the backbone's last conv block for held-out test crops."""
    import keras
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm

    model = keras.models.load_model(MODELS / "udder_classifier.keras")
    backbone = next(l for l in model.layers if isinstance(l, keras.Model))
    conv = backbone.get_layer("out_relu")
    inner = keras.Model(backbone.input, [conv.output, backbone.output])
    head = model.get_layer("p_mastitis")
    pre = [model.get_layer(n) for n in ("detail_match_down", "detail_match_up", "mobilenet_preprocess")]

    test_rows = [r for r in load_rows() if r["split"] == "test"]
    rows = []
    for label in ("healthy", "mastitis"):     # 12 per class, one crop per source photo where possible
        seen, picked = set(), []
        for r in test_rows:
            if r["label"] == label and r["group"] not in seen:
                seen.add(r["group"])
                picked.append(r)
        rows += picked[:12]
    tiles = []
    for r in rows:
        img = Image.open(ROOT / r["crop_path"]).convert("RGB")
        x = tf.constant(to_input(img)[None])
        for l in pre:
            x = l(x)
        with tf.GradientTape() as tape:
            fmap, pooled = inner(x, training=False)
            tape.watch(fmap)
            logit = tf.reduce_sum(pooled * head.kernel[:, 0]) + head.bias[0]
        grads = tape.gradient(logit, fmap)[0]
        cam = tf.nn.relu(tf.reduce_sum(fmap[0] * tf.reduce_mean(grads, axis=(0, 1)), axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        heat = Image.fromarray((cm.jet(cam)[..., :3] * 255).astype(np.uint8)).resize((INPUT, INPUT))
        base = img.resize((INPUT, INPUT))
        p = float(tf.sigmoid(logit))
        tile = Image.blend(base, heat, 0.4)
        canvas = Image.new("RGB", (INPUT, INPUT + 18), "black")
        canvas.paste(tile, (0, 18))
        from PIL import ImageDraw
        ImageDraw.Draw(canvas).text((3, 3), f"{r['label'][:4]} p={p:.2f}", fill="yellow")
        tiles.append(canvas)
    cols = 6
    sheet = Image.new("RGB", (cols * INPUT, ((len(tiles) + cols - 1) // cols) * (INPUT + 18)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * INPUT, (i // cols) * (INPUT + 18)))
    REPORTS.mkdir(exist_ok=True)
    sheet.save(REPORTS / "udder_gradcam.jpg", quality=85)
    print(f"saved -> {REPORTS / 'udder_gradcam.jpg'} (labels = true class, p = model P(mastitis))")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "evaluate", "export", "gradcam"], required=True)
    stage = ap.parse_args().stage
    dict(features=stage_features, evaluate=stage_evaluate, export=stage_export, gradcam=stage_gradcam)[stage]()
