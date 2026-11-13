# Udder Mastitis Screening

Whole-udder photo -> **healthy / affected** with a calibrated probability and a confidence level.
A screening aid, not a veterinary diagnosis. Validated only on web/Roboflow images so far.

## Current model (Oct 2026)

```
photo -> quality check -> udder detector (YOLO11n, ONNX) -> crop -> classifier (MobileNetV2 + logistic regression, Keras)
```

| File | What it is |
|---|---|
| `models/udder_classifier.keras` | frozen ImageNet MobileNetV2 + logistic regression + calibration, folded into one 9.2 MB file; ~110 ms/photo on CPU |
| `models/udder_detector.onnx` | YOLO11n that finds the udder (or a teat close-up) in a whole photo |
| `models/udder_model_card.json` | input format, output meaning, confidence bands, limitations |
| `src/predict.py` | `UdderScreener().predict(photo)` -> one result dict; this is what the app calls |

Weights are not in git; get them from the GitHub Release.

**Results** (classifier, 10x repeated 5-fold CV grouped by source photo; `reports/udder_final_evaluation.json`):
ROC-AUC 0.85, sensitivity 0.76, specificity 0.78. Held-out test (10 mastitis / 24 healthy source photos):
ROC-AUC 0.89, sensitivity 0.70, specificity 0.96. Confidence bands (CV): **high** = 34% of photos, right 95% of
the time; **medium** = 41%, right 75%; **low** = 25%, right 58% (close to a coin flip, so show it as low confidence).

**Data:** Roboflow "cow and mastitis detection" v16 (CC BY 4.0), curated to whole-udder views:
256 healthy / 155 mastitis crops (177 / 68 distinct source photos). Teat close-ups go to the teat model;
thermal images, tiny crops and unreadable images are removed. See `data/curation/` and `../docs/audit_notes.md`.

**End-to-end** (whole photo -> detector -> score; classifier fitted without the test photos;
`reports/udder_pipeline_evaluation.json`):

| On held-out test photos (21 mastitis / 34 healthy) | ROC-AUC | healthy flagged | mastitis caught |
|---|---|---|---|
| whole photo only | 0.89 | 18% | 81% |
| hand-labelled crop (best case) | 0.93 | 15% | 86% |
| **average of whole photo + detector crop (what `predict.py` does)** | **0.90** | **15%** | **89%** |

The averaging rule was picked after seeing these test photos and the differences are 1-3 photos, so treat it as a
sensible design choice, not a proven gain. Detector: test mAP50 0.87. It sends 3/55 real udder photos away
(1 "no udder found", 2 "use Teat Check"), redirects 21/23 teat close-ups to Teat Check, and finds no udder in 9/13
tick (cow-skin) photos. Outside healthy udder photos never used in training: 7/38 called affected (18%).
About 240 ms per photo on a desktop CPU for both models.

**Known limits:** only 68 distinct mastitis photos; no field photos yet; cannot detect subclinical mastitis;
about 1 in 6 healthy udders gets flagged; some non-udder photos (4/13 tick photos) still get a verdict.

### Rebuild, in order
```
python src/curate_roboflow.py --stage flags      # then visual triage -> data/curation/view_tags.csv
python src/curate_roboflow.py --stage build
python src/compare_models.py --stage features    # optional: model comparison
python src/compare_models.py --stage compare
python src/build_udder_model.py --stage features
python src/build_udder_model.py --stage evaluate
python src/build_udder_model.py --stage export
python src/build_udder_model.py --stage gradcam
..\venv_yolo\Scripts\python src/detector.py --stage prepare|train|detect|export   # PyTorch venv
python src/evaluate_pipeline.py                  # end-to-end: whole photo vs crop vs detector crop
```
TensorFlow venv: TF 2.21, Keras 3.15, plus opencv-python-headless, scikit-image, onnxruntime.
Detector venv (`venv_yolo`): PyTorch 2.14 (CUDA 12.6), Ultralytics 8.4.

---

## Original baseline (teammate's pipeline, before the Roboflow curation)

Built for the Dairy Mate teat/udder dual-pipeline task, as the second (udder) half — see
the companion `teats_pipeline/` for the teats model.

## Honest headline: this dataset is too small to trust yet

Only **95 images have a confident label** right now (55 mastitis + 40 healthy). The other
**~328 images in the udder pool have no reliable label** — their filenames (stock-photo
names, generic numbers, screenshots) give no signal, and the teats dataset already proved
filename patterns like "sehat" can't be trusted as a label shortcut in this data source
family. These 328 are excluded from training entirely, not guessed at.

With only 66 training images, **the model overfits badly**: 98.5% training accuracy but
only 41–75% balanced accuracy on validation depending on threshold (13 validation images —
small enough that the calibration curve is genuinely noisy, see
`reports/hyperparameter_tuning_log.csv` and the threshold sweep in
`src/calibrate_threshold.py`'s output). This is a real, useful finding for the team, not a
bug: **it means the ~328 unlabeled images need to be labeled before this pipeline can be
trusted for anything beyond a rough baseline.**

The labeling tool from earlier in this project (browser-based, keyboard-shortcut driven)
is the fastest path to fixing this — labeling those 328 images would roughly 4x the usable
dataset.

## What's genuinely solid here despite the small data

**ROC-AUC on held-out test: 0.905** — the model's probability *ranking* is actually good
(it generally scores mastitis images higher than healthy ones), even though accuracy at
any single fixed threshold is unstable. This suggests the underlying features/model choice
are sound; the problem is specifically the tiny labeled dataset making threshold
calibration unreliable, not the modeling approach itself.

## Clean dataset (after dedup + label-contradiction check)

- Mastitis: 55 images (explicit "Mastitis-" filename prefix only — anything ambiguous excluded)
- Healthy: 40 images (18 from `udder_images.zip`'s healthy folder + 22 new, zero-overlap, from `udder.zip`)
- Imbalance ratio: 1.36:1 (much milder than teats' 28.5:1)
- Checked for the same photo-appears-in-both-classes contradiction found in the teats
  dataset — **none found here**, this dataset is clean on that front

## Pipeline

| Step | Script |
|---|---|
| Clean dataset assembly (confident labels only) | `src/build_clean_dataset.py` |
| Group-aware leakage-safe split | `src/prepare_dataset.py` |
| Segmentation (GrabCut, all 95 images) | `src/segment.py` |
| Feature extraction (223-dim) | `src/features.py` |
| Model comparison (RF, GB, SVM) | `src/model_development.py` |
| Hyperparameter tuning | `src/tune_hyperparameters.py` |
| Final model training | `src/train_final.py` |
| Threshold calibration | `src/calibrate_threshold.py` |
| Test evaluation | `src/evaluate.py` |
| Error analysis | `src/error_analysis.py` |

## Model comparison (3-fold group CV on 66 training images)

| Model | Balanced Accuracy | Recall | F1 |
|---|---|---|---|
| **SVM (rbf, class-weighted)** | **0.698** | 0.814 | 0.763 |
| RandomForest (class-weighted) | 0.601 | 0.765 | 0.700 |
| GradientBoosting (sample-weighted) | 0.560 | 0.575 | 0.600 |

SVM won — a genuinely different, honest finding from the teats/original udder work where
tree ensembles won. With this little data, SVM's stronger regularization generalizes
better than tree ensembles, which have more capacity to memorize a small training set.

## Final test results (16 held-out images, 7 healthy / 9 mastitis)

| Metric | Value |
|---|---|
| Accuracy | 0.688 |
| Balanced Accuracy | 0.722 |
| Precision | 1.000 |
| Recall | 0.444 |
| F1 | 0.615 |
| **ROC-AUC** | **0.905** |

All 7 healthy test images were classified correctly (0 false mastitis alarms); 5 of 9
mastitis images were missed at this threshold. Given the strong ROC-AUC, a different
threshold choice could trade this back — but calibrating that threshold reliably needs
more validation data than the 13 images currently available.

## Recommendation for the team

1. **Label the ~328 unlabeled images** — this is the single highest-value next step,
   using the labeling tool already built for this project
2. Once labeled, retune the threshold and re-run `model_development.py` — a 4x larger
   dataset likely changes both the winning model and the achievable accuracy substantially
3. Treat the current 0.722 balanced accuracy as a baseline floor, not a final result
