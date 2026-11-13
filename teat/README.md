# Teat Classification Pipeline — Dairy Mate

Classifies a close-up photo of a cow's teat as **healthy** or **teat problem**. "Teat problem" is a mix of visible teat conditions (rough teat-end rings / hyperkeratosis, chemical burns, frostbite, pox, cracks, lesions), not confirmed mastitis, so the app should say "Teat problem detected", never "mastitis". It's a screening aid, not a diagnosis.

## Current model: v2 (Oct 2026)

`models/teat_classifier_v2.keras`: MobileNetV2, two-stage fine-tune (same recipe as v1), trained on the v2 dataset below. Calibration and confidence bands are in `models/teat_model_card.json`; `scripts/predict.py` is what the app calls. It sits behind the shared photo gate (`../common/gate.py`), so whole-udder and non-teat photos don't reach it.

### Results (held-out test split, grouped by photo moment; 95% intervals from a group bootstrap)

| | ROC-AUC | Problems caught | Healthy correct |
|---|---|---|---|
| All test photos | 0.95 (0.90-0.97) | 90% | 87% |
| SCTL photos only (both classes share one photo style, most honest number) | 0.84 (0.77-0.90) | 50% | 92% |

Teat-end score vs. how often it's flagged: score 1 1.5%, score 2 14%, score 3 45%, score 4 80%.
Full numbers: `reports/teat_v2_evaluation.json`. Model comparisons (frozen features, oversampling): `reports/teat_model_comparison.json`, `reports/teat_v2_os_evaluation.json`.

Tried and not kept:
- **Tighter crop with the gate's box**: no effect possible. On these close-ups the gate's teat box is the whole photo (median 100% of the image, 90% of boxes cover at least 95%; `curation/gate_boxes_v2.csv`). The gate boxes the whole teat, not the teat end.
- **384px input instead of 224px** (more pixels on the teat end; SCTL photos are ~450-600px): rough rings caught 34% (v2: 45%), non-SCTL healthy flagged 11/12 (v2: 8/12), SCTL-only AUC 0.84 (same). All within the confidence intervals, but no sign it helps. `reports/teat_v2_384_evaluation.json`.
- **Teat-end crop from a dedicated detector** (YOLO11n trained on 224 hand-boxed photos from a 304-photo sample, `curation/teat_end_labels.csv`, `scripts/teat_end_detector.py`; its top box hits a labelled teat end on 32/34 test photos). Every photo was cropped to a padded square around the teat end (`scripts/teat_end_crop.py`, whole photo kept when nothing is found: 2% of SCTL, ~33% of web photos) and the classifier retrained with the v2 recipe. Rough rings caught 41% (12/29; v2: 45%, 13/29), SCTL-only AUC 0.83 (v2: 0.84), non-SCTL healthy flagged 10/12 (v2: 8/12). No gain, so v2 stays. SCTL photos are already tight on the teat end (median crop keeps ~75% of the photo), so the crop adds little zoom where it matters. `reports/teat_v2_te_evaluation.json`.

### Known weaknesses
- **Healthy teats that don't look like the SCTL photos are often flagged** (8 of 12 in test). There are only 15 distinct non-SCTL healthy teat photos to learn from; more real healthy teat photos is the fix.
- **Misses about half of rough teat-end rings** (score 3).
- Validated only on web / Roboflow / SCTL images, not field photos.
- Can't detect subclinical mastitis from a photo.
- Grad-CAM (v2, test split, `reports/gradcam_v2/`): on confident SCTL problem calls the heat sits on the teat end, which is what we want. Borderline problem calls (score 0.5-0.65) and most mistakes put the heat on the background next to the teat, image borders/rotation corners, or straw/hair. Wrongly flagged healthy web photos are mostly crops of the same few pink/black-skin photos, with heat spread over skin and background. Missed rough rings show little or no heat on the ring itself.

## v1 (July 2026), kept for reference

v1's "~97% test accuracy" was **not valid**: the old split put augmented copies of the same photo in both train and test (110 of 207 test source photos were also in train/val), healthy images were almost all one photo style (SCTL Score 1) while "mastitis" was everything else, and ~40% of the "mastitis" images were whole-udder photos. v1 called every healthy whole-udder photo a problem. Details: `../docs/audit_notes.md`.

## Dataset v2 (`scripts/build_dataset_v2.py`)

- healthy: SCTL teat-end Score 1 + Score 2, plus healthy teat close-ups from other sources
- problem: SCTL Score 3 + Score 4, plus teat-lesion close-ups (Roboflow / teat-scoring sites)
- whole-udder photos moved out to the udder model (`curation/udder_v2_candidates.csv`)
- label fixes: 22 teat-scoring-site "normal" photos had been uploaded as mastitis -> healthy. Note: `sehat-N` filenames are NOT trustworthy labels (they are diseased teats)
- hand photos excluded (MediaPipe), duplicates and near-duplicates grouped, contradictory photos dropped
- split 70/15/15 by photo moment / source photo, stratified by label and source; no group crosses splits

1,852 images: healthy 939 SCTL + 45 web, problem 205 SCTL + 663 web. Manifest: `curation/teat_v2_manifest.csv`.

## Scripts

- `build_dataset_v2.py --stage views | hands | build` (views needs `venv_yolo`, hands needs `venv_handdetect`)
- `train_teat_model.py --data_dir data/processed_v2 --out_model models/teat_classifier_v2.keras --history_out reports/teat_v2_training_history.json`
- `evaluate_v2.py [--model ... --out ...]` (grouped CIs, SCTL-only, stress test)
- `compare_frozen.py --stage features | compare`
- `predict.py photo.jpg` (gate -> model -> calibrated result)
- `select_teat_end_sample.py`, `label_teat_end.py` (teat-end box labelling, local page at http://localhost:8765; guide: `../docs/teat_end_labelling.md`)
- `gate_boxes.py` (gate teat box for every dataset image -> `curation/gate_boxes_v2.csv`)
- `select_teat_end_sample.py` + `label_teat_end.py` (304-photo teat-end box sample and its local labelling page), `teat_end_detector.py --stage prepare|train|export` (YOLO11n teat-end detector, `venv_yolo`), `teat_end_crop.py` (crop the dataset to teat ends); `evaluate_v2.py --teat_end` evaluates a model trained on crops
- `train_teat_model.py --img_size N` trains at another input size; `evaluate_v2.py` and `gradcam_visualize.py` read the size from the model
- v1 scripts (`prepare_dataset.py`, `evaluate.py`, `gradcam_visualize.py`, `detect_hands_gloves.py`, `quarantine_flagged_hands.py`, `run_pipeline.py`) are kept; `prepare_dataset.py` has the leaky split, don't use it for new work.

## Environment

- Python 3.12, installed from python.org — not the Microsoft Store version, that one breaks paths
- Main env: TensorFlow 2.21 / Keras 3.15 (`requirements.txt`), plus onnxruntime + opencv for the gate
- MediaPipe needs its own venv (`venv_handdetect`). TensorFlow and MediaPipe conflict on protobuf versions on Windows and won't run together in one env.
- `venv_yolo` (PyTorch + Ultralytics) only for the view-tagging stage and the gate detector
