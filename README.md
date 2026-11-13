# DairyMate models

Computer-vision models for screening dairy cattle health from a phone photo. They are screening aids, **not** a veterinary diagnosis.

| Folder | Model | Status |
|---|---|---|
| `common/` | Shared photo gate (YOLO11n): finds a whole udder or a teat close-up and routes the photo | Done. Teat close-ups -> Teat Check 100% (298/298 test photos); whole udders -> Udder Check 96%; tick/cow-skin photos only 46% sent back. |
| `udder/` | Whole udder: healthy vs. affected (MobileNetV2 features + logistic regression, calibrated) | v1 done (`udder/src/predict.py`). End-to-end test: ROC-AUC 0.93, 90% of mastitis caught, 12% of healthy flagged. Only 68 distinct mastitis photos. |
| `teat/` | Teat close-up: healthy vs. teat problem (fine-tuned MobileNetV2, calibrated) | v2 done (`teat/scripts/predict.py`). Grouped test: ROC-AUC 0.95 (0.90-0.97), 0.84 on SCTL photos alone. Weak on rough rings (score 3: ~45% caught) and on healthy teats that don't look like the SCTL set (8/12 flagged). v1's ~97% was inflated by train/test leakage. |
| `tick/` | Ticks visible: patch-based MobileNetV2 (96x96 patches, top-5 mean, calibrated) | v1 (`tick/scripts/predict.py`). Grouped test: ROC-AUC 0.99 (0.96-1.00), 57/61 tick photos caught, 95% of tick-free cows correct, but all tick photos come from one pen (6 test groups). **Fails on close-up web photos (4/34 caught); not ready for field use.** The old Colab patch metrics were from a leakier split. |
| `web/` | Farmer website (FastAPI + plain HTML/JS) for all three checks, with an optional WhatsApp sign-up stored in a Google Sheet | In progress: works locally end to end (ONNX, no TensorFlow); deploying to Render free. See `web/README.md`. |

Every model has so far been validated only on web/Roboflow images, not on field photos.

## Layout

```
common/  gate.py (shared photo gate; weights in common/models/, not in git)
teat/    scripts/ (build_dataset_v2, train, evaluate_v2, predict), curation/*.csv, reports/, models/*.json
udder/   src/ (curate, build model, detector, evaluate, predict), data/curation/*.csv, reports/
tick/    scripts/ (download_dataset, build_split, train_patch_classifier, evaluate, predict), curation/split_manifest.csv, reports/, models/*.json
web/     static/ (website), server/ (FastAPI), apps_script/ (Google Sheet), deploy/ (ONNX + weights for Render), tests/; render.yaml (Render config)
docs/    audit notes
```

Each app is one call: `UdderScreener().predict(photo)` / `TeatScreener().predict(photo)` / `TickScreener().predict(photo)` -> status
(ok / retake / use the other check), label (healthy / affected), calibrated probability, confidence level
(high / medium / low) and i18n keys for advice and the screening-aid disclaimer.

Raw images, datasets, venvs and model weights are not committed (see `.gitignore`). Weights are shared via GitHub Releases.

## Data sources and credits

- Roboflow Universe, "cow and mastitis detection" by kirubel-yemane, v16, CC BY 4.0:
  https://universe.roboflow.com/kirubel-yemane/cow-and-mastitis-detection
- SCTL teat-end images (healthy teats, "Score_1"), CC BY 4.0
- Griffith University ticks dataset (Roboflow), CC BY 4.0 (tick model; the rf16 set above also supplies its tick-free cows)
