# Tick Check — Dairy Mate

Looks at a photo of a cow and says whether **ticks are visible**: "Ticks detected" or "No ticks seen in this photo", with a calibrated probability and a confidence level. It is a screening aid, not a diagnosis. **Right now it only works on photos like its training set (one crush pen, ticks a few pixels wide). It misses most ticks on close-up photos from other farms.** See "Known weaknesses" before using it.

## Beta and photo collection

Tick Check launches as **Beta** with the udder and teat checks (decision 2026-10-05). The model card has `"release_status": "beta"`, and every `predict()` result carries `release_status` so the app can show a Beta badge.

Close-up phone photos are what the model fails on, and none are available yet. The app will invite farmers to share their tick photos so a close-up training set can be built. Requirements for that upload feature (website/app phase):
- **Opt-in only**, with a plain-language consent line (Hindi/Kannada too): the photo is stored and used to improve the model.
- **Ask the farmer what they saw**: "ticks" / "no ticks" / "not sure". This is a weak label, so photos still need review before training.
- **Collect tick-free photos too.** Without them, close-up false alarms can't be measured or fixed.
- **Strip EXIF/GPS** on upload, store no names or phone numbers, and allow deletion on request.
- **Keep the farm/upload id** so the training split can be grouped by farm (no farm in both train and test).

## How it works (`scripts/predict.py`)

1. Light quality check (`../common/gate.py`: unreadable, tiny or near-black photos -> retake).
2. The photo is stretched to 640x640, the same geometry as the training images.
3. Lenient animal check: a stock COCO YOLO11n (`models/animal_detector.onnx`, never trained on ticks) looks for an animal. If it finds one, the photo is cropped to it (+10%). If not, the whole photo is scored anyway and the result says so (`animal_found: false`).
4. The crop is tiled into 96x96 patches (stride 48). A MobileNetV2 patch classifier (`models/tick_patch_B.onnx`) scores each patch: "is there a tick in this patch?"
5. Image score = mean of the 5 highest patch scores, calibrated with Platt scaling fitted on the validation split (`models/tick_model_card.json`).
6. Verdict: affected (p >= 0.5) or healthy. There is **no "not sure" verdict**. Low confidence changes the wording ("Possible ticks", "No ticks seen, low confidence") and the advice ("check the animal by hand: ears, neck, dewlap, udder, tail base; ask a vet if unsure"). A "healthy" verdict is never reported as high confidence, because the model can miss ticks with p near 0 (see below).

The shared udder/teat photo gate is **not** used to reject or redirect tick photos. It would send 78% of the test tick photos to "retake" and 22% to Udder Check, and 34% of the web tick photos to Teat Check (`reports/tick_gate_routing.json`).

The result also lists the patches scoring >= 0.5 (`tick_patches`, as fractions of the photo) so the app can outline where it saw ticks.

## Data and split (`scripts/download_dataset.py`, `scripts/build_split.py`)

- **Positives:** Griffith University "Ticks" (Roboflow, CC BY 4.0). It has 455 distinct images, almost all of the same few Hereford calves in one crush pen, filmed as frames of a handful of videos plus consecutive standalone photos. Ticks are about 4x6 px at 640x640. Only 20 images have no tick.
- **Negatives:** 1,185 presumed tick-free cows from the Roboflow "cow and mastitis detection" v16 export (dairy, CCTV and stock photos). They were **not** labelled for ticks.
- **Grouped split:** by source video, by runs of 10 consecutive standalone photos (IMG_134x, ...) and by CCTV/video series. Groups linked by a near-duplicate (dHash) are merged. 1,640 images in 48 groups: train 1,007, val 328, test 305. Test is 68 Griffith images (61 with ticks, 7 without, **6 groups**) plus 237 rf16 cows (12 groups). Manifest: `curation/split_manifest.csv`.

## Models tried (selection on validation only)

Both are MobileNetV2 (ImageNet), fine-tuned in PyTorch on patches sampled on the fly (`scripts/train_patch_classifier.py`, 15 epochs x 250 steps, GTX 1050 Ti):

| Validation (n=328: 79 with ticks, 249 without) | A: Griffith negatives only | B: + 30% negatives from rf16 cows |
|---|---|---|
| Patch PR-AUC (training checkpoint metric) | 0.58 | 0.73 |
| Tick-free cow patches scored >= 0.5 (all val grid patches) | 12.5% | 2.7% |
| Image ROC-AUC (top-5 mean) | 0.90 | 0.98 |
| Image ROC-AUC (top-5 mean), Griffith images only (12 tick-free) | 0.97 | 0.95 |
| Sensitivity / specificity at calibrated p >= 0.5 | 80% / 82% | 95% / 94% |

Aggregation rules tried: max, top-k mean (k = 2, 3, 5), count of patches >= 0.5/0.7/0.9. The selection criterion was the mean of all-image AUC and Griffith-only AUC. **Chosen: B, top-5 mean.** Note that B's gain is almost all on the rf16 negatives, which B saw (other photos from the same source) in training. Part of the gain is B learning "this looks like a dairy/CCTV photo, not the Griffith pen", not ticks. `reports/tick_selection_val.json`.

## Results (held-out test, run once; 95% intervals from a group bootstrap over 18 groups)

| Test (n=305: 61 with ticks, 244 without) | Value |
|---|---|
| ROC-AUC | 0.99 (0.96-1.00) |
| Ticks caught (sensitivity) | 93%, 57/61 (0.85-1.00) |
| Tick-free correct (specificity) | 95%, 233/244 (0.85-0.99) |
| Confusion matrix | TP 57, FN 4, TN 233, FP 11 |

Split by source: Griffith pen photos 57/61 caught and 6/7 tick-free correct (7 is far too few to measure specificity in the pen). rf16 cows 227/237 correct. The 61 positives come from only 6 test groups, so they are not independent. Patch level (Griffith test, 3,996 patches): precision 0.72, recall 0.77 at 0.5, PR-AUC 0.81. Full numbers and per-image scores: `reports/tick_evaluation.json`.

**The old Colab numbers are not comparable.** The Colab patch metrics (patch F1 0.77, P 0.72, R 0.83, val PR-AUC ~0.69) are likely optimistic. A regex patch took the grouping from 17 source videos to 152 groups by making each standalone photo (IMG_1390, IMG_1391, ...) its own group, so near-identical consecutive photos could fall on both sides of the split. The Colab negatives were also capped at 40 per image. The new split groups consecutive photos together and evaluates every grid patch the app would score. The Colab checkpoint is not recoverable (a Drive search found no model files).

### Coat-colour shortcut check (`reports/tick_coat_colour_check.json`)

This repeats the Colab check. "Brownness" is mean(R - B) of each patch, correlated (Pearson) with the predicted probability and with the true label on test patches:

| | pred vs brownness | label vs brownness |
|---|---|---|
| New, Griffith test patches (n=3,996) | -0.235 | -0.197 |
| New, all test patches incl. rf16 cows (n=27,397) | -0.126 | -0.092 |
| Old Colab (different patch set) | -0.167 | -0.247 |

The prediction tracks brownness only slightly more than the label does (same sign, gap ~0.04), so there is no strong coat-colour shortcut. There is a mild lean: within tick-free patches, paler patches score a bit higher (r = -0.18), consistent with false alarms on the white parts of Hereford coats.

### External web photos: sanity check, not validation

There are 35 unique web photos of infested animals: 13 in `tick_images/infested` and 23 from `ticks_images.zip` (one exact duplicate dropped, and several more are crops of the same photo). One is a goat with ringworm, so 34 relevant photos. All are infested, and there are **no tick-free web photos**, so only misses can be counted. **B called 5 of 35 affected (4 of the cattle-tick photos plus the goat).** Heavy close-up infestations scored near 0. For information only (not used to choose): A called 26/35 affected, but A also flags 12.5% of tick-free cow patches, so with no healthy external photos that can't be read as A being better.

The likely cause is scale and domain. Training ticks are a few pixels wide on one pen's calves, while web close-ups show ticks 30-80 px wide on other breeds. B's tick-free examples also come from a different photo source than all of its tick examples.

## Tried and not kept

- **YOLO11n tick detectors (Colab), five variants:** 640 baseline, 1280, cattle-crop, cattle-crop 1280 and P2 head. All failed on ticks (tick mAP50 ~0.001-0.037), which is why this is a patch classifier.
- **Variant A** (no rf16 negatives): kept as `tick_patch_A` for reference. It lost on validation (AUC 0.90 vs 0.98, 12.5% vs 2.7% false-positive patches on tick-free cows).
- **Count-above-threshold and max aggregation:** lower validation AUC than top-5 mean for B (`reports/tick_selection_val.json`).
- **Multi-scale scoring (no retraining)** (`scripts/multiscale.py`): the animal crop also scored at 1/2, 1/4 and 1/8 size so close-up ticks shrink to training size; top-5 mean pooled over scales, recalibrated on val. Settings fixed in advance, not tuned on the web photos. Web tick photos caught 7/35 (1x: 5/35); the 1/8 scale never fires. `reports/tick_patch_B_multiscale.json`.
- **Variant C, zoomed training** (`--scale-max 6`: training windows down to 16 px, so ticks reach close-up size): val AUC 0.97 / specificity 90% (B: 0.98 / 94%), so B stays under the val-only rule. Web tick photos caught 11/35 (B: 5/35), with or without multi-scale. Test was also run for the record: AUC 0.99, 61/61 caught, 231/244 tick-free correct (same 6 positive groups). `reports/tick_patch_C_multiscale.json`; weights `tick_patch_C.*` kept for reference.

## Known weaknesses

- **Does not work on close-up or other-farm photos yet:** 4 of 34 web cattle-tick photos caught. A farmer's phone close-up is exactly this kind of photo.
- **One source of positives:** every training/val/test tick image is from one Roboflow set (a few calves, one pen, a few videos). The test AUC mostly measures "Griffith pen with ticks vs other cow photos without", and the CIs rest on 6 positive groups.
- **Within-pen specificity is unmeasured:** only 7 tick-free Griffith test images.
- **Negatives are presumed tick-free,** not checked for ticks, and are mostly whole cows at a distance.
- **Tiny ticks only:** trained at about 4x6 px per tick. Multi-scale scoring and zoomed training (above) did not fix close-ups.
- **Can't tell tick close-ups from tick-free close-ups:** on 35 web tick photos vs 164 presumed tick-free cow close-ups (137 SCTL healthy teats, 27 healthy udders) the ranking AUC is 0.34 for B, i.e. tick-free close-ups score *higher*. It is 0.62 for C. With the app's threshold B already calls 8/27 healthy udder photos and 12/137 healthy teat close-ups "ticks". Those stand-in negatives are teat/udder skin, not tick-free coat close-ups, so this is a rough stress test.
- **No field photos** anywhere in train, val or test.
- What would fix it: labelled phone photos of infested **and** tick-free cattle at several distances (close-up skin, ears, udder, tail base), from more than one farm.

## How to run

Uses `venv_yolo` (Python 3.12, PyTorch cu126); versions are in `requirements.txt`. Weights (`models/*.onnx`, `*.pt`) are not in git and go to GitHub Releases.

```
python tick/scripts/download_dataset.py                 # Roboflow key from ROBOFLOW_API_KEY or tick/data/roboflow_key.txt (gitignored)
python tick/scripts/build_split.py                      # -> curation/split_manifest.csv, data/split/
python tick/scripts/train_patch_classifier.py --name tick_patch_B --rf-neg-frac 0.3
python tick/scripts/evaluate.py maps --model tick_patch_B           # patch maps for val/test/external -> data/eval_cache/
python tick/scripts/evaluate.py select --models tick_patch_A tick_patch_B   # VAL only -> models/tick_model_card.json
python tick/scripts/evaluate.py test                                # once -> reports/tick_evaluation.json
python tick/scripts/evaluate.py coat                                # coat-colour check
python tick/scripts/evaluate.py gate                                # shared-gate routing on tick photos
python tick/scripts/predict.py photo.jpg                            # the app call: TickScreener().predict(photo)
```

Set `TORCH_HOME` to a folder on F: before training, or the ImageNet weights download to the C: user cache.

## Credits

Griffith University "Ticks" dataset (Roboflow Universe, CC BY 4.0). Roboflow "cow and mastitis detection" v16 by kirubel-yemane (CC BY 4.0) for the tick-free cows.
