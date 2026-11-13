# Audit notes (2026-10-04/05)

Findings from the Phase 0 audit, kept for future reference.

## Teat model (`teat/`)
- **Split leakage:** `prepare_dataset.py` only removes byte-identical duplicates. Roboflow `.rf.<hash>` augmented copies of one photo have different bytes, so 198 source photos ended up in more than one split, and 110 of the 207 test source photos also appear in train/val. The ~97% test accuracy is inflated.
- **Source confound:** 449 of the 463 healthy images come from the SCTL "Score_1" set (teat-end close-ups). All "mastitis" images are 512x512 Roboflow exports. In a stress test, the model called 27/27 healthy whole-udder photos and 13/13 tick (cow body) photos a "problem", mostly with confidence above 0.98.
- The positive class is a mix of teat conditions (hyperkeratosis, chemical burns, pox, frostbite…), not confirmed mastitis. The UI should say "Teat problem detected".
- Inference: 224x224 RGB, raw 0-255 input (preprocessing happens inside the model), single sigmoid output (0 = healthy). About 62 ms/image on a desktop CPU. The resize method changes results slightly, so match the training resize.

## Udder (`udder/`)
- A teammate's pipeline: GrabCut + 223 handcrafted features + RBF SVM, on 95 labelled images (about 63 distinct photos). Test set: 16 images (11 distinct photos). Recall 0.44, ROC-AUC 0.905 (rough 95% CI 0.75 to 1.0).
- The Roboflow "cow and mastitis detection" project (v16, no augmentation) is the source of most of our teat AND udder images. Its `Mastitis_infected_udder` class mixes whole-udder photos with teat-lesion close-ups.
- Curation decisions: the udder model covers whole-udder views only; teat-lesion close-ups go to the teat model; thermal images are dropped; crops under 96 px are dropped. The view triage (`udder/data/curation/view_tags.csv`) was done visually by an AI assistant, not by a vet.
- Clean result: 256 healthy crops (177 distinct sources) and 155 mastitis crops (68 distinct sources), split by source group.
- Remaining shortcuts to handle at training time: mastitis crops are about twice the resolution of healthy ones; greyscale and black-corner images are more common in the mastitis class; one Adobe stock video supplies 59 healthy frames (one group).

## Tick
- YOLO11n tick detection failed (median box about 4x6.5 px, mAP50 0.001 to 0.037). The interim model is a patch classifier. Patch metrics are not image-level metrics; an aggregation rule still needs validation.
