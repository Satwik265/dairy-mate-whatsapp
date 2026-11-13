# Teat-end labelling

**Why:** the teat model misses about half of rough teat-end rings, and Grad-CAM shows it often looks at the background next to the teat instead of the ring. The shared photo gate can't crop tighter: on these close-ups its teat box covers the whole photo, because it boxes the whole teat. A small detector trained on teat-end boxes would let us crop each photo to the teat end before classifying it.

## Run it

```
cd "F:\Dairy Mate\teat"
..\venv\Scripts\python.exe scripts\label_teat_end.py
```

Open http://localhost:8765 in a browser. Press Ctrl+C in the terminal to stop. Everything runs on your machine and nothing is uploaded.

- Photos to label: `teat/curation/teat_end_sample.csv`. It has 304 photos, one per photo moment, from `scripts/select_teat_end_sample.py`. It has extra rough-ring (score 3) and score 4 photos on purpose.
- Labels: `teat/curation/teat_end_labels.csv`. Each save is written straight away. When you reopen the page it starts at the first photo you haven't done.
- The split comes from the classifier's split: 209 train, 45 val and 50 test photos. A detector trained on these labels never sees a classifier test photo.

## Keys

| | |
|---|---|
| drag | draw a box (more than one is fine) |
| Enter | save the boxes and go to the next photo |
| N | no teat end visible, go to the next photo |
| U | unusable photo, go to the next photo |
| Z | undo the last box (right-click a box to delete it) |
| ← → | previous / next photo without saving |

## What to box

- **The teat end:** the tip with the opening (orifice) and the whole ring of skin around it, plus a small margin. That's roughly the last third of the teat. Never box the whole teat.
- **Same size for healthy and rough rings.** The box marks *where* the teat end is, not *what condition* it's in. If rough rings got bigger boxes, the detector would leak the label.
- **One box per teat end you can clearly see.** Skip teat ends that are blurry or far away.
- **Teat end cut off by the photo edge:** box the visible part if the orifice area shows. Otherwise press N.
- **U** is only for photos that aren't a teat at all, or are too blurry to place a box.

The whole set takes about 1 to 1.5 hours at 10 to 15 seconds a photo. You can split it over several sittings.

## After labelling

1. Turn the labels into a YOLO dataset and train a teat-end detector. It's a small YOLO11n, like the gate, in `venv_yolo`.
2. Check the detector on the 50 test photos.
3. Crop every teat photo to its detected teat end, retrain the classifier on the crops, and compare with v2 using `evaluate_v2.py`. Watch the rough rings caught (v2: 45%) and the SCTL-only AUC (v2: 0.84).
4. Keep the cropped model only if it's better. In the app this would be: gate, then teat-end detector, then crop, then classifier.
