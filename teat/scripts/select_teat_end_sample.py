"""
Pick the photos to label with teat-end boxes (for a small teat-end detector that crops tighter than the gate).

One photo per photo moment / source group, so the labels cover as many different teats as possible. The split
is kept from the classifier's v2 split: the detector trains on classifier-train photos, tunes on val, and is
checked on test photos it never saw, so later teat-end crops of test photos stay honest.

Rough rings (SCTL score 3) and score 4 are over-represented on purpose: they are the misses we're trying to fix.

    python scripts/select_teat_end_sample.py            -> curation/teat_end_sample.csv
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

TEAT = Path(__file__).resolve().parents[1]
MANIFEST = TEAT / "curation" / "teat_v2_manifest.csv"
OUT = TEAT / "curation" / "teat_end_sample.csv"
SEED = 42

# groups to take per (split, source, label, score); capped at what exists
TARGET = {
    "train": {("sctl", "healthy", "1"): 40, ("sctl", "healthy", "2"): 40, ("sctl", "problem", "3"): 40,
              ("sctl", "problem", "4"): 99, ("web", "healthy", ""): 99, ("web", "problem", ""): 70},
    "val":   {("sctl", "healthy", "1"): 8, ("sctl", "healthy", "2"): 8, ("sctl", "problem", "3"): 10,
              ("sctl", "problem", "4"): 99, ("web", "healthy", ""): 99, ("web", "problem", ""): 13},
    "test":  {("sctl", "healthy", "1"): 8, ("sctl", "healthy", "2"): 8, ("sctl", "problem", "3"): 12,
              ("sctl", "problem", "4"): 99, ("web", "healthy", ""): 99, ("web", "problem", ""): 16},
}


def main():
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(MANIFEST)))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["split"], r["source"], r["label"], r["score"])][r["group"]].append(r)

    picked = []
    for split, targets in TARGET.items():
        for kind, n in targets.items():
            by_group = groups[(split,) + kind]
            names = sorted(by_group)
            rng.shuffle(names)
            for g in names[:n]:
                picked.append(rng.choice(sorted(by_group[g], key=lambda r: r["id"])))

    rng.shuffle(picked)  # mixed order so labelling doesn't drift with one photo style
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "label", "source", "score", "group"])
        w.writeheader()
        for r in picked:
            w.writerow({k: r[k] for k in w.fieldnames})

    counts = defaultdict(int)
    for r in picked:
        counts[(r["split"], r["source"], r["label"], r["score"])] += 1
    for k in sorted(counts):
        print(k, counts[k])
    print(f"{len(picked)} photos -> {OUT}")


if __name__ == "__main__":
    main()
