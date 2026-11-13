"""
Run the shared photo gate over a teat dataset and save the teat_closeup box for every image,
so crop experiments don't have to re-run the detector.

    python teat/scripts/gate_boxes.py --data_dir teat/data/processed_v2 --out teat/curation/gate_boxes_v2.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

TEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAT.parent))
from common.gate import PhotoGate, open_photo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(TEAT / "data" / "processed_v2"))
    ap.add_argument("--out", default=str(TEAT / "curation" / "gate_boxes_v2.csv"))
    args = ap.parse_args()

    gate = PhotoGate()
    rows = []
    for path in sorted(Path(args.data_dir).glob("*/*/*")):
        split, label = path.parts[-3], path.parts[-2]
        found = gate.detect(open_photo(path))
        conf, box = found.get("teat_closeup", (0.0, [0.0, 0.0, 1.0, 1.0]))
        area = (box[2] - box[0]) * (box[3] - box[1])
        rows.append(dict(split=split, label=label, file=path.name, conf=round(conf, 4),
                         x0=box[0], y0=box[1], x1=box[2], y1=box[3], area=round(area, 4)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    areas = np.array([r["area"] for r in rows])
    missing = sum(r["conf"] == 0 for r in rows)
    print(f"{len(rows)} images, no teat box: {missing}")
    print("box area as fraction of photo: " + ", ".join(
        f"p{q}={np.percentile(areas, q):.2f}" for q in (10, 25, 50, 75, 90)))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
