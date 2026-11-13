"""
Teats pipeline - Step 1: annotation manifest + group-aware split.
"""
import re
import csv
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
CLASSES = {"healthy": 0, "mastitis": 1}

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def base_group_id(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\..*$", "", stem)
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return stem


def build_manifest():
    rows = []
    for cls_name, label in CLASSES.items():
        cls_dir = RAW_DIR / cls_name
        for f in sorted(cls_dir.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                rows.append({
                    "filepath": str(f),
                    "label": label,
                    "class_name": cls_name,
                    "group_id": f"{cls_name}__{base_group_id(f.name)}",
                })
    return rows


def group_stratified_split(rows):
    groups_by_class = defaultdict(set)
    for r in rows:
        groups_by_class[r["label"]].add(r["group_id"])

    train_groups, val_groups, test_groups = set(), set(), set()
    for label, groups in groups_by_class.items():
        groups = sorted(groups)
        random.shuffle(groups)
        n = len(groups)
        n_train = max(1, int(n * TRAIN_FRAC))
        n_val = max(1, int(n * VAL_FRAC))
        train_groups.update(groups[:n_train])
        val_groups.update(groups[n_train:n_train + n_val])
        test_groups.update(groups[n_train + n_val:])

    train = [r for r in rows if r["group_id"] in train_groups]
    val = [r for r in rows if r["group_id"] in val_groups]
    test = [r for r in rows if r["group_id"] in test_groups]
    return train, val, test


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "class_name", "group_id"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(name, rows):
    n0 = sum(1 for r in rows if r["label"] == 0)
    n1 = sum(1 for r in rows if r["label"] == 1)
    groups = len(set(r["group_id"] for r in rows))
    print(f"{name:5s}: {len(rows):4d} images ({n0} healthy, {n1} mastitis) across {groups} source groups")


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    write_csv(SPLITS_DIR / "annotations_full.csv", rows)
    print(f"Full manifest: {len(rows)} images\n")

    train, val, test = group_stratified_split(rows)

    tr_g = set(r["group_id"] for r in train)
    va_g = set(r["group_id"] for r in val)
    te_g = set(r["group_id"] for r in test)
    assert not (tr_g & va_g) and not (tr_g & te_g) and not (va_g & te_g), "Group leakage detected!"

    summarize("Train", train)
    summarize("Val", val)
    summarize("Test", test)
    print("\nNo group leakage between splits (verified).")

    write_csv(SPLITS_DIR / "train.csv", train)
    write_csv(SPLITS_DIR / "val.csv", val)
    write_csv(SPLITS_DIR / "test.csv", test)


if __name__ == "__main__":
    main()
