"""
prepare_dataset.py

Takes raw images from:
    data/raw_healthy_sources/    (already populated: cleaned, deduped healthy teat images)
    data/raw_mastitis_sources/

And produces a stratified train/val/test split at:
    data/processed/train/{healthy,mastitis}
    data/processed/val/{healthy,mastitis}
    data/processed/test/{healthy,mastitis}

Run:
    python scripts/prepare_dataset.py
"""

import os
import shutil
import hashlib
import random
import argparse
import sys
from PIL import Image

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_clean_files(src_dir, label):
    """Walk src_dir, drop duplicates (by content hash) and corrupt files."""
    seen_hashes = {}
    kept = []
    n_dup, n_corrupt, n_skipped_ext = 0, 0, 0

    for root, _, files in os.walk(src_dir):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                n_skipped_ext += 1
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
                h = hashlib.md5(data).hexdigest()
            except Exception:
                n_corrupt += 1
                continue
            if h in seen_hashes:
                n_dup += 1
                continue
            try:
                im = Image.open(path)
                im.verify()
            except Exception:
                n_corrupt += 1
                continue
            seen_hashes[h] = path
            kept.append(path)

    print(f"[{label}] kept={len(kept)}  dropped_duplicates={n_dup}  "
          f"dropped_corrupt={n_corrupt}  skipped_wrong_ext={n_skipped_ext}")
    return kept


def stratified_split(file_list, ratios, seed=RANDOM_SEED):
    rng = random.Random(seed)
    files = file_list[:]
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def copy_split(split_dict, class_name, out_root):
    for split_name, files in split_dict.items():
        out_dir = os.path.join(out_root, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for i, path in enumerate(files):
            ext = os.path.splitext(path)[1].lower()
            newname = f"{class_name}_{split_name}_{i:04d}{ext}"
            shutil.copy(path, os.path.join(out_dir, newname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy_dir", default="data/raw_healthy_sources")
    parser.add_argument("--mastitis_dir", default="data/raw_mastitis_sources")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)

    healthy_files = collect_clean_files(args.healthy_dir, "healthy")
    mastitis_files = collect_clean_files(args.mastitis_dir, "mastitis")

    if len(mastitis_files) == 0:
        print("\n*** No mastitis images found in data/raw_mastitis_sources ***")
        print("Drop your mastitis teat images (or the folder/zip contents) into that "
              "directory, then re-run this script. Stopping here.")
        sys.exit(1)

    healthy_split = stratified_split(healthy_files, SPLIT_RATIOS)
    mastitis_split = stratified_split(mastitis_files, SPLIT_RATIOS)

    copy_split(healthy_split, "healthy", args.out_dir)
    copy_split(mastitis_split, "mastitis", args.out_dir)

    print("\n=== Split summary ===")
    for split_name in ["train", "val", "test"]:
        h = len(healthy_split[split_name])
        m = len(mastitis_split[split_name])
        print(f"{split_name:5s}: healthy={h:4d}  mastitis={m:4d}  ratio=1:{m/max(h,1):.2f}")

    print(f"\nProcessed dataset written to: {args.out_dir}")


if __name__ == "__main__":
    main()
