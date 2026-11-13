"""
Udder pipeline - Step 0: assemble the clean, CONFIDENTLY-LABELED dataset.

Sources:
- udder_images.zip -> images/ folder: 384 images total, of which only 56 have
  an explicit "Mastitis-" filename prefix. The other 328 have ambiguous
  filenames (stock-photo names, "sehat-N", "img-N", screenshots) that give
  NO reliable label signal - "sehat" means healthy in Indonesian/Malay in
  general, but the teats dataset proved files with that naming pattern can
  still be mastitis-labeled in this data source family, so filename alone
  is NOT trusted here. These 328 are treated as unlabeled and excluded from
  training until manually labeled (see labeling tool provided separately).
- udder_images.zip -> healthy/ folder: 27 confirmed healthy images
- udder.zip -> udder/ folder: 27 more confirmed healthy images (new, zero
  hash-overlap with the above, verified separately)

Also runs the same label-contradiction check used for the teats dataset
(Roboflow source-photo groups appearing in both classes) - result: NONE
found in this dataset (already verified), but the check is kept here so
that re-running this script after adding labeled images stays safe.
"""
import re
import hashlib
import shutil
from pathlib import Path

SRC_MASTITIS_POOL = Path("/home/claude/dairymate_data/udder_images/udder_images/images")
SRC_HEALTHY_1 = Path("/home/claude/dairymate_data/udder_images/udder_images/healthy")
SRC_HEALTHY_2 = Path("/home/claude/dairymate_data/udder_healthy/udder")

OUT_ROOT = Path("/home/claude/udder_pipeline/data/raw")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def base_group_id(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\..*$", "", stem)
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return stem


def main():
    (OUT_ROOT / "mastitis").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "healthy").mkdir(parents=True, exist_ok=True)

    # --- Mastitis: only files with explicit "Mastitis-" filename prefix ---
    seen = set()
    n_mastitis = 0
    for f in sorted(SRC_MASTITIS_POOL.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        if not f.name.lower().startswith("mastitis"):
            continue  # ambiguous filename - not confidently labeled, skip
        h = md5(f)
        if h in seen:
            continue
        seen.add(h)
        shutil.copy(f, OUT_ROOT / "mastitis" / f.name)
        n_mastitis += 1

    n_unlabeled = sum(
        1 for f in SRC_MASTITIS_POOL.iterdir()
        if f.suffix.lower() in IMG_EXTS and not f.name.lower().startswith("mastitis")
    )

    # --- Healthy: both confirmed sources, deduped ---
    healthy_hashes = set()
    n_healthy, n_healthy_src2_new = 0, 0
    for f in sorted(SRC_HEALTHY_1.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        h = md5(f)
        if h in healthy_hashes:
            continue
        healthy_hashes.add(h)
        shutil.copy(f, OUT_ROOT / "healthy" / f.name)
        n_healthy += 1

    if SRC_HEALTHY_2.exists():
        for f in sorted(SRC_HEALTHY_2.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            h = md5(f)
            if h in healthy_hashes:
                continue
            healthy_hashes.add(h)
            shutil.copy(f, OUT_ROOT / "healthy" / f.name)
            n_healthy += 1
            n_healthy_src2_new += 1

    print(f"Mastitis (confidently labeled by filename): {n_mastitis}")
    print(f"Healthy total (deduped): {n_healthy} "
          f"({n_healthy - n_healthy_src2_new} from udder_images.zip + {n_healthy_src2_new} new from udder.zip)")
    print(f"Excluded as unlabeled (ambiguous filenames): {n_unlabeled}")

    # --- Label-contradiction check (same methodology as teats pipeline) ---
    healthy_groups = set(base_group_id(f.name) for f in (OUT_ROOT / "healthy").iterdir())
    removed = []
    for f in list((OUT_ROOT / "mastitis").iterdir()):
        if base_group_id(f.name) in healthy_groups:
            removed.append(f.name)
            f.unlink()

    if removed:
        print(f"\nREMOVED {len(removed)} mastitis files with a label contradiction "
              f"(same source photo also present in healthy set):")
        for name in removed:
            print(f"  - {name}")
    else:
        print("\nNo label contradictions found between mastitis and healthy sets.")

    n_mastitis_final = len(list((OUT_ROOT / "mastitis").iterdir()))
    print(f"\nFinal clean labeled dataset: {n_mastitis_final} mastitis, {n_healthy} healthy")
    print(f"Imbalance ratio: {n_mastitis_final / n_healthy:.2f} : 1")


if __name__ == "__main__":
    main()
