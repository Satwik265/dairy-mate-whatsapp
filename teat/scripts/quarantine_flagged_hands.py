"""
quarantine_flagged_hands.py

Reads hand_detection_report.csv and moves every image flagged=True out of
the training folders into a quarantine folder, so the dataset used for
training no longer contains the hand/glove shortcut.

Images are MOVED, not deleted — they land in data/quarantined_hand_images/
so you can review or restore them later if needed.

Run this BEFORE prepare_dataset.py / train_teat_model.py so the cleaned
data flows through the rest of the pipeline.

Usage:
    python scripts/quarantine_flagged_hands.py
    python scripts/quarantine_flagged_hands.py --dry_run   # preview only, moves nothing
"""

import os
import csv
import shutil
import argparse


def find_by_basename(basename, search_dirs):
    """Fallback: if the exact recorded path doesn't resolve, search for a file
    with the same name anywhere under the given directories."""
    matches = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            if basename in files:
                matches.append(os.path.join(root, basename))
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="reports/hand_detection_report.csv")
    parser.add_argument("--quarantine_dir", default="data/quarantined_hand_images")
    parser.add_argument("--search_dirs", nargs="+",
                         default=["data/raw_healthy_sources", "data/raw_mastitis_sources"],
                         help="Fallback search locations if a recorded path doesn't resolve directly.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Preview what would be moved without actually moving anything.")
    args = parser.parse_args()

    print(f"Current working directory: {os.getcwd()}")

    if not os.path.exists(args.csv_path):
        print(f"Could not find {args.csv_path}. Run detect_hands_gloves.py first.")
        return

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    flagged_rows = [r for r in rows if r["flagged"].strip().lower() == "true"]
    print(f"{len(flagged_rows)} / {len(rows)} images flagged in the report.")

    # Diagnostic: show exactly what path resolution looks like for the first flagged row
    if flagged_rows:
        sample_raw = flagged_rows[0]["image_path"]
        sample_resolved = sample_raw.replace("\\", os.sep).replace("/", os.sep)
        print(f"\n[diagnostic] first flagged row, raw CSV value: {sample_raw!r}")
        print(f"[diagnostic] resolved path checked: {sample_resolved!r}")
        print(f"[diagnostic] absolute path checked: {os.path.abspath(sample_resolved)!r}")
        print(f"[diagnostic] exists? {os.path.exists(sample_resolved)}\n")

    moved, missing, recovered_by_search = 0, 0, 0

    for r in flagged_rows:
        src_path = r["image_path"].replace("\\", os.sep).replace("/", os.sep)

        if not os.path.exists(src_path):
            # Fallback: search by basename in case the recorded relative path
            # doesn't match the current working directory / folder layout.
            basename = os.path.basename(src_path)
            candidates = find_by_basename(basename, args.search_dirs)
            if len(candidates) == 1:
                src_path = candidates[0]
                recovered_by_search += 1
                print(f"  (recovered via basename search: {src_path})")
            elif len(candidates) > 1:
                print(f"  AMBIGUOUS (multiple matches, skipping): {basename} -> {candidates}")
                missing += 1
                continue
            else:
                print(f"  MISSING (not found by path or basename search): {src_path}")
                missing += 1
                continue

        # Preserve which source folder it came from (e.g. raw_healthy_sources /
        # raw_mastitis_sources) as a subfolder inside quarantine, so nothing
        # collides and you can see where each image originally lived.
        parent_folder_name = os.path.basename(os.path.dirname(src_path))
        dest_dir = os.path.join(args.quarantine_dir, parent_folder_name)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))

        if args.dry_run:
            print(f"  [dry run] would move: {src_path}  ->  {dest_path}")
        else:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(src_path, dest_path)
            print(f"  moved: {src_path}  ->  {dest_path}")
        moved += 1

    print(f"\n{'Would move' if args.dry_run else 'Moved'}: {moved}")
    if recovered_by_search:
        print(f"(of which {recovered_by_search} were found via basename fallback, not the recorded path)")
    if missing:
        print(f"Missing (skipped): {missing}")

    if not args.dry_run and moved > 0:
        print(f"\nQuarantined images are saved under: {args.quarantine_dir}/")
        print("Next steps:")
        print("  python scripts/prepare_dataset.py")
        print("  python scripts/train_teat_model.py")


if __name__ == "__main__":
    main()
