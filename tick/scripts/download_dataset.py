"""
Download the Griffith University "Ticks" dataset (Roboflow Universe, CC BY 4.0) into tick/data/raw/.

    python tick/scripts/download_dataset.py [--version 1] [--force]

Uses the Roboflow REST API directly (no `roboflow` package, so nothing gets installed into the shared venvs).
The API key is read from the ROBOFLOW_API_KEY environment variable, or from tick/data/roboflow_key.txt
(tick/data/ is gitignored). Never put the key in the repo.

Why version 1: every published version is augmented. v1 is the mildest (stretch to 640x640, plus 2 outputs per
train image with 0.34% noise). Both copies of a train image keep the same original filename before ".rf.<hash>",
so build_split.py keeps one copy per original filename.
"""

import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests

TICK = Path(__file__).resolve().parents[1]
WORKSPACE, PROJECT = "judyyanggriffitheduau", "ticks-oswus"
API = "https://api.roboflow.com"


def api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        f = TICK / "data" / "roboflow_key.txt"
        if f.exists():
            key = f.read_text().strip()
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY or put the key in tick/data/roboflow_key.txt")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--format", default="yolov11")
    ap.add_argument("--out", default=str(TICK / "data" / "raw"))
    ap.add_argument("--force", action="store_true", help="delete and re-download an existing export")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.force:
            print(f"{out} already has files; use --force to re-download")
            return
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    url = f"{API}/{WORKSPACE}/{PROJECT}/{args.version}/{args.format}"
    r = requests.get(url, params={"api_key": api_key()}, timeout=120)
    r.raise_for_status()
    link = r.json()["export"]["link"]
    print("downloading export zip ...")
    z = requests.get(link, timeout=600)
    z.raise_for_status()
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
    for split in ("train", "valid", "test"):
        n = len(list((out / split / "images").glob("*"))) if (out / split / "images").exists() else 0
        print(f"  {split}: {n} images")
    print("wrote", out)


if __name__ == "__main__":
    main()
