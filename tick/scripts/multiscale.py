"""
Multi-scale scoring experiment for the tick patch model (no retraining).

Why: the patch model learned ticks at ~4x6 px (Griffith pen footage stretched to 640x640). In close-up phone/web
photos ticks are 30-80 px in the same 640 frame, and the model scores them near 0 (4/34 web photos caught).
Here the animal crop is also scored at 1/2, 1/4 and 1/8 size, so close-up ticks shrink towards the training scale.
Image score = mean of the top-5 patch scores pooled over all scales; Platt calibration refit on val only.

Settings are fixed in advance (not tuned on the web photos, which are the only close-ups with ticks we have).
Tick-free close-ups don't exist in the tick data, so false alarms on close-ups are estimated on presumed
tick-free cow close-ups from the other two models: SCTL teat photos (score 1-2, test split) and healthy udders.

    ..\\venv_yolo\\Scripts\\python tick/scripts/multiscale.py maps  --model tick_patch_B
    ..\\venv_yolo\\Scripts\\python tick/scripts/multiscale.py report --model tick_patch_B
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TICK = Path(__file__).resolve().parents[1]
ROOT = TICK.parent
sys.path.insert(0, str(TICK / "scripts"))
sys.path.insert(0, str(ROOT))
from predict import FRAME, AnimalDetector, PatchModel, aggregate, open_photo, pad_box, patch_map, to_frame  # noqa: E402
from evaluate import external_files, fit_platt, group_bootstrap, image_metrics, roc_auc  # noqa: E402

SCALES = (1.0, 0.5, 0.25, 0.125)
RULE = {"name": "topk_mean", "k": 5}
CACHE = TICK / "data" / "eval_cache"
REPORTS = TICK / "reports"
CLOSEUP_NEG = {
    "teat_sctl_healthy_test": None,                      # filled from teat/curation/teat_v2_manifest.csv
    "udder_healthy": ROOT / "udder_images" / "healthy",
}


def closeup_negatives():
    out = []
    for r in csv.DictReader(open(ROOT / "teat" / "curation" / "teat_v2_manifest.csv")):
        if r["source"] == "sctl" and r["split"] == "test" and r["label"] == "healthy":
            out.append(("teat_sctl_healthy_test", ROOT / "teat" / "data" / "processed_v2" / "test" / "healthy" / f"{r['id']}.jpg"))
    out += [("udder_healthy", p) for p in sorted(CLOSEUP_NEG["udder_healthy"].iterdir()) if p.is_file()]
    return out


def scale_maps(frame, box, model):
    """Patch probabilities for the crop at every scale -> {scale: probs}."""
    crop = frame.crop(box)
    out = {}
    for s in SCALES:
        if s == 1.0:
            out[s] = patch_map(frame, box, model)[1]
            continue
        w, h = crop.size
        small = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        out[s] = patch_map(small, (0, 0) + small.size, model)[1]
    return out


def stage_maps(args):
    det = AnimalDetector()
    model = PatchModel(TICK / "models" / f"{args.model}.onnx", threads=4)
    rows = [dict(source=r["source"], split=r["split"], group=r["group"], label=int(r["has_ticks"]), path=ROOT / r["split_image"])
            for r in csv.DictReader(open(TICK / "curation" / "split_manifest.csv")) if r["split"] in ("val", "test")]
    rows += [dict(source="external_web", split="external", group="web_" + p.stem, label=1, path=p) for p in external_files()]
    rows += [dict(source=kind, split="closeup_neg", group=kind + "_" + p.stem, label=0, path=p) for kind, p in closeup_negatives()]
    out = []
    for i, r in enumerate(rows):
        frame = to_frame(open_photo(r["path"]))
        found = det.best_animal(frame)
        box = pad_box(found[2], FRAME, FRAME) if found else (0, 0, FRAME, FRAME)
        maps = scale_maps(frame, box, model)
        out.append(dict(r, path=str(r["path"]), animal_found=found is not None,
                        probs={s: p.astype(np.float32) for s, p in maps.items()}))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{args.model}_multiscale.npy", np.array(out, dtype=object), allow_pickle=True)
    print("cached", len(out), "images")


def score(it, scales):
    return aggregate(np.concatenate([it["probs"][s] for s in scales]), RULE)


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def stage_report(args):
    items = list(np.load(CACHE / f"{args.model}_multiscale.npy", allow_pickle=True))
    configs = {"1x only (current app)": (1.0,), "1x + 1/2 + 1/4 + 1/8": SCALES}
    rep = {"model": args.model, "rule": RULE, "configs": {}}
    for name, scales in configs.items():
        by = lambda split: [it for it in items if it["split"] == split]
        val = by("val")
        xv = logit(np.array([score(it, scales) for it in val]))
        A, B = fit_platt(xv, np.array([it["label"] for it in val]))
        cal = lambda its: 1 / (1 + np.exp(-(A * logit(np.array([score(it, scales) for it in its])) + B)))
        c = dict(calibration=dict(A=A, B=B))
        for split in ("val", "test"):
            its = by(split)
            p = cal(its)
            c[split] = image_metrics(its, p, lambda q: int(q >= 0.5))
            g = [i for i, it in enumerate(its) if it["source"] == "griffith"]
            c[split + "_griffith_only"] = image_metrics([its[i] for i in g], p[g], lambda q: int(q >= 0.5))
            if split == "test":
                c["test_ci95_group_bootstrap"] = group_bootstrap(its, p)
        ext = by("external")
        pe = cal(ext)
        c["external_web_caught"] = f"{int((pe >= 0.5).sum())}/{len(ext)}"
        c["external_web_per_image"] = {Path(it["path"]).name: round(float(q), 3) for it, q in zip(ext, pe)}
        neg = by("closeup_neg")
        pn = cal(neg)
        for kind in CLOSEUP_NEG:
            sel = [q for it, q in zip(neg, pn) if it["source"] == kind]
            c[f"false_alarms_{kind}"] = f"{int((np.array(sel) >= 0.5).sum())}/{len(sel)}"
        # which scale gives each image's top patch (does the pyramid fire at small scales on tick-free close-ups?)
        rep["configs"][name] = c
        print(f"\n== {name}  (A={A:.3f} B={B:.3f})")
        for k in ("val", "test", "test_griffith_only"):
            m = c[k]
            print(f"  {k:20s} AUC {m['roc_auc']:.3f}  sens {m['sensitivity']:.2f} ({m['confusion']['tp']}/{m['n_affected']})  "
                  f"spec {m['specificity']:.2f} ({m['confusion']['tn']}/{m['n_healthy']})")
        print("  test CI95:", c["test_ci95_group_bootstrap"])
        print("  external web caught:", c["external_web_caught"])
        for kind in CLOSEUP_NEG:
            print(f"  false alarms {kind}: {c[f'false_alarms_{kind}']}")
    # top-scoring scale per image for the external photos
    rep["external_top_scale"] = {Path(it["path"]).name: float(max(SCALES, key=lambda s: aggregate(it["probs"][s], RULE)))
                                 for it in items if it["split"] == "external"}
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{args.model}_multiscale.json"
    json.dump(rep, open(out, "w"), indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["maps", "report"])
    ap.add_argument("--model", default="tick_patch_B")
    a = ap.parse_args()
    {"maps": stage_maps, "report": stage_report}[a.stage](a)
