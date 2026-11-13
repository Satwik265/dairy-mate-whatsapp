"""
Image-level evaluation of the tick patch classifier through the exact app pipeline (predict.py: stretch to 640,
COCO animal crop, 96/48 grid, ONNX patch model).

    python tick/scripts/evaluate.py maps   --model tick_patch_A      # patch maps for val+test (+ external) -> cache
    python tick/scripts/evaluate.py select --models tick_patch_A tick_patch_B   # VAL ONLY: model + rule + calibration
    python tick/scripts/evaluate.py test                              # report the chosen setup once on TEST
    python tick/scripts/evaluate.py coat                              # coat-colour shortcut check on TEST patches
    python tick/scripts/evaluate.py gate                              # how the shared udder/teat gate routes tick photos

Image label: affected = the image has at least one Tick / Ticks_Cluster box (Griffith); healthy = Griffith images with
no tick box + presumed tick-free rf16 cows. NOTE the two sources differ in camera, farm and breed, so an AUC that
mixes them partly measures "is this the Griffith pen"; Griffith-only numbers are reported separately.

select: rules tried on val: max, top-k mean (k=2,3,5), count of patches >= t (t=0.5,0.7,0.9). Criterion: mean of
val ROC-AUC (all val images) and val ROC-AUC on Griffith images only. Calibration: Platt scaling on val with
class-balanced weights (equal prior odds), decision at calibrated p >= 0.5. Written to models/tick_model_card.json.
test: run once; confusion matrix, sensitivity/specificity, ROC-AUC, PR-AUC, 95% CIs from a group bootstrap.
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
from predict import (FRAME, AnimalDetector, PatchModel, aggregate, ensure_min_size, grid,  # noqa: E402
                     open_photo, pad_box, patch_map, quality_problem, to_frame)

CACHE = TICK / "data" / "eval_cache"
REPORTS = TICK / "reports"
CARD = TICK / "models" / "tick_model_card.json"
# External web photos of tick-infested cattle (sanity check only): tick_images/infested (13) and ticks_images.zip (23 files,
# extracted to .scratch/ticks_web). Exact byte duplicates are dropped (10.jpeg == 13.jpeg); several others are crops of
# the same source photo, so the effective n is smaller than the file count.
EXTERNAL = [ROOT / ".scratch" / "ticks_web" / "ticks_images", ROOT / "tick_images" / "infested"]


def external_files():
    import hashlib
    seen, out = set(), []
    for d in EXTERNAL:
        for p in (sorted(d.iterdir()) if d.exists() else []):
            h = hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None
            if h and h not in seen:
                seen.add(h)
                out.append(p)
    return out
RULES = ([{"name": "max"}] + [{"name": "topk_mean", "k": k} for k in (2, 3, 5)]
         + [{"name": "count_above", "t": t} for t in (0.5, 0.7, 0.9)])


def rule_str(r):
    return r["name"] + "".join(f"_{k}{v}" for k, v in r.items() if k != "name")


def transform(score, rule):
    if rule["name"] == "count_above":
        return float(np.log1p(score))
    s = min(max(score, 1e-6), 1 - 1e-6)
    return float(np.log(s / (1 - s)))


def roc_auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def pr_auc(y, s):
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, len(y) + 1)) * y).sum() / max(y.sum(), 1))


def fit_platt(x, y, ridge=1e-2, iters=200):
    """Logistic regression p = sigmoid(A x + B), class-balanced weights (equal prior odds), small ridge on A so a
    nearly separable val set can't send A to infinity. Damped Newton with backtracking on the penalised loss."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean())) / len(y)

    def loss(A, B):
        z = A * x + B
        return float((w * (np.logaddexp(0, z) - y * z)).sum() + 0.5 * ridge * A * A)

    A, B = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A * x + B)))
        g = np.array([(w * (p - y) * x).sum() + ridge * A, (w * (p - y)).sum()])
        h = w * p * (1 - p)
        H = np.array([[(h * x * x).sum() + ridge, (h * x).sum()], [(h * x).sum(), h.sum() + 1e-12]])
        step = np.linalg.solve(H, g)
        t, L0 = 1.0, loss(A, B)
        while loss(A - t * step[0], B - t * step[1]) > L0 and t > 1e-6:
            t /= 2
        A, B = A - t * step[0], B - t * step[1]
        if np.abs(t * step).max() < 1e-9:
            break
    return float(A), float(B)


def griffith_tick_centres(path):
    out = []
    txt = Path(path).with_suffix(".txt")
    if txt.exists():
        for line in txt.read_text().splitlines():
            v = line.split()
            if v and int(v[0]) in (1, 2):
                out.append((float(v[1]) * FRAME, float(v[2]) * FRAME))
    return np.array(out).reshape(-1, 2)


# ------------------------------------------------------------------ maps
def stage_maps(args):
    det = AnimalDetector()
    model = PatchModel(TICK / "models" / f"{args.model}.onnx", threads=4)
    rows = [r for r in csv.DictReader(open(TICK / "curation" / "split_manifest.csv")) if r["split"] in ("val", "test")]
    ext = external_files()
    rows += [dict(source="external_web", split="external", group="web_" + p.stem, stem=p.stem, has_ticks="1",
                  split_image=str(p.relative_to(ROOT))) for p in ext]
    out = []
    for i, r in enumerate(rows):
        img = open_photo(ROOT / r["split_image"])
        frame = to_frame(img)
        found = det.best_animal(frame)
        box = pad_box(found[2], FRAME, FRAME) if found else (0, 0, FRAME, FRAME)
        wins, probs = patch_map(frame, box, model)
        wins = np.array(wins)
        if r["source"] == "griffith":
            c = griffith_tick_centres(ROOT / r["split_image"])
            plab = np.array([bool(((c[:, 0] >= a) & (c[:, 0] < cc) & (c[:, 1] >= b) & (c[:, 1] < d)).any())
                             for a, b, cc, d in wins]) if len(c) else np.zeros(len(wins), bool)
            n_ticks_in_crop = int(((c[:, 0] >= box[0]) & (c[:, 0] < box[2]) & (c[:, 1] >= box[1]) & (c[:, 1] < box[3])).sum()) if len(c) else 0
        else:
            plab = np.zeros(len(wins), bool) if r["source"] == "rf16_cows" else np.full(len(wins), -1)
            n_ticks_in_crop = -1
        n_ticks = len(c) if r["source"] == "griffith" else -1
        out.append(dict(source=r["source"], split=r["split"], group=r["group"], stem=r["stem"],
                        label=int(r["has_ticks"]), animal_found=found is not None,
                        animal_class=found[1] if found else -1, box=list(box), probs=probs.astype(np.float32),
                        patch_labels=plab.astype(np.int8), n_ticks=n_ticks, n_ticks_in_crop=n_ticks_in_crop))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{args.model}.npy", np.array(out, dtype=object), allow_pickle=True)
    print("cached", len(out), "images ->", CACHE / f"{args.model}.npy")


def load(model):
    return list(np.load(CACHE / f"{model}.npy", allow_pickle=True))


def image_metrics(items, scores, decide):
    y = np.array([it["label"] for it in items])
    pred = np.array([decide(s) for s in scores])
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    return dict(n=len(y), n_affected=int(y.sum()), n_healthy=int((1 - y).sum()),
                roc_auc=roc_auc(y, scores), pr_auc=pr_auc(y, scores),
                confusion={"tp": tp, "fn": fn, "tn": tn, "fp": fp},
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))


def patch_metrics(items, thr=0.5):
    out = {}
    for name, sel in (("griffith", [it for it in items if it["source"] == "griffith"]),
                      ("rf16_cows", [it for it in items if it["source"] == "rf16_cows"])):
        if not sel:
            continue
        p = np.concatenate([it["probs"] for it in sel])
        l = np.concatenate([it["patch_labels"] for it in sel]).astype(int)
        d = dict(n_patches=int(len(p)), n_positive=int(l.sum()), fp_rate_at_0_5=float(((p >= thr) & (l == 0)).sum() / max((l == 0).sum(), 1)))
        if l.sum():
            d.update(pr_auc=pr_auc(l, p), roc_auc=roc_auc(l, p),
                     precision_at_0_5=float(((p >= thr) & (l == 1)).sum() / max((p >= thr).sum(), 1)),
                     recall_at_0_5=float(((p >= thr) & (l == 1)).sum() / l.sum()))
        out[name] = d
    return out


def stage_select(args):
    results, best = [], None
    for model in args.models:
        items = [it for it in load(model) if it["split"] == "val"]
        y = np.array([it["label"] for it in items])
        g = np.array([it["source"] == "griffith" for it in items])
        for rule in RULES:
            s = np.array([aggregate(it["probs"], rule) for it in items])
            a_all, a_g = roc_auc(y, s), roc_auc(y[g], s[g])
            row = dict(model=model, rule=rule_str(rule), val_roc_auc=a_all, val_roc_auc_griffith_only=a_g,
                       val_pr_auc=pr_auc(y, s), criterion=(a_all + a_g) / 2)
            results.append(row)
            print(f"{model:14s} {rule_str(rule):16s} AUC all {a_all:.3f}  griffith-only {a_g:.3f}  PR-AUC {row['val_pr_auc']:.3f}")
            if best is None or row["criterion"] > best[0]["criterion"]:
                best = (row, rule, model)
    row, rule, model = best
    items = [it for it in load(model) if it["split"] == "val"]
    y = np.array([it["label"] for it in items])
    x = np.array([transform(aggregate(it["probs"], rule), rule) for it in items])
    A, B = fit_platt(x, y)
    cal = 1 / (1 + np.exp(-(A * x + B)))
    val = image_metrics(items, cal, lambda p: int(p >= 0.5))
    gsel = [i for i, it in enumerate(items) if it["source"] == "griffith"]
    val_g = image_metrics([items[i] for i in gsel], cal[gsel], lambda p: int(p >= 0.5))
    print(f"\nchosen: {model} {rule_str(rule)}  A={A:.4f} B={B:.4f}")
    print("val (calibrated, p>=0.5):", json.dumps(val))
    card = {
        "model": f"{model}.onnx",
        "animal_detector": "animal_detector.onnx",
        "task": "photo of cattle -> P(ticks visible)",
        "pipeline": "stretch photo to 640x640; COCO animal box + 10% pad (whole photo if none); 96x96 patches, stride 48",
        "patch_input": "RGB 96x96, (x - ImageNet mean*255) / (std*255), NCHW float32; output logit",
        "aggregation": rule,
        "score_transform": "log1p" if rule["name"] == "count_above" else "logit",
        "calibration": {"A": A, "B": B, "fitted_on": "val split (class-balanced, equal prior odds)"},
        "decision": "affected if calibrated p >= 0.5",
        "patch_threshold": 0.5,
        "confidence_bands": [["high", 0.85], ["medium", 0.65], ["low", 0.0]],
        "selection": "model, rule and calibration chosen on val only; test reported once (reports/tick_evaluation.json)",
    }
    json.dump(card, open(CARD, "w"), indent=2)
    REPORTS.mkdir(exist_ok=True)
    json.dump(dict(candidates=results, chosen=dict(model=model, rule=rule, A=A, B=B),
                   val_image_level=val, val_image_level_griffith_only=val_g,
                   val_patch_level=patch_metrics(items)),
              open(REPORTS / "tick_selection_val.json", "w"), indent=2)
    print("wrote", CARD, "and", REPORTS / "tick_selection_val.json")


def group_bootstrap(items, cal, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = sorted({it["group"] for it in items})
    idx = {g: [i for i, it in enumerate(items) if it["group"] == g] for g in groups}
    y = np.array([it["label"] for it in items])
    stats = {"roc_auc": [], "sensitivity": [], "specificity": []}
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(groups, len(groups), replace=True)])
        yy, pp = y[pick], cal[pick]
        if yy.min() == yy.max():
            continue
        stats["roc_auc"].append(roc_auc(yy, pp))
        stats["sensitivity"].append(((pp >= 0.5) & (yy == 1)).sum() / (yy == 1).sum())
        stats["specificity"].append(((pp < 0.5) & (yy == 0)).sum() / (yy == 0).sum())
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in stats.items()}


def stage_test(args):
    card = json.load(open(CARD))
    model, rule = card["model"].replace(".onnx", ""), card["aggregation"]
    A, B = card["calibration"]["A"], card["calibration"]["B"]
    allitems = load(model)

    def cal_of(items):
        return np.array([1 / (1 + np.exp(-(A * transform(aggregate(it["probs"], rule), rule) + B))) for it in items])

    test = [it for it in allitems if it["split"] == "test"]
    cal = cal_of(test)
    rep = {"model": model, "rule": rule, "test_image_level": image_metrics(test, cal, lambda p: int(p >= 0.5)),
           "test_image_level_ci95_group_bootstrap": group_bootstrap(test, cal)}
    for src in ("griffith", "rf16_cows"):
        sel = [i for i, it in enumerate(test) if it["source"] == src]
        rep[f"test_image_level_{src}_only"] = image_metrics([test[i] for i in sel], cal[sel], lambda p: int(p >= 0.5))
    rep["test_patch_level"] = patch_metrics(test)
    rep["test_animal_found_rate"] = {src: f"{sum(it['animal_found'] for it in test if it['source'] == src)}/"
                                          f"{sum(1 for it in test if it['source'] == src)}" for src in ("griffith", "rf16_cows")}
    gt = [it for it in test if it["source"] == "griffith"]
    rep["test_tick_boxes_outside_animal_crop"] = f"{sum(it['n_ticks'] - it['n_ticks_in_crop'] for it in gt)}/{sum(it['n_ticks'] for it in gt)}"
    rep["test_per_image"] = [dict(source=it["source"], group=it["group"], stem=it["stem"], label=it["label"],
                                  p=round(float(c), 4), animal_found=bool(it["animal_found"]))
                             for it, c in zip(test, cal)]
    keep = {p.stem for p in external_files()}
    ext = [it for it in allitems if it["split"] == "external" and it["stem"] in keep]
    ecal = cal_of(ext)
    rep["external_web_sanity"] = dict(
        note="web photos of tick-infested cattle (one is a goat with ringworm; exact duplicates dropped; no tick-free web photos, so only misses count); sanity check, not validation",
        n=len(ext), called_affected=int((ecal >= 0.5).sum()), animal_found=int(sum(it["animal_found"] for it in ext)),
        per_image=[dict(stem=it["stem"], p=round(float(c), 4), animal_found=bool(it["animal_found"])) for it, c in zip(ext, ecal)])
    json.dump(rep, open(REPORTS / "tick_evaluation.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("test_per_image", "external_web_sanity")}, indent=1))
    print("external:", rep["external_web_sanity"]["called_affected"], "/", len(ext), "called affected")


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def stage_coat(args):
    """Coat-colour shortcut check (repeats the Colab cell): on TEST patches, correlate the predicted patch probability
    with patch 'brownness' = mean(R - B), and the true patch label with brownness. pred >> label -> colour shortcut."""
    card = json.load(open(CARD))
    model = card["model"].replace(".onnx", "")
    rows = {(r["split"], r["stem"]): r for r in csv.DictReader(open(TICK / "curation" / "split_manifest.csv"))}
    test = [it for it in load(model) if it["split"] == "test"]
    P, L, Bn, S = [], [], [], []
    for it in test:
        frame = to_frame(open_photo(ROOT / rows[("test", it["stem"])]["split_image"]))
        x0, y0, x1, y1 = it["box"]
        crop_ = np.asarray(ensure_min_size(frame.crop((x0, y0, x1, y1)))).astype(np.float32)
        wins = grid(crop_.shape[1], crop_.shape[0])
        assert len(wins) == len(it["probs"]), it["stem"]
        Bn += [float((crop_[b:d, a:c, 0] - crop_[b:d, a:c, 2]).mean()) for a, b, c, d in wins]
        P += list(it["probs"]); L += list(it["patch_labels"]); S += [it["source"]] * len(wins)
    P, L, Bn, S = np.array(P), np.array(L), np.array(Bn), np.array(S)
    out = {"definition": "brownness = mean(R - B) over the 96x96 patch (same as the Colab check); Pearson r "
                         "(point-biserial for the 0/1 label); test split, every grid patch the app scores",
           "colab_reference": {"pred_vs_brownness": -0.167, "label_vs_brownness": -0.247,
                               "note": "old Colab patch set (capped negatives, leaky 152-group split); not the same patches"}}
    for name, sel in (("griffith_test_patches", S == "griffith"), ("all_test_patches_incl_rf16_cows", np.ones(len(S), bool))):
        d = dict(n_patches=int(sel.sum()), n_positive=int((L[sel] == 1).sum()),
                 pred_vs_brownness=round(pearson(P[sel], Bn[sel]), 3))
        if (L[sel] == 1).any():
            d["label_vs_brownness"] = round(pearson(L[sel], Bn[sel]), 3)
            d["pred_vs_brownness_within_negatives"] = round(pearson(P[sel & (L == 0)], Bn[sel & (L == 0)]), 3)
            d["pred_vs_brownness_within_positives"] = round(pearson(P[sel & (L == 1)], Bn[sel & (L == 1)]), 3)
        out[name] = d
        print(name, d)
    json.dump(out, open(REPORTS / "tick_coat_colour_check.json", "w"), indent=2)
    print("wrote", REPORTS / "tick_coat_colour_check.json")


def stage_gate(args):
    """How the shared udder/teat photo gate (common/gate.py) would route tick photos (test split + external)."""
    from common.gate import PhotoGate
    gate = PhotoGate()
    rows = [r for r in csv.DictReader(open(TICK / "curation" / "split_manifest.csv")) if r["split"] == "test"]
    ext = external_files()
    sets = {"griffith_test": [ROOT / r["split_image"] for r in rows if r["source"] == "griffith"],
            "rf16_cows_test": [ROOT / r["split_image"] for r in rows if r["source"] == "rf16_cows"],
            "external_web": ext}
    out = {"note": "routing the shared gate WOULD give; the tick app does not use it to reject or redirect photos"}
    for name, paths in sets.items():
        counts = {"udder": 0, "teat": 0, "retake": 0}
        for p in paths:
            img = open_photo(p)
            where = None if quality_problem(img) else gate.route(gate.detect(img))
            counts[where or "retake"] += 1
        out[name] = dict(n=len(paths), **{k: round(v / max(len(paths), 1), 3) for k, v in counts.items()}, counts=counts)
        print(name, out[name])
    json.dump(out, open(REPORTS / "tick_gate_routing.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["maps", "select", "test", "coat", "gate"])
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    {"maps": stage_maps, "select": stage_select, "test": stage_test, "coat": stage_coat,
     "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
