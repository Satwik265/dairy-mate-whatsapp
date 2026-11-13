"""
Train the tick patch classifier ("is there a tick in this 96x96 patch?"): MobileNetV2 (ImageNet), PyTorch.

    python tick/scripts/train_patch_classifier.py --name tick_patch_A --rf-neg-frac 0.0
    python tick/scripts/train_patch_classifier.py --name tick_patch_B --rf-neg-frac 0.3
    python tick/scripts/train_patch_classifier.py --name tick_patch_C --rf-neg-frac 0.3 --scale-max 6

Ported from the Colab ticks_train_patch_classifier.py (TF, frozen base). Changes and why:
  - PyTorch, because TensorFlow on this Windows machine is CPU-only and torch cu126 uses the GPU.
  - Patches are sampled on the fly from cattle crops instead of being written to disk with a 40-negatives-per-image cap.
    The Colab patch metrics (P 0.72 / R 0.83) were on that capped negative set, so precision was flattered;
    here validation uses EVERY grid patch (stride 48) of each val crop, as the app does.
  - Mild scale jitter (window 0.8x-1.4x of 96 px; --scale-max widens it) and flips/rot90/colour jitter.
  - Optional negatives from presumed tick-free cows (rf16, train split only): --rf-neg-frac of the negative samples.

Training crops use the ground-truth Cattle box (Griffith) / cow boxes (rf16), padded 10%, in the 640x640 frame.
Positive patch = a Tick or Ticks_Cluster box centre falls inside the window (same rule as the Colab builder).
Checkpoint = best patch-level PR-AUC on the val grid (Griffith val patches + up to 20 grid patches per rf16 val cow).
Outputs (tick/models/, gitignored): <name>.pt, <name>.onnx; history -> tick/reports/<name>_history.json.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

TICK = Path(__file__).resolve().parents[1]
ROOT = TICK.parent
sys.path.insert(0, str(TICK / "scripts"))
from predict import PATCH, STRIDE, MEAN, STD, grid, pad_box, ensure_min_size  # noqa: E402

TICK_CLASSES = {1, 2}           # Griffith: 0 Cattle, 1 Tick, 2 Ticks_Cluster
GRIFFITH_CATTLE = 0


def read_boxes(txt, w, h):
    out = []
    if not Path(txt).exists():
        return out
    for line in Path(txt).read_text().splitlines():
        if line.strip():
            v = line.split()
            c, nums = int(v[0]), [float(t) for t in v[1:]]
            if len(nums) == 4:                       # YOLO box
                xc, yc, bw, bh = nums[0] * w, nums[1] * h, nums[2] * w, nums[3] * h
                out.append((c, (xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2)))
            else:                                    # YOLO polygon (some rf16 labels) -> its bounding box
                xs, ys = [t * w for t in nums[0::2]], [t * h for t in nums[1::2]]
                out.append((c, (min(xs), min(ys), max(xs), max(ys))))
    return out


def load_items(split, sources):
    """-> list of dict(img=PIL 640 frame, crops=[padded boxes], ticks=np.array of centres, source)."""
    items = []
    for r in csv.DictReader(open(TICK / "curation" / "split_manifest.csv")):
        if r["split"] != split or r["source"] not in sources:
            continue
        path = ROOT / r["split_image"]
        img = Image.open(path).convert("RGB")
        if img.size != (640, 640):
            img = img.resize((640, 640), Image.BILINEAR)
        boxes = read_boxes(path.with_suffix(".txt"), 640, 640)
        if r["source"] == "griffith":
            cattle = [b for c, b in boxes if c == GRIFFITH_CATTLE]
            if cattle:   # one crop around all cattle boxes
                crops = [pad_box((min(b[0] for b in cattle), min(b[1] for b in cattle),
                                  max(b[2] for b in cattle), max(b[3] for b in cattle)), 640, 640)]
            else:
                crops = [(0, 0, 640, 640)]
            ticks = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for c, b in boxes if c in TICK_CLASSES]).reshape(-1, 2)
        else:
            crops = [pad_box(b, 640, 640) for c, b in boxes]
            ticks = np.zeros((0, 2))
        crops = [c for c in crops if c[2] - c[0] >= 24 and c[3] - c[1] >= 24] or [(0, 0, 640, 640)]
        items.append(dict(img=img, crops=crops, ticks=ticks, source=r["source"], stem=r["stem"]))
    return items


def inside(ticks, x0, y0, x1, y1):
    if len(ticks) == 0:
        return False
    return bool(((ticks[:, 0] >= x0) & (ticks[:, 0] < x1) & (ticks[:, 1] >= y0) & (ticks[:, 1] < y1)).any())


def augment(patch, rng):
    if rng.random() < 0.5:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.5:
        patch = patch.transpose(Image.ROTATE_90)
    patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.8, 1.2))
    patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(0.8, 1.2))
    patch = ImageEnhance.Color(patch).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.15:
        patch = patch.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return patch


class Sampler:
    def __init__(self, items, rf_neg_frac, scale=(0.8, 1.4), seed=0):
        self.rng = random.Random(seed)
        self.grif = [it for it in items if it["source"] == "griffith"]
        self.rf = [it for it in items if it["source"] != "griffith"]
        self.pos = [(i, t) for i, it in enumerate(self.grif) for t in it["ticks"]]
        self.rf_neg_frac = rf_neg_frac if self.rf else 0.0
        self.scale = scale

    def _window(self, side, cx, cy):
        x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
        x0 = min(max(x0, 0), 640 - side); y0 = min(max(y0, 0), 640 - side)
        return x0, y0, x0 + side, y0 + side

    def sample(self, positive):
        rng = self.rng
        s = math.exp(rng.uniform(math.log(self.scale[0]), math.log(self.scale[1])))
        side = int(round(PATCH / s))
        if positive:
            i, (tx, ty) = self.pos[rng.randrange(len(self.pos))]
            it = self.grif[i]
            cx = tx + rng.uniform(-0.35, 0.35) * side
            cy = ty + rng.uniform(-0.35, 0.35) * side
            win = self._window(side, cx, cy)
            if not inside(it["ticks"], *win):      # clipping at the border moved the tick out: centre on it
                win = self._window(side, tx, ty)
            label = 1
        else:
            pool = self.rf if rng.random() < self.rf_neg_frac else self.grif
            for _ in range(50):
                it = pool[rng.randrange(len(pool))]
                c = it["crops"][rng.randrange(len(it["crops"]))]
                cx, cy = rng.uniform(c[0], c[2]), rng.uniform(c[1], c[3])
                win = self._window(side, cx, cy)
                if not inside(it["ticks"], *win):
                    break
            label = 0
        patch = it["img"].crop(win).resize((PATCH, PATCH), Image.BILINEAR)
        return np.asarray(augment(patch, rng)), label

    def batch(self, n, pos_frac=0.5):
        labels = [1 if self.rng.random() < pos_frac else 0 for _ in range(n)]
        xs, ys = zip(*(self.sample(l == 1) for l in labels))
        return np.stack(xs), np.array(ys, dtype=np.float32)


def val_grid(items, rf_cap=20, seed=0):
    """Every grid patch of each Griffith val crop; up to rf_cap random grid patches per rf16 val cow."""
    rng = random.Random(seed)
    X, Y, src = [], [], []
    for it in items:
        for c in it["crops"][:1] if it["source"] == "griffith" else it["crops"]:
            crop = ensure_min_size(it["img"].crop(c))
            arr = np.asarray(crop)
            wins = grid(*crop.size)
            if it["source"] != "griffith" and len(wins) > rf_cap:
                wins = rng.sample(wins, rf_cap)
            for a, b, cc, d in wins:
                X.append(arr[b:d, a:cc])
                Y.append(int(inside(it["ticks"], a + c[0], b + c[1], cc + c[0], d + c[1])))
                src.append(it["source"])
    return np.stack(X), np.array(Y), np.array(src)


def pr_auc(y, s):
    """Average precision (step-wise), same definition as sklearn.metrics.average_precision_score."""
    order = np.argsort(-s, kind="mergesort")
    y = np.asarray(y)[order]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def build_model():
    import torch
    import torchvision
    m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(m.last_channel, 1))
    return m


def to_tensor(x, device):
    import torch
    t = torch.from_numpy(((x.astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2).copy())
    return t.to(device)


def predict_np(model, X, device, batch=256):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(to_tensor(X[i:i + batch], device))).cpu().numpy().ravel())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rf-neg-frac", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-max", type=float, default=1.4,
                    help="largest zoom of the training window (1.4 = A/B; 6 = ticks up to close-up size, model C)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = {"griffith", "rf16_cows"} if args.rf_neg_frac > 0 else {"griffith"}
    train_items = load_items("train", sources)
    val_items = load_items("val", {"griffith", "rf16_cows"})
    sampler = Sampler(train_items, args.rf_neg_frac, scale=(0.8, args.scale_max), seed=args.seed)
    Xv, Yv, Sv = val_grid(val_items)
    print(f"device={device} train images={len(train_items)} tick centres={len(sampler.pos)}; "
          f"val grid patches={len(Xv)} (positive {Yv.sum()}, griffith {int((Sv == 'griffith').sum())})", flush=True)

    model = build_model().to(device)
    head = list(model.classifier.parameters())
    base = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": base, "lr": args.lr}, {"params": head, "lr": args.lr * 3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 3],
                                                total_steps=args.epochs * args.steps, pct_start=0.15)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, history = -1, []
    out_pt = TICK / "models" / f"{args.name}.pt"
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tl = 0
        for _ in range(args.steps):
            x, y = sampler.batch(args.batch)
            logits = model(to_tensor(x, device)).squeeze(1)
            loss = lossf(logits, torch.from_numpy(y).to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item()
        pv = predict_np(model, Xv, device)
        g = Sv == "griffith"
        ap_all, ap_g = pr_auc(Yv, pv), pr_auc(Yv[g], pv[g])
        fp_rf = float((pv[~g] >= 0.5).mean())
        rec = float((pv[Yv == 1] >= 0.5).mean())
        prec = float(((pv >= 0.5) & (Yv == 1)).sum() / max((pv >= 0.5).sum(), 1))
        history.append(dict(epoch=ep, loss=tl / args.steps, val_pr_auc=ap_all, val_pr_auc_griffith=ap_g,
                            val_precision_at_0_5=prec, val_recall_at_0_5=rec, val_rf16_patch_fp_rate=fp_rf))
        print(f"ep {ep:2d} loss {tl / args.steps:.4f} valPR-AUC {ap_all:.3f} (griffith {ap_g:.3f}) "
              f"P {prec:.3f} R {rec:.3f} rf16 FP {fp_rf:.4f}  {time.time() - t0:.0f}s", flush=True)
        if ap_all > best:
            best = ap_all
            torch.save(model.state_dict(), out_pt)

    model.load_state_dict(torch.load(out_pt, map_location=device))
    model.eval().cpu()
    onnx_path = TICK / "models" / f"{args.name}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, PATCH, PATCH), str(onnx_path), input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}}, opset_version=17, dynamo=False)
    (TICK / "reports").mkdir(exist_ok=True)
    json.dump(dict(args=vars(args), best_val_pr_auc=best, n_val_patches=int(len(Xv)), n_val_pos=int(Yv.sum()),
                   history=history), open(TICK / "reports" / f"{args.name}_history.json", "w"), indent=2)
    print("best val PR-AUC", round(best, 4), "->", out_pt, onnx_path)


if __name__ == "__main__":
    main()
