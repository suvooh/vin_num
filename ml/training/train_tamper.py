"""Train a plate-text tamper detector on synthetic edits of real plate photos.

    python -I -X utf8 training/train_tamper.py --images data/openimages/images --bbox data/openimages/val_bbox.csv --steps 1200 --out runs/tamper

Same patch network as the screen/print model (native-resolution 96x96, no resizing). Positives are
patches centred on a synthetic edit; negatives are patches from untouched plates, given the
same optional JPEG re-save. Validation uses photos never seen in training.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from synth.tamper import maybe_resave, tamper_plate
from torch import nn
from training.pad_model import PadNet


class TamperNet(nn.Module):
    """PadNet on a fixed high-pass residual. Edits hide in what a low-pass filter removes
    (noise level, resampling and blur seams), not in the picture content, so the content is
    subtracted first. The filter lives inside the model, so ONNX export is self-contained."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        k = (k[:, None] * k[None]) / 256.0
        w = k[None, None].repeat(3, 1, 1, 1)
        self.register_buffer("lp", w)
        self.core = PadNet(1.0)

    def forward(self, x):
        low = nn.functional.conv2d(nn.functional.pad(x, (2, 2, 2, 2), mode="replicate"), self.lp, groups=3)
        return self.core((0.5 + 8.0 * (x - low)).clamp(0, 1))

P = 96
PLATE = "/m/01jfm_"


def load_pool(images: Path, bbox: Path, min_w=50):
    gt = {}
    with open(bbox, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["LabelName"] == PLATE and r["Confidence"] == "1" and r["IsGroupOf"] == "0":
                gt.setdefault(r["ImageID"], []).append(tuple(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")))
    pool = []
    for p in sorted(images.glob("*.jpg")):
        if p.stem not in gt:
            continue
        a = np.asarray(Image.open(p).convert("RGB"))
        h, w = a.shape[:2]
        boxes = [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)) for x0, y0, x1, y1 in gt[p.stem]]
        boxes = [b for b in boxes if b[2] - b[0] >= min_w and b[3] - b[1] >= 14]
        if boxes and min(h, w) >= P:
            pool.append((a, boxes))
    return pool


def patch_at(img, cx, cy):
    h, w = img.shape[:2]
    x = int(np.clip(cx - P // 2, 0, w - P))
    y = int(np.clip(cy - P // 2, 0, h - P))
    return np.ascontiguousarray(img[y : y + P, x : x + P])


def make_sample(pool, rng, tampered: bool):
    img, boxes = pool[int(rng.integers(len(pool)))]
    b = boxes[int(rng.integers(len(boxes)))]
    if tampered:
        edited, rect, kind = tamper_plate(img, b, rng)
        out = maybe_resave(edited, rng)
        cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
        cx += int(rng.integers(-12, 13)); cy += int(rng.integers(-8, 9))
    else:
        out = maybe_resave(img, rng)
        cx = int(rng.integers(b[0] + 8, max(b[0] + 9, b[2] - 8)))
        cy = int(rng.integers(b[1] + 4, max(b[1] + 5, b[3] - 4)))
        kind = "real"
    return patch_at(out, cx, cy), kind


class Stream(IterableDataset):
    def __init__(self, pool, batch, seed, n):
        self.pool, self.bs, self.seed, self.n = pool, batch, seed, n

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, wid])
        for _ in range(self.n // nw):
            xs, ys = [], []
            for _ in range(self.bs):
                t = rng.random() < 0.5
                xs.append(make_sample(self.pool, rng, t)[0])
                ys.append(1.0 if t else 0.0)
            yield torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().div_(255), torch.tensor(ys)


def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, pool, n=400, seed=99):
    model.eval()
    rng = np.random.default_rng(seed)
    real = [make_sample(pool, rng, False)[0] for _ in range(n)]
    tam = [make_sample(pool, rng, True) for _ in range(n)]

    def sc(ps):
        x = torch.from_numpy(np.stack(ps)).permute(0, 3, 1, 2).float().div_(255)
        return torch.cat([model(x[i : i + 64]) for i in range(0, len(ps), 64)]).numpy()

    sr, st = sc(real), sc([t[0] for t in tam])
    thr = float(np.percentile(sr, 95))
    res = {"auc": auc(st, sr), "tpr@5%fp": float((st > thr).mean())}
    kinds = np.array([t[1] for t in tam])
    for k in ("glyph", "clone", "smear"):
        if (kinds == k).any():
            res[f"tpr_{k}"] = float((st[kinds == k] > thr).mean())
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--bbox", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pool = load_pool(a.images, a.bbox)
    cut = int(len(pool) * 0.75)
    train, val = pool[:cut], pool[cut:]
    print(f"plate photos: {len(train)} train / {len(val)} val (disjoint photos)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model = TamperNet()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    loader = DataLoader(Stream(train, a.batch, 7, a.steps + 5), batch_size=None, num_workers=a.workers, prefetch_factor=4)
    t0, run = time.perf_counter(), None
    for step, (x, y) in enumerate(loader, 1):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = loss.item() if run is None else 0.97 * run + 0.03 * loss.item()
        if step % 50 == 0:
            print(f"step {step:5d} loss {run:.4f} {(time.perf_counter() - t0) / step:.2f}s/step", flush=True)
        if step % 400 == 0 or step == a.steps or step == 200:
            m = evaluate(model, val)
            print("  VAL " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
            torch.save({"model": model.state_dict(), "width": 1.0, "step": step, "val": m}, a.out / "tamper.pt")
        if step >= a.steps:
            break
    print(f"done {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
