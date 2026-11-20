"""Train the recapture (presentation-attack) classifier on simulated attacks of real photos.

    python -I -X utf8 -m training.train_pad ...   # (use the path form below under -I)
    python -I -X utf8 training/train_pad.py --images data/openimages/images --steps 3000 --out runs/pad

Honest framing: bona fide = real photos (Open Images, CC BY 2.0) through a camera chain;
attack = SIMULATED screen/print photograph of a different photo. Validation uses photos the
model never saw. Two splits are reported:
  in-dist   screens at display pitch 0.8-2.0 (seen range)
  ood-pitch screens at pitch 2.0-3.2 (never seen in training)  <- the generalisation signal
A good number here means the pipeline works and the cue is learnable. It does NOT mean it detects
attacks made with real phones/monitors; that needs real attack captures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # python -I drops cwd from sys.path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from synth.recapture import bona_fide, print_recapture, screen_recapture
from training.pad_model import PadNet

PATCH, SRC = 96, 128


def load_pool(folder: Path) -> list[np.ndarray]:
    out = []
    for p in sorted(folder.glob("*.jpg")):
        try:
            im = np.asarray(Image.open(p).convert("RGB"))
            if min(im.shape[:2]) >= SRC:
                out.append(im)
        except Exception:
            pass
    return out


def crop(img, rng, size):
    h, w = img.shape[:2]
    y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    return np.ascontiguousarray(img[y : y + size, x : x + size])


def make_sample(pool, rng, kind: str, pitch=(0.25, 2.0)):
    """kind: real | screen | print. The attack re-photographs a DIFFERENT photo than any real sample."""
    src = crop(pool[int(rng.integers(len(pool)))], rng, SRC)
    if kind == "real":
        out = bona_fide(src, rng)[0]
    elif kind == "screen":
        out = screen_recapture(src, rng, pitch=pitch)[0]
    else:
        out = print_recapture(src, rng)[0]
    p = crop(out, rng, PATCH)
    k = int(rng.integers(4))
    p = np.rot90(p, k)                                  # phones are held either way round
    if rng.random() < 0.5:
        p = p[:, ::-1]
    return np.ascontiguousarray(p)


class Stream(IterableDataset):
    def __init__(self, pool, batch, seed, n_batches):
        self.pool, self.bs, self.seed, self.n = pool, batch, seed, n_batches

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, wid])
        for _ in range(self.n // nw):
            xs, ys = [], []
            for _ in range(self.bs):
                r = rng.random()
                kind = "real" if r < 0.5 else ("screen" if r < 0.85 else "print")
                xs.append(make_sample(self.pool, rng, kind))
                ys.append(0.0 if kind == "real" else 1.0)
            yield (torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().div_(255), torch.tensor(ys))


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, val_pool, n=240, seed=777):
    model.eval()
    rng = np.random.default_rng(seed)

    def run(kind, pitch=(0.25, 2.0)):
        xs = np.stack([make_sample(val_pool, rng, kind, pitch) for _ in range(n)])
        x = torch.from_numpy(xs).permute(0, 3, 1, 2).float().div_(255)
        return torch.cat([model(x[i : i + 64]) for i in range(0, n, 64)]).numpy()

    real = run("real")
    res = {
        "auc_screen_in": auc(run("screen", (0.8, 2.0)), real),
        "auc_screen_lowpitch": auc(run("screen", (0.25, 0.8)), real),
        "auc_screen_ood_pitch": auc(run("screen", (2.0, 3.2)), real),
        "auc_print": auc(run("print"), real),
    }
    thr = float(np.percentile(real, 95))
    res["tpr@5%fp_screen_in"] = float((run("screen", (0.8, 2.0)) > thr).mean())
    res["tpr@5%fp_screen_lowpitch"] = float((run("screen", (0.25, 0.8)) > thr).mean())
    res["tpr@5%fp_screen_ood"] = float((run("screen", (2.0, 3.2)) > thr).mean())
    res["tpr@5%fp_print"] = float((run("print") > thr).mean())
    model.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pool = load_pool(a.images)
    cut = int(len(pool) * 0.75)
    train_pool, val_pool = pool[:cut], pool[cut:]      # disjoint PHOTOS, not just disjoint crops
    print(f"photos: {len(train_pool)} train / {len(val_pool)} val   (attack content comes from the same split)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PadNet(a.width).to(device)
    print(f"device {device}, params {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    loader = DataLoader(Stream(train_pool, a.batch, 1234, a.steps + 5), batch_size=None, num_workers=a.workers, prefetch_factor=4)
    t0, run = time.perf_counter(), None
    for step, (x, y) in enumerate(loader, 1):
        logit = model(x.to(device))
        loss = F.binary_cross_entropy_with_logits(logit, y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.97 * run + 0.03 * loss.item()
        if step % 50 == 0:
            print(f"step {step:5d} loss {run:.4f}  {(time.perf_counter() - t0) / step:.2f}s/step", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            m = evaluate(model, val_pool)
            print("  VAL " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
            torch.save({"model": model.state_dict(), "width": a.width, "step": step, "val": m}, a.out / "pad.pt")
        if step >= a.steps:
            break
    print(f"done {(time.perf_counter() - t0) / 60:.1f} min -> {a.out / 'pad.pt'}")


if __name__ == "__main__":
    main()
