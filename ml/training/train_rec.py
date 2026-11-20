"""Train the line recogniser (CTC).

    # smoke test on CPU, synthetic data generated on the fly
    python -m training.train_rec --data online --steps 1500 --batch 64 --workers 6 --out runs/smoke

    # real run on a GPU box, pre-generated shards
    python -m training.train_rec --data /data/shards --steps 60000 --batch 256 --workers 12 --amp --out runs/v1

Reports greedy exact-match and CER on a FIXED synthetic validation stream. That number only
says the network fits the synthetic distribution; it is NOT an accuracy claim for real photos
(see eval/ for that, and the README's "what the numbers mean").
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.data import OnlineSynth, Shards, gpu_augment
from training.models import LineRecognizer, count_params
from vinplate.decode.ctc import greedy
from vinplate.vocab import CHARS, UNK


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> dict:
    model.eval()
    n = exact = edits = chars = 0
    q_label = q_pred = q_both = 0
    for bi, (x, _y, _l, texts) in enumerate(loader):
        if bi >= max_batches:
            break
        lp = F.log_softmax(model(x.to(device)).float(), -1).cpu().numpy()
        for k, t in enumerate(texts):
            p, _ = greedy(lp[k])
            n += 1
            exact += p == t
            edits += lev(p, t)
            chars += len(t)
            if len(p) == len(t):  # abstention agreement only where positions line up
                for a, b in zip(p, t):
                    q_label += b == UNK
                    q_pred += a == UNK
                    q_both += a == UNK and b == UNK
    model.train()
    return {
        "line_exact": exact / max(n, 1), "cer": edits / max(chars, 1), "n": n,
        "abstain_precision": q_both / max(q_pred, 1), "abstain_recall": q_both / max(q_label, 1),
    }


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="online", help="'online' or a shard directory")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--p-vin", type=float, default=0.4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))

    model = LineRecognizer(a.width).to(device)
    ema = EMA(model)
    print(f"device={device} params={count_params(model) / 1e6:.2f}M width={a.width}", flush=True)

    if a.data == "online":
        train_ds = OnlineSynth(a.batch, a.p_vin, a.seed, batches_per_epoch=a.steps + 10)
    else:
        train_ds = Shards(a.data, a.batch, a.p_vin, a.seed)
    train = DataLoader(train_ds, batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None,
                       persistent_workers=False)
    val = DataLoader(OnlineSynth(a.batch, a.p_vin, seed=9_999_991, batches_per_epoch=a.val_batches),
                     batch_size=None, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.08,
                                                anneal_strategy="cos", div_factor=20, final_div_factor=200)
    scaler = torch.amp.GradScaler(enabled=a.amp and device == "cuda")

    model.train()
    t0 = time.perf_counter()
    step, run = 0, 0.0
    for x, y, lens, _texts in train:
        x = gpu_augment(x.to(device, non_blocking=True))
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=a.amp and device == "cuda"):
            logits = model(x)
        lp = F.log_softmax(logits.float(), -1).permute(1, 0, 2)               # [T,B,V]
        T = lp.shape[0]
        loss = F.ctc_loss(lp, y.to(device), torch.full((x.shape[0],), T, dtype=torch.long), lens,
                          blank=0, zero_infinity=True, reduction="mean")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)
        step += 1
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 25 == 0:
            print(f"step {step:6d} loss {run:7.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.perf_counter() - t0) / step:.2f}s/step", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            m = evaluate(ema.m, val, device, a.val_batches)
            print(f"  VAL(ema) step {step}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)
            torch.save({"model": ema.m.state_dict(), "width": a.width, "step": step, "val": m}, out / "rec_ema.pt")
        if step >= a.steps:
            break
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {out / 'rec_ema.pt'}")


if __name__ == "__main__":
    main()
