"""Training data: on-the-fly synthetic lines, or pre-generated shards (much faster to train on).

Batches are domain-homogeneous (all plate rows, or all VIN lines) because the two use
different canvas widths; each ``DataLoader`` item is already a batch.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from synth.sampler import sample_plate_lines, sample_vin_line
from vinplate.geometry.lines import to_gray
from vinplate.vocab import encode_text


def collate(imgs: list[np.ndarray], texts: list[str]):
    x = torch.from_numpy(np.stack(imgs)).unsqueeze(1).float().div_(255.0)   # [B,1,H,W]
    labels = [torch.tensor(encode_text(t), dtype=torch.long) for t in texts]
    lens = torch.tensor([len(t) for t in labels], dtype=torch.long)
    return x, torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long), lens, texts


class OnlineSynth(IterableDataset):
    """Infinite stream of batches. Deterministic given (seed, worker, epoch)."""

    def __init__(self, batch_size: int, p_vin: float = 0.4, seed: int = 0, batches_per_epoch: int = 1000):
        self.bs, self.p_vin, self.seed, self.n = batch_size, p_vin, seed, batches_per_epoch
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        for _ in range(self.n // nw):
            vin = rng.random() < self.p_vin
            imgs, texts = [], []
            while len(imgs) < self.bs:
                for ln in (sample_vin_line(rng) if vin else sample_plate_lines(rng)):
                    imgs.append(to_gray(ln.img))
                    texts.append(ln.text)
            yield collate(imgs[: self.bs], texts[: self.bs])


class Shards(IterableDataset):
    """Pre-generated shards written by scripts/make_shards.py: <dir>/{plate,vin}_*.npz with
    ``x`` uint8 [N,H,W] and ``text`` unicode [N]. Shuffles within a shard and across shards."""

    def __init__(self, root: str, batch_size: int, p_vin: float = 0.4, seed: int = 0, split: str = "train"):
        self.files = {d: sorted(glob.glob(os.path.join(root, f"{d}_{split}_*.npz"))) for d in ("plate", "vin")}
        if not any(self.files.values()):
            raise FileNotFoundError(f"no shards in {root} for split {split}")
        self.bs, self.p_vin, self.seed = batch_size, p_vin, seed
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        pools = {d: [f for i, f in enumerate(fs) if i % nw == wid] for d, fs in self.files.items()}
        cur: dict[str, tuple | None] = {"plate": None, "vin": None}
        pos = {"plate": 0, "vin": 0}
        order = {d: list(rng.permutation(len(fs))) for d, fs in pools.items()}
        while True:
            dom = "vin" if (rng.random() < self.p_vin and pools["vin"]) or not pools["plate"] else "plate"
            if cur[dom] is None or pos[dom] + self.bs > len(cur[dom][1]):
                if not order[dom]:
                    return
                z = np.load(pools[dom][order[dom].pop()])
                perm = rng.permutation(len(z["text"]))
                cur[dom] = (z["x"][perm], z["text"][perm])
                pos[dom] = 0
            x, t = cur[dom]
            sl = slice(pos[dom], pos[dom] + self.bs)
            pos[dom] += self.bs
            yield collate(list(x[sl]), [str(s) for s in t[sl]])


def gpu_augment(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Cheap extra variety applied on the accelerator: gain/bias, gamma and noise."""
    b = x.shape[0]
    dev = x.device
    gain = 0.8 + 0.4 * torch.rand(b, 1, 1, 1, device=dev)
    bias = (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.2
    gamma = torch.exp((torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.6)
    x = (x.clamp(1e-4, 1) ** gamma) * gain + bias
    x = x + torch.randn_like(x) * 0.02 * torch.rand(b, 1, 1, 1, device=dev)
    return x.clamp(0, 1)
