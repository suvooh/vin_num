"""Pre-generate synthetic training lines into compressed .npz shards (grayscale uint8).

Generating on the fly costs ~50-70 ms per line per core; a GPU would starve. Shards are made
once, in parallel, and read at disk speed. Train and validation shards use disjoint seed
streams so a validation line can never also be a training line.

    python scripts/make_shards.py --out /data/shards --plate 1200000 --vin 800000 --workers 16
    # 2M lines is roughly 10-14 GB compressed.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.sampler import sample_plate_lines, sample_vin_line  # noqa: E402
from vinplate.geometry.lines import to_gray  # noqa: E402

SPLIT_ID = {"train": 0, "val": 1}


def work(job):
    out, domain, split, idx, n, seed = job
    rng = np.random.default_rng([seed, SPLIT_ID[split], 0 if domain == "plate" else 1, idx])
    fn = sample_plate_lines if domain == "plate" else sample_vin_line
    xs, ts = [], []
    while len(xs) < n:
        for ln in fn(rng):
            xs.append(to_gray(ln.img))
            ts.append(ln.text)
    path = Path(out) / f"{domain}_{split}_{idx:05d}.npz"
    np.savez_compressed(path, x=np.stack(xs[:n]), text=np.array(ts[:n]))
    return path.name, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plate", type=int, default=1_200_000, help="train plate-row lines")
    ap.add_argument("--vin", type=int, default=800_000, help="train VIN lines")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--shard", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    jobs = []
    for domain, total in (("plate", a.plate), ("vin", a.vin)):
        for split, count in (("train", total), ("val", int(total * a.val_frac))):
            n_shards = max(1, -(-count // a.shard)) if count else 0
            jobs += [(a.out, domain, split, i, min(a.shard, count - i * a.shard), a.seed) for i in range(n_shards)]
    t0, done = time.perf_counter(), 0
    with Pool(a.workers) as pool:
        for name, n in pool.imap_unordered(work, jobs):
            done += n
            print(f"{name}  ({done:,} lines, {done / (time.perf_counter() - t0):.0f} lines/s)", flush=True)
    print(f"done: {done:,} lines in {(time.perf_counter() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
