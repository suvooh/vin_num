"""Presentation-attack evaluation on SIMULATED recaptures of real car photos.

    python -I -X utf8 eval/pad_eval.py --images data/openimages/images --n 150 [--model runs/pad/pad.pt]

Classes: bona fide = a real photo through a camera chain; attack = a simulated photograph of a
screen or print showing a *different* real photo. Reports per-score AUC and, at a threshold
chosen for ~5% bona-fide false positives, the detection rate per attack type.

Read this correctly: it measures how well each cue separates the SIMULATOR's attacks from real
photos. It is not a measurement of real phone-screen attacks (different panels, optics, ISPs,
moire suppression), and the real photos here are ~640 px Flickr renditions, not native phone
captures. It tells you which cues are worth building on and whether the pipeline works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # python -I drops cwd from sys.path
from synth.recapture import bona_fide, print_recapture, screen_recapture
from vinplate.authenticity.forensics import banding_score, chroma_moire, moire_score


def load_pool(folder: Path):
    out = []
    for p in sorted(folder.glob("*.jpg")):
        try:
            out.append(np.asarray(Image.open(p).convert("RGB")))
        except Exception:
            pass
    return out


def crop(img, rng, size=256):
    h, w = img.shape[:2]
    if min(h, w) < size:
        return None
    y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    return np.ascontiguousarray(img[y : y + size, x : x + size])


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scores(img):
    g = img[..., 1].astype(np.float32)
    return {"moire": moire_score(g), "chroma_moire": chroma_moire(img), "banding": banding_score(g)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    pool = load_pool(a.images)
    if len(pool) < 20:
        raise SystemExit(f"need >= 20 photos in {a.images}, have {len(pool)}")
    rng = np.random.default_rng(a.seed)
    half = len(pool) // 2          # content for attacks comes from the OTHER half of the photos
    rows = {"real": [], "screen": [], "print": []}
    detail = []
    tries = 0
    while min(len(v) for v in rows.values()) < a.n and tries < a.n * 8:
        tries += 1
        src_real = crop(pool[int(rng.integers(0, half))], rng)
        src_att = crop(pool[int(rng.integers(half, len(pool)))], rng)
        if src_real is None or src_att is None:
            continue
        if len(rows["real"]) < a.n:
            rows["real"].append(scores(bona_fide(src_real, rng)[0]))
        if len(rows["screen"]) < a.n:
            im, meta = screen_recapture(src_att, rng)
            rows["screen"].append(scores(im))
            detail.append(("screen", meta["pitch"], meta["layout"], rows["screen"][-1]))
        if len(rows["print"]) < a.n:
            im, meta = print_recapture(src_att, rng)
            rows["print"].append(scores(im))
            detail.append(("print", meta["mode"], None, rows["print"][-1]))

    print(f"photos in pool: {len(pool)}  samples per class: { {k: len(v) for k, v in rows.items()} }  (SIMULATED attacks)")
    print(f"{'score':14s} {'AUC screen':>11s} {'AUC print':>10s} | detection rate @ 5% bona-fide FP (screen / print)")
    for k in ("moire", "chroma_moire", "banding"):
        neg = np.array([r[k] for r in rows["real"]])
        thr = float(np.percentile(neg, 95))
        sc = np.array([r[k] for r in rows["screen"]])
        pr = np.array([r[k] for r in rows["print"]])
        print(f"{k:14s} {auc(sc, neg):11.3f} {auc(pr, neg):10.3f} | {(sc > thr).mean():6.1%} / {(pr > thr).mean():6.1%}   (thr {thr:.2f})")
    # screens by display pitch band
    print("\nscreen detection by display pitch (camera px per display px), moire OR chroma_moire @ 5% FP:")
    t1 = np.percentile([r["moire"] for r in rows["real"]], 95)
    t2 = np.percentile([r["chroma_moire"] for r in rows["real"]], 95)
    for lo, hi in ((0.8, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 3.3)):
        sel = [d[3] for d in detail if d[0] == "screen" and lo <= d[1] < hi]
        if sel:
            print(f"  pitch {lo:.1f}-{hi:.1f}: n={len(sel):3d}  detected {np.mean([(s['moire'] > t1) or (s['chroma_moire'] > t2) for s in sel]):6.1%}")
    for mode in ("halftone", "glossy"):
        sel = [d[3] for d in detail if d[0] == "print" and d[1] == mode]
        if sel:
            print(f"  print/{mode:9s}: n={len(sel):3d}  detected {np.mean([(s['moire'] > t1) or (s['chroma_moire'] > t2) for s in sel]):6.1%}")


if __name__ == "__main__":
    main()
