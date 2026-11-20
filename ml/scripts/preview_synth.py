"""Render a contact sheet of synthetic lines so the generator can be eyeballed.

    python scripts/preview_synth.py --n 40 --domain plate --out sheet.png [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.sampler import sample_plate_lines, sample_vin_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--domain", choices=["plate", "vin", "mixed"], default="plate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    lines = []
    while len(lines) < a.n:
        dom = a.domain if a.domain != "mixed" else ("vin" if rng.random() < 0.5 else "plate")
        lines += sample_vin_line(rng) if dom == "vin" else sample_plate_lines(rng)
    dt = (time.perf_counter() - t0) / len(lines)
    lines = lines[: a.n]

    h, w = lines[0].img.shape[:2]
    cell_w, cell_h = max(l.img.shape[1] for l in lines) + 8, max(l.img.shape[0] for l in lines) + 22
    rows = (len(lines) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (cell_w * a.cols, cell_h * rows), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, l in enumerate(lines):
        x, y = (i % a.cols) * cell_w, (i // a.cols) * cell_h
        sheet.paste(Image.fromarray(l.img), (x + 4, y + 18))
        d.text((x + 4, y + 2), f"{l.text}  [{l.profile}] {','.join(l.tags)}"[:110], fill=(0, 0, 0))
    sheet.save(a.out)
    print(f"{len(lines)} lines, {dt * 1000:.1f} ms/line -> {a.out}")


if __name__ == "__main__":
    main()
