"""Synthetic plate/VIN text tampering on REAL photos.

Edits applied inside a plate box (ground-truth boxes from Open Images):
  glyph  erase a stretch of characters to the local plate colour and draw 1-3 different
         characters in another font, size-matched, softened and noised to the surroundings
  clone  copy-move: paste a same-size patch from elsewhere on the plate (cloned character)
  smear  blur-and-noise a stretch (what a crude 'remove this digit' leaves)
Whether or not the file was edited, it then may be re-saved as JPEG at a random quality, with
the same probability for tampered and untouched images - an editor re-saves, but so does a
messenger, and the detector must not learn "recompressed = tampered".

LIMIT (same lesson as the screen model): this is a simulation of tampering. Real edits made in
Photoshop or by diffusion inpainting will differ, so scores here are not field accuracy.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from synth.common import font_bank, load_font, size_for_cap_height

CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def _feather(h, w, r):
    m = np.ones((h, w), np.float32)
    k = max(1, int(r))
    ramp = np.linspace(0.2, 1, k, dtype=np.float32)
    m[:k] *= ramp[:, None]; m[-k:] *= ramp[::-1][:, None]
    m[:, :k] *= ramp[None]; m[:, -k:] *= ramp[::-1][None]
    return m


def tamper_plate(img: np.ndarray, box: tuple[int, int, int, int], rng: np.random.Generator):
    """Returns (edited image, edit rect (x0,y0,x1,y1), kind)."""
    out = img.astype(np.float32).copy()
    x0, y0, x1, y1 = box
    pw, ph = x1 - x0, y1 - y0
    ew = max(10, int(pw * rng.uniform(0.12, 0.30)))
    eh = max(10, int(ph * rng.uniform(0.40, 0.75)))
    ex = int(rng.integers(x0 + int(0.08 * pw), max(x0 + int(0.08 * pw) + 1, x1 - int(0.08 * pw) - ew)))
    ey = int(y0 + (ph - eh) * rng.uniform(0.2, 0.8))
    ex1, ey1 = min(img.shape[1], ex + ew), min(img.shape[0], ey + eh)
    reg = out[ey:ey1, ex:ex1]
    if reg.size == 0:
        return img, (ex, ey, ex1, ey1), "none"
    lum = reg.mean(-1)
    bg = reg[lum >= np.percentile(lum, 70)].mean(0) if (lum >= np.percentile(lum, 70)).any() else reg.mean((0, 1))
    ink = reg[lum <= np.percentile(lum, 15)].mean(0)
    noise_sd = float(ndi.laplace(lum).std() / 6) + 1.0
    kind = str(rng.choice(["glyph", "clone", "smear"], p=[0.5, 0.25, 0.25]))
    h, w = reg.shape[:2]
    mask = _feather(h, w, max(2, min(h, w) // 6))
    if kind == "glyph":
        canvas = Image.new("RGB", (w, h), tuple(int(v) for v in bg))
        n = int(rng.integers(1, 4))
        fp = str(font_bank().pick(str(rng.choice(["hsrp", "print", "painted"])), rng))
        size = size_for_cap_height(fp, h * rng.uniform(0.6, 0.85))
        f = load_font(fp, size)
        d = ImageDraw.Draw(canvas)
        txt = "".join(rng.choice(list(CHARS), n))
        d.text((w / 2, h / 2), txt, font=f, fill=tuple(int(v) for v in ink), anchor="mm")
        new = np.asarray(canvas, np.float32)
        new = ndi.gaussian_filter(new, (rng.uniform(0.3, 1.1),) * 2 + (0,), truncate=2.5)
        new = new + rng.standard_normal(new.shape).astype(np.float32) * noise_sd
    elif kind == "clone":
        sx = int(np.clip(ex + rng.choice([-1, 1]) * rng.integers(w, 2 * w + 1), 0, img.shape[1] - w))
        new = out[ey : ey + h, sx : sx + w].copy()
    else:
        new = ndi.gaussian_filter(reg, (rng.uniform(1.2, 2.8),) * 2 + (0,), truncate=2.5)
        new = new * 0.5 + bg * 0.5 + rng.standard_normal(new.shape).astype(np.float32) * noise_sd
    out[ey:ey1, ex:ex1] = reg * (1 - mask[..., None]) + new[:h, :w] * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), (ex, ey, ex1, ey1), kind


def maybe_resave(img: np.ndarray, rng: np.random.Generator, p: float = 0.7) -> np.ndarray:
    if rng.random() > p:
        return img
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(rng.integers(55, 96)))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
