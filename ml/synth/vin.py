"""Render VIN / chassis-number lines in the four ways they appear on Indian vehicles:

  label     printed sticker (door jamb / engine bay): monospace-ish print, optional "VIN:" /
            "CHASSIS NO" prefix, barcode and neighbouring text lines as clutter
  engraved  riveted metal VIN plate: brushed metal, dark engraved characters, rivets
  dot       dot-matrix stamped chassis number (steel stamping machines): dots on paint
  stencil   punched-die / stencil stamped characters on painted or rusty chassis metal

There is no public VIN/chassis photo dataset for India; these styles are written from how
the marks look, not fitted to real samples. They exist to teach glyph shapes and degradation
robustness; real captures are needed to close the gap (see README, "Data").
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from synth.common import font_bank, jitter, load_font, size_for_cap_height
from synth.plates import Rendered
from synth.text import sample_vin

_PREFIXES = ["VIN", "VIN NO", "CHASSIS NO", "CHASIS NO", "CHASSIS"]
_STYLE_P = {"label": 0.34, "engraved": 0.22, "dot": 0.20, "stencil": 0.24}


def _metal(rng, W, H):
    base = rng.uniform(150, 205)
    g = np.linspace(rng.uniform(-18, 18), rng.uniform(-18, 18), H)[:, None] * np.ones((1, W))
    brushed = ndi.gaussian_filter(rng.standard_normal((H, W), dtype=np.float32) * 14, (0.4, rng.uniform(6, 18)), truncate=2.5)
    c = base + g + brushed
    tint = rng.uniform(0.95, 1.05, 3)
    return np.clip(c[..., None] * tint, 0, 255).astype(np.float32)


def _paint(rng, W, H):
    kind = rng.choice(["grey", "black", "rust", "silver"])
    col = {"grey": (110, 112, 115), "black": (38, 38, 40), "rust": (120, 70, 45), "silver": (170, 172, 175)}[kind]
    base = np.ones((H, W, 3), np.float32) * np.asarray(jitter(rng, col, 10), np.float32)
    blot = ndi.gaussian_filter(rng.standard_normal((H, W), dtype=np.float32), rng.uniform(2, 8), truncate=2.5)
    blot = blot / (np.abs(blot).max() + 1e-6) * rng.uniform(10, 35)
    fine = rng.standard_normal((H, W), dtype=np.float32) * np.float32(rng.uniform(4, 12))
    out = base + (blot + fine)[..., None]
    if kind != "rust" and rng.random() < 0.4:  # rust bloom
        rm = np.clip(ndi.gaussian_filter(rng.random((H, W), dtype=np.float32), 4, truncate=2.5) - 0.55, 0, 1) * 6
        out = out * (1 - rm[..., None]) + np.array([130, 75, 40]) * rm[..., None]
    return np.clip(out, 0, 255), kind


def _line_layout(rng, text, font, cap, stamped):
    """x positions per char with monospace-ish advance + (for stamped) wobble."""
    adv = np.array([font.getlength(c) for c in text], np.float32)
    mono = adv.max() * rng.uniform(1.0, 1.3) if not stamped else adv.max() * rng.uniform(1.05, 1.5)
    xs = np.cumsum(np.r_[0, np.full(len(text) - 1, mono)]) if len(text) > 1 else np.zeros(1)
    if stamped:
        xs = xs + np.cumsum(rng.normal(0, 0.06 * mono, len(text)))
    return xs, mono


def render_vin(rng: np.random.Generator, spec: dict | None = None) -> Rendered:
    spec = spec or sample_vin(rng)
    text = spec["text"]
    names = list(_STYLE_P)
    style = str(rng.choice(names, p=np.array([_STYLE_P[n] for n in names])))
    prefix = str(rng.choice(_PREFIXES)) if (style == "label" and rng.random() < 0.18) else ""

    n = len(text) + (len(prefix) + 2 if prefix else 0)
    cap = rng.uniform(24, 32)
    role = {"label": "print", "engraved": "print", "dot": "print", "stencil": "stamp"}[style]
    fpath = str(font_bank().pick(role, rng))
    weight = int(rng.choice([400, 500, 600, 700])) if "[" in fpath else 0
    font = load_font(fpath, size_for_cap_height(fpath, cap, weight), weight)
    stamped = style in ("dot", "stencil")
    full = (prefix + ": " if prefix else "") + text
    xs, mono = _line_layout(rng, full, font, cap, stamped)
    pad_x = rng.uniform(0.6, 1.6) * mono
    W = int(xs[-1] + font.getlength(full[-1]) + 2 * pad_x)
    H = int(cap * rng.uniform(1.9, 2.8))

    if style == "label":
        bg = jitter(rng, (242, 242, 238), 8)
        base = np.ones((H, W, 3), np.float32) * np.asarray(bg, np.float32)
        fg = jitter(rng, (22, 22, 24), 10)
    elif style == "engraved":
        base = _metal(rng, W, H)
        fg = jitter(rng, (35, 35, 38), 12)
    else:
        base, ptype = _paint(rng, W, H)
        lum = base.mean()
        fg = (int(max(0, lum - rng.uniform(35, 70))),) * 3 if rng.random() < 0.7 else (int(min(255, lum + rng.uniform(30, 60))),) * 3

    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(canvas)

    # clutter: neighbouring lines of text / barcode (labels), rivets (plates)
    if style in ("label", "engraved") and rng.random() < 0.5:
        f2 = load_font(fpath, max(6, int(cap * 0.5)), weight)
        junk = "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789 -."), int(rng.integers(8, 28))))
        d.text((rng.uniform(0, 0.4) * W, H * rng.choice([0.04, 0.8])), junk, font=f2, fill=fg)
    if style == "label" and rng.random() < 0.25:
        x = rng.uniform(0.05, 0.4) * W
        by = H * rng.choice([0.84, 0.0])
        while x < W * 0.9:
            bw = int(rng.integers(1, 4))
            d.rectangle([x, by, x + bw, by + H * 0.15], fill=fg)
            x += bw + int(rng.integers(1, 4))
    if style == "engraved":
        for cx in (0.012 * W, 0.988 * W):
            r = 0.09 * H
            d.ellipse([cx - r, H / 2 - r, cx + r, H / 2 + r], fill=jitter(rng, (120, 120, 125), 15))

    base_y = H * rng.uniform(0.58, 0.72)
    boxes = []
    amp = rng.uniform(0, 0.08) * cap if stamped else 0
    ph = rng.uniform(0, 6.28)
    for i, ch in enumerate(full):
        x = pad_x + xs[i]
        dy = amp * np.sin(i * rng.uniform(0.2, 0.5) + ph) + (rng.normal(0, 0.03 * cap) if stamped else 0)
        l, t, r_, b = font.getbbox(ch, anchor="ls")
        glyph = Image.new("L", (r_ - l + 12, b - t + 12), 0)
        ImageDraw.Draw(glyph).text((6 - l, 6 - t), ch, font=font, fill=255, anchor="ls")
        if stamped:
            glyph = glyph.rotate(rng.normal(0, 3.0), resample=Image.BICUBIC, expand=True)
        g = np.asarray(glyph, np.float32) / 255
        if style == "dot":
            pitch = max(2.0, cap / rng.uniform(8, 11))
            hh, ww = g.shape
            dots = np.zeros_like(g)
            rr = pitch * rng.uniform(0.32, 0.46)
            for yy in np.arange(pitch / 2, hh, pitch):
                for xx in np.arange(pitch / 2, ww, pitch):
                    if g[int(yy), int(xx)] > 0.45:
                        y0, y1, x0, x1 = int(yy - rr), int(yy + rr) + 1, int(xx - rr), int(xx + rr) + 1
                        dots[max(0, y0) : y1, max(0, x0) : x1] = 1
            g = dots
        if style in ("dot", "stencil") and rng.random() < 0.5:
            g = ndi.grey_erosion(g, size=(2, 2)) if rng.random() < 0.5 else ndi.grey_dilation(g, size=(2, 2))
        gimg = Image.fromarray((g * 255).astype(np.uint8))
        ox, oy = int(x + l - 6), int(base_y + dy + t - 6)
        # impressed look: light edge offset one pixel under the dark glyph
        if style in ("engraved", "dot", "stencil"):
            hl = tuple(int(min(255, c + 55)) for c in (np.asarray(base[min(H - 1, max(0, oy)), min(W - 1, max(0, ox))]).astype(int)))
            canvas.paste(Image.new("RGB", gimg.size, hl), (ox + 1, oy + 1), gimg.point(lambda v: v // 2))
        canvas.paste(Image.new("RGB", gimg.size, tuple(fg)), (ox, oy), gimg)
        mask.paste(255, (ox, oy), gimg)
        boxes.append((int(x + l), int(base_y + dy + t), int(x + r_), int(base_y + dy + b)))

    img = np.asarray(canvas, np.float32)
    if style == "label":
        img = ndi.gaussian_filter(img, (0.5, 0.5, 0), truncate=2.0)        # ink spread / thermal print softness
    # label text of the characters that are the VIN value only (prefix chars excluded)
    skip = (len(prefix) + 2) if prefix else 0
    label_boxes = boxes[skip:]
    value_text = text
    if prefix:  # the transcript keeps the prefix word (grammar can strip it); ':' is not in the vocab
        label_boxes = boxes[: len(prefix)] + boxes[len(prefix) + 2 :]
        value_text = prefix.replace(" ", "") + text
    return Rendered(
        img=np.clip(img, 0, 255), rows=[value_text], boxes=[label_boxes], row_y=[(0, H)],
        textmask=(np.asarray(mask) > 40).astype(np.float32),
        meta=dict(kind="vin", style=style, legacy=spec["legacy"], prefix=prefix, text=text),
    )
