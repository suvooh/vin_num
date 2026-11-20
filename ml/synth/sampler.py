"""One call -> one training line: image, transcript (with '?' for hidden characters), tags.

Pipeline per sample
    render canonical plate / VIN  ->  add surround  ->  DAMAGE (glare/dirt/occlusion, alpha)
    -> per-character visibility -> '?' labels  ->  OPTICS (perspective/blur/noise/JPEG/...)
    -> crop the line(s) with detector-like jitter  ->  vinplate.geometry.fit_line

The label rule is the point of the design: a character whose glyph pixels are mostly covered
by damage is labelled '?', so the recogniser learns to say "unreadable" rather than invent a
digit. Characters that are merely blurry / noisy keep their label - that is what the network
is supposed to learn to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from synth.degrade import apply_damage, apply_optics, sample_severity
from synth.plates import Rendered, render_plate
from synth.vin import render_vin
from vinplate.geometry import PLATE_LINE_SHAPE, VIN_LINE_SHAPE, fit_line

HIDDEN_ABOVE = 0.65   # fraction of a glyph covered by damage beyond which it is labelled '?'


@dataclass
class Line:
    img: np.ndarray                  # uint8 HxWx3, exactly the recogniser input shape
    text: str                        # transcript, '?' where the character is hidden
    domain: str                      # "plate" | "vin"
    profile: str                     # clean | mild | moderate | severe
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# Vehicle body / bumper colours (white, silver, greys, black, red, blue, green, beige, brown)
_BODY = np.array([(235, 235, 235), (190, 192, 196), (120, 122, 126), (60, 62, 66), (25, 25, 28),
                  (170, 30, 35), (35, 70, 150), (40, 95, 60), (205, 190, 160), (110, 85, 60)], np.float32)


def _surround(r: Rendered, rng, tags) -> Rendered:
    """Pad with vehicle body the way a loose detector box / rectifier margin does."""
    h, w = r.img.shape[:2]
    m = rng.uniform(0, 0.09, 4) * np.array([w, h, w, h])  # l, t, r, b
    l, t, rr, b = (int(v) for v in m)
    col = _BODY[int(rng.integers(len(_BODY)))] + rng.normal(0, 8, 3).astype(np.float32)
    grad = np.linspace(-18, 18, h + t + b, dtype=np.float32)[:, None, None] * np.float32(rng.uniform(0.3, 1.2))
    out = np.empty((h + t + b, w + l + rr, 3), np.float32)
    out[:] = col + grad + rng.standard_normal(out.shape, dtype=np.float32) * 3
    out[t : t + h, l : l + w] = r.img
    tm = np.zeros(out.shape[:2], np.float32)
    tm[t : t + h, l : l + w] = r.textmask
    boxes = [[(x0 + l, y0 + t, x1 + l, y1 + t) for x0, y0, x1, y1 in row] for row in r.boxes]
    return Rendered(np.clip(out, 0, 255), r.rows, boxes, [(a + t, b_ + t) for a, b_ in r.row_y], tm, r.meta)


def _visibility_labels(r: Rendered, alpha: np.ndarray) -> list[str]:
    out = []
    H, W = alpha.shape
    for text, row in zip(r.rows, r.boxes):
        # for VIN rows with a prefix, boxes and text are aligned 1:1 (see render_vin)
        chars = []
        for ch, (x0, y0, x1, y1) in zip(text, row):
            x0, x1 = max(0, x0), min(W, x1)
            y0, y1 = max(0, y0), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                chars.append(ch)
                continue
            tm = r.textmask[y0:y1, x0:x1]
            tot = tm.sum()
            if tot < 5:
                chars.append(ch)
                continue
            hidden = float((alpha[y0:y1, x0:x1] * tm).sum() / tot)
            chars.append("?" if hidden > HIDDEN_ABOVE else ch)
        out.append("".join(chars))
    return out


def _crop_rows(img: np.ndarray, r: Rendered, rng) -> list[np.ndarray]:
    H, W = img.shape[:2]
    crops = []
    for (y0, y1) in r.row_y:
        rh = y1 - y0
        if len(r.rows) == 1 and len(r.row_y) == 1:
            crops.append(img)  # single-row: the whole (surrounded, degraded) canvas
            continue
        a = int(max(0, y0 - rng.uniform(0.0, 0.3) * rh))
        b = int(min(H, y1 + rng.uniform(0.0, 0.3) * rh))
        crops.append(img[a:b])
    return crops


def sample_plate_lines(rng: np.random.Generator) -> list[Line]:
    """A plate yields 1 line (1-row) or 2 lines (2-row); the pipeline concatenates posteriors."""
    r = _surround(render_plate(rng), rng, [])
    return _finish(r, rng, "plate", PLATE_LINE_SHAPE)


def sample_vin_line(rng: np.random.Generator) -> list[Line]:
    r = _surround(render_vin(rng), rng, [])
    return _finish(r, rng, "vin", VIN_LINE_SHAPE)


def _degrade(r: Rendered, rng):
    """Damage -> '?' labels -> optics. Returns (uint8 image, per-row texts, profile, severity, tags)."""
    profile, s = sample_severity(rng)
    # Damage that erases (almost) the whole line teaches nothing but "say ?" - keep a small
    # share of them (the network must learn to reject) and resample the rest milder.
    allow_wipeout = rng.random() < 0.03
    for attempt in range(4):
        tags: list[str] = []
        img, alpha = apply_damage(r.img, rng, s * (0.8 ** attempt), tags)
        texts = _visibility_labels(r, alpha)
        n = sum(len(t) for t in texts)
        if allow_wipeout or sum(t.count("?") for t in texts) <= 0.5 * n:
            break
    img = apply_optics(img, rng, s, tags)
    return np.clip(img, 0, 255).astype(np.uint8), texts, profile, s, tags


def _finish(r: Rendered, rng, domain: str, shape) -> list[Line]:
    img8, texts, profile, s, tags = _degrade(r, rng)
    interp = str(rng.choice(["bilinear", "bicubic", "area"]))
    lines = []
    for crop, text in zip(_crop_rows(img8, r, rng), texts):
        lines.append(Line(fit_line(crop, shape, interp), text, domain, profile, list(tags),
                          dict(r.meta, severity=round(s, 2))))
    return lines


def sample_plate_whole(rng: np.random.Generator) -> Line:
    """The WHOLE plate (1 or 2 rows) as one degraded image with the full transcript - the unit
    the deployed pipeline sees after rectification. For plate-level evaluation, not training."""
    r = _surround(render_plate(rng), rng, [])
    img8, texts, profile, s, tags = _degrade(r, rng)
    return Line(img8, "".join(texts), "plate", profile, tags, dict(r.meta, severity=round(s, 2), rows=len(texts)))


def sample_vin_whole(rng: np.random.Generator) -> Line:
    r = _surround(render_vin(rng), rng, [])
    img8, texts, profile, s, tags = _degrade(r, rng)
    return Line(img8, texts[0], "vin", profile, tags, dict(r.meta, severity=round(s, 2)))


def sample_lines(rng: np.random.Generator, p_vin: float = 0.4) -> list[Line]:
    return sample_vin_line(rng) if rng.random() < p_vin else sample_plate_lines(rng)
