"""Plates of every colour scheme + lighting degradations, for testing ``vinplate.category``.

``render_scheme`` draws a plate with the repo's own renderer (``synth.plates.render_plate``: HSRP / painted,
1-row / 2-row, IND strip, border, embossing) in the colours of one scheme and returns the TEXT-LINE crop the
detector would hand to the classifier, the plate text and the scheme's expected category.  The scheme colours
are the ones in the category rule table (the user's infographics, recalled / not verified); each scheme's
colours are jittered more than the renderer does by default because real sheeting varies a lot.

``degrade`` applies a named lighting variant to a crop (uint8 in, uint8 out).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import numpy as np
from scipy import ndimage as ndi

from synth import plates as _plates
from synth.common import jitter
from synth.text import sample_plate

# scheme -> (expected category key, background rgb, ink rgb, layout the text must have or None for "standard")
SCHEMES: dict[str, tuple[str, tuple, tuple, str | None]] = {
    "private": ("private", (248, 248, 246), (18, 18, 18), None),
    "commercial": ("commercial", (255, 206, 22), (18, 18, 18), None),
    "rental": ("rental", (22, 22, 22), (255, 206, 22), None),
    "ev_private": ("ev_private", (22, 150, 72), (246, 246, 246), None),
    "ev_commercial": ("ev_commercial", (22, 150, 72), (255, 214, 40), None),
    "temporary": ("temporary", (255, 206, 22), (196, 28, 36), None),
    "trade": ("trade", (196, 28, 36), (246, 246, 246), "dealer"),
    "trade_plain": ("trade", (196, 28, 36), (246, 246, 246), None),    # red plate, ordinary text
    "diplomatic": ("diplomatic", (0, 72, 172), (246, 246, 246), "diplomatic"),
    "military": ("military", (26, 26, 30), (246, 246, 246), "military"),
    "bharat": ("private", (248, 248, 246), (18, 18, 18), "bharat"),
    # The REAL Indian EV plate is a teal (measured 174-175 deg hue on a photo of a green-plate car), not a leaf green: the first
    # version of the rules called it blue / diplomatic. Kept as separate schemes so the leaf-green ones above still run.
    "ev_private_teal": ("ev_private", (26, 172, 158), (246, 246, 246), None),
    "ev_commercial_teal": ("ev_commercial", (26, 172, 158), (255, 214, 40), None),
}

LIGHTING = ["clean", "dim", "bright", "warm", "cool", "shade", "glare", "blur_noise"]


@dataclass
class SchemePlate:
    crop: np.ndarray        # HxWx3 uint8, one text line
    text: str
    scheme: str
    expected: str           # expected category key
    layout: str             # the text layout the grammar gives (what the classifier receives)


def _text_for(rng, scheme_layout: str | None):
    """-> (spec for render_plate)."""
    if scheme_layout == "dealer":
        rto, num = str(int(rng.integers(1, 100))).zfill(2), str(int(rng.integers(100, 10000))).zfill(4)
        groups = ["DLR", rto, num]
        return dict(layout="dealer", groups=groups, text="".join(groups), kind="dealer")
    if scheme_layout == "military":
        digits = "".join(str(int(d)) for d in rng.integers(0, 10, 6))
        groups = [str(int(rng.integers(10, 99))), str(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"))), digits,
                  str(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ")))]
        return dict(layout="military", groups=groups, text="".join(groups), kind="military")
    want = scheme_layout or "standard"
    for _ in range(2000):
        spec = sample_plate(rng)
        if spec["layout"] == want:
            return spec
    raise RuntimeError(f"could not sample a {want} plate")


def render_scheme(rng: np.random.Generator, scheme: str, *, colour_jitter: int = 12) -> SchemePlate:
    expected, bg0, ink0, layout = SCHEMES[scheme]
    spec = _text_for(rng, layout)
    bg = jitter(rng, bg0, colour_jitter)
    ink = jitter(rng, ink0, colour_jitter)
    # plates.render_plate picks its own style; force the scheme's colours
    with mock.patch.object(_plates, "_pick_style", lambda r, lay: (scheme, bg, ink)):
        r = _plates.render_plate(rng, spec)
    # the text line the detector would return: the row with the most characters, a small margin around its glyphs
    ri = int(np.argmax([len(t) for t in r.rows]))
    bx = r.boxes[ri]
    x0, x1 = min(b[0] for b in bx), max(b[2] for b in bx)
    y0, y1 = min(b[1] for b in bx), max(b[3] for b in bx)
    h = y1 - y0
    H, W = r.img.shape[:2]
    px, py = int(0.2 * h), int(0.2 * h)
    cx0, cx1 = max(0, x0 - px), min(W, x1 + px)
    cy0, cy1 = max(0, y0 - py), min(H, y1 + py)
    crop = np.clip(np.rint(r.img[cy0:cy1, cx0:cx1]), 0, 255).astype(np.uint8)
    return SchemePlate(crop=crop, text=spec["text"], scheme=scheme, expected=expected, layout=spec["layout"])


def degrade(crop: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """A named lighting variant of a crop (uint8 -> uint8)."""
    x = crop.astype(np.float64)
    h, w = x.shape[:2]
    if kind == "clean":
        return crop
    if kind == "dim":
        x = 255.0 * (x / 255.0) ** 1.4 * 0.62
    elif kind == "bright":
        x = x * 1.45 + 12.0
    elif kind == "warm":
        x = x * np.array([1.12, 1.0, 0.78])
    elif kind == "cool":
        x = x * np.array([0.82, 0.98, 1.14])
    elif kind == "shade":
        ramp = np.linspace(0.5, 1.0, w)[None, :, None]
        x = x * (ramp if rng.random() < 0.5 else ramp[:, ::-1])
    elif kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.3, 0.7) * h
        g = np.exp(-(((xx - cx) / (0.22 * w)) ** 2 + ((yy - cy) / (0.9 * h)) ** 2))
        x = x + 110.0 * g[..., None]
    elif kind == "blur_noise":
        x = np.stack([ndi.gaussian_filter(x[..., c], 1.2) for c in range(3)], axis=-1)
        x = x + rng.normal(0, 7.0, x.shape)
    else:
        raise ValueError(kind)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)
