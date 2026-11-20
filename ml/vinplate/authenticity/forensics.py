"""Classical pixel forensics: cheap corroboration, NOT detection you can rely on alone.

Honest limits (from the research behind this module):
  * ELA / noise residual / JPEG-grid checks are strong only on first-generation camera JPEGs
    and are wiped out by one re-save, a resize or a messaging-app recompression. On such
    images they return low-information scores - which is why every score here ships with the
    ``reliable`` flag and a reason, and why none of them can clear an image, only raise doubt.
  * Moire / periodicity catches photos *of screens* only when the screen's pixel grid aliases
    against the sensor; many recaptures show none.
  * Thresholds are PROVISIONAL, tuned on synthetic cases in tests/. They must be calibrated on
    real recapture/tamper data (which has to be collected: no public plate/VIN set exists).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

THRESH = dict(ela_inconsistency=3.5, noise_inconsistency=3.0, moire=12.0, grid_disagree=0.35,
              chroma_moire=10.5, banding=15.0)   # just above the p99 of ~100 real Flickr crops: a WEAK calibration


@dataclass
class ForensicsReport:
    scores: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    calibrated: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _blocks(a: np.ndarray, b: int) -> np.ndarray:
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def ela_inconsistency(rgb: np.ndarray, quality: int = 90, block: int = 16) -> float:
    """Recompress and compare. A pasted/edited region has a different error level from its
    surroundings; returns (p95 / median) of per-block error. ~1-2 for untouched images."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    re = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.float32)
    err = np.abs(rgb.astype(np.float32) - re).mean(axis=2)
    bm = _blocks(err, block).ravel()
    return float(np.percentile(bm, 95) / (np.median(bm) + 0.05))


def noise_inconsistency(gray: np.ndarray, block: int = 24) -> float:
    """Sensor noise is locally consistent; a spliced region from another source is not.
    Residual = image - 3x3 median; per-block MAD; returns p95/median over textured blocks."""
    g = gray.astype(np.float32)
    res = np.abs(g - ndi.median_filter(g, 3))
    bm = _blocks(res, block).ravel()
    bm = bm[bm > 0.05]
    if len(bm) < 8:
        return 1.0
    return float(np.percentile(bm, 95) / (np.median(bm) + 1e-3))


def moire_score(gray: np.ndarray) -> float:
    """Strongest isolated spectral peak relative to its neighbourhood (high-passed image).
    Screen pixel grids alias into narrow periodic peaks; natural scenes have smooth spectra."""
    g = gray.astype(np.float32)
    g = g - ndi.gaussian_filter(g, 3)
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g * win)))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    spec[r < 0.04 * min(h, w)] = 0
    # local background = neighbourhood mean (a 15x15 median here cost ~0.7 s per image)
    bg = ndi.uniform_filter(spec, size=15, mode="nearest")
    ratio = spec / (bg + 1e-6)
    ratio[r < 0.04 * min(h, w)] = 0
    return float(np.sort(ratio.ravel())[-3:].mean())


GRID_VISIBLE = 1.25   # measured: never-JPEG / resized images top out ~1.20; JPEG q<=75 >= 1.35


def _grid_phase(a: np.ndarray) -> tuple[int, int, float]:
    """(phase_x, phase_y, visibility). An aligned 8x8 grid puts its strongest discontinuity
    between columns 7|8 (diff index 7), so phase = argmax + 1 (mod 8) and 0 means aligned.
    Visibility = peak / mean over the 8 phases (min of the two axes)."""
    dx = np.abs(np.diff(a, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(a, axis=0)).mean(axis=1)
    px = np.array([dx[k::8].mean() for k in range(8)])
    py = np.array([dy[k::8].mean() for k in range(8)])
    return (int(px.argmax() + 1) % 8, int(py.argmax() + 1) % 8,
            float(min(px.max() / (px.mean() + 1e-6), py.max() / (py.mean() + 1e-6))))


def chroma_moire(rgb: np.ndarray) -> float:
    """Moire in the *chroma* planes (R-G, B-G). A screen's RGB sub-pixel stripes alias into
    colour fringes that luminance-only checks miss. Same peak-vs-neighbourhood statistic."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return max(moire_score(r - g), moire_score(b - g))


def banding_score(gray: np.ndarray) -> float:
    """Rolling-shutter / PWM banding: horizontal brightness bands from a flickering light source
    or a refreshing display. Row-mean profile, detrended, strongest spectral peak (periods of
    8-300 rows) relative to the median. WEAK: natural horizontal structure (shelves, fences,
    sky/ground) also produces peaks, and many cameras avoid banding by exposure choice."""
    m = gray.astype(np.float32).mean(axis=1)
    m = m - ndi.uniform_filter1d(m, 41)
    n = len(m)
    if n < 128:
        return 0.0
    spec = np.abs(np.fft.rfft(m * np.hanning(n)))
    freqs = np.fft.rfftfreq(n)
    band = (freqs >= 1 / 300) & (freqs <= 1 / 8)
    s = spec[band]
    return float(s.max() / (np.median(s) + 1e-6))


def jpeg_grid(gray: np.ndarray, tile: int = 96) -> dict:
    """Blockiness phase of the 8x8 JPEG grid, globally and per tile. A crop/shift after
    compression moves the global phase off (0,0); a splice from a differently-aligned source
    leaves tiles that disagree with the rest. Only informative on moderately compressed JPEGs
    (quality <~ 90): at q=95 the grid is as faint as on a never-compressed image, and any
    resize destroys it - ``strength`` < GRID_VISIBLE says so and callers must not infer."""
    g = gray.astype(np.float32)
    gx, gy, strength = _grid_phase(g)
    seen = disagree = 0
    for y in range(0, g.shape[0] - tile + 1, tile):
        for x in range(0, g.shape[1] - tile + 1, tile):
            tx, ty, tv = _grid_phase(g[y : y + tile, x : x + tile])
            if tv >= GRID_VISIBLE:          # a tile that cannot see the grid says nothing
                seen += 1
                disagree += (tx, ty) != (gx, gy)
    return {"phase": (gx, gy), "strength": strength,
            "disagree": (disagree / seen) if seen >= 4 else 0.0, "tiles_seen": seen}


def analyze_pixels(rgb: np.ndarray, was_jpeg: bool = True) -> ForensicsReport:
    """``rgb``: HxWx3 uint8 of the *full* uploaded image (not a crop)."""
    rep = ForensicsReport()
    gray = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)
    rep.scores["ela_inconsistency"] = round(ela_inconsistency(rgb), 2)
    rep.scores["noise_inconsistency"] = round(noise_inconsistency(gray), 2)
    rep.scores["moire"] = round(moire_score(gray), 2)
    if was_jpeg:
        gr = jpeg_grid(gray)
        rep.scores["jpeg_grid_strength"] = round(gr["strength"], 3)
        rep.scores["jpeg_grid_disagree"] = round(gr["disagree"], 3)
        rep.scores["jpeg_grid_phase_x"], rep.scores["jpeg_grid_phase_y"] = gr["phase"]
        grid_visible = gr["strength"] >= GRID_VISIBLE
        if not grid_visible:
            rep.notes.append("JPEG grid not visible (high-quality original, resized or re-encoded): grid-based checks uninformative")
        elif gr["phase"] != (0, 0):
            rep.flags.append("jpeg_grid_shifted")
        if grid_visible and gr["disagree"] > THRESH["grid_disagree"]:
            rep.flags.append("jpeg_grid_inconsistent")
    if rep.scores["ela_inconsistency"] > THRESH["ela_inconsistency"]:
        rep.flags.append("ela_inconsistent")
    if rep.scores["noise_inconsistency"] > THRESH["noise_inconsistency"]:
        rep.flags.append("noise_inconsistent")
    if rep.scores["moire"] > THRESH["moire"]:
        rep.flags.append("periodic_pattern")
    rep.scores["chroma_moire"] = round(chroma_moire(rgb), 2)
    rep.scores["banding"] = round(banding_score(gray), 2)
    if rep.scores["chroma_moire"] > THRESH["chroma_moire"]:
        rep.flags.append("chroma_periodic_pattern")
    if rep.scores["banding"] > THRESH["banding"]:
        rep.flags.append("horizontal_banding")
    return rep
