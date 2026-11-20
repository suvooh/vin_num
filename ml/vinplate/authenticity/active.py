"""Active / multi-frame physical checks for presentation attacks (photo of a screen or print).

These need capture support: the app must lock exposure and ISO and take a short burst, with one
frame flash-OFF and one flash-ON a few tens of ms apart, plus (for parallax) IMU data. They are
the strongest *physical* separators between a real vehicle and a re-photographed picture of one,
and they do not depend on the attacker's screen resolution or moire suppression.

1. ``retroreflection``  HSRP sheeting is retroreflective: flash light returns straight to the
   lens, so under flash the plate brightens far more than the (diffusely reflecting) paint and
   plastic around it. A screen EMITS its own light (flash barely changes it); a print reflects
   diffusely like everything else (plate and surround brighten alike). Ratio of the plate's
   flash gain to its surround's flash gain separates the three.
     * applies ONLY to retroreflective plates (HSRP). An old painted plate is "not retroreflective"
       for innocent reasons, so the caller must say ``hsrp_expected``; otherwise the verdict is
       "inconclusive" - never a fraud signal on its own.
2. ``planarity``  A screen or print is one flat plane; a real scene has depth. Between two frames
   with camera TRANSLATION, a flat scene's motion is a single homography; depth breaks it.
   Weak evidence: needs real translation (not pure rotation, which also looks planar), a textured
   scene with real depth, and does not cover a screen held in front of a real background. Use it
   only as supporting evidence.

Every verdict has an "inconclusive" outcome and says why. Thresholds are provisional and were
only exercised against a physics *simulation* in tests - they have not been validated on real
captures. That validation is the first thing to do with the capture-app data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------- helpers
def _lum_linear(rgb: np.ndarray) -> np.ndarray:
    """Approximate linear luminance (sRGB-ish gamma 2.2)."""
    x = (rgb.astype(np.float32) / 255.0) ** 2.2
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel displacement (dx, dy) of content from ``a`` to ``b`` (feature at p in a is at
    p+d in b) by phase correlation, and the correlation peak height (confidence 0..1)."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-6
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[iy, ix])

    def refine(m1, m0, p1):  # parabolic peak interpolation
        den = m1 - 2 * m0 + p1
        return 0.0 if abs(den) < 1e-9 else 0.5 * (m1 - p1) / den

    dx = ix + refine(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    dy = iy + refine(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    dx = dx - w if dx > w / 2 else dx
    dy = dy - h if dy > h / 2 else dy
    return float(dx), float(dy), peak


def align_translation(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Shift ``mov`` (HxWx3) by whole pixels so it lines up with ``ref`` - compensates the small
    hand-shake between a flash-off and a flash-on frame. Uses luminance phase correlation."""
    ga, gb = _lum_linear(ref) ** 0.5, _lum_linear(mov) ** 0.5
    dx, dy, _ = phase_shift(gb, ga)   # displacement that moves mov onto ref
    return np.roll(np.roll(mov, int(round(dy)), axis=0), int(round(dx)), axis=1)


# ------------------------------------------------------------------ 1. retroreflection
MAX_GAIN_SPREAD = 2.5      # p90/p10 of the flash gain over the bright plate background: above this it is not uniform
HOTSPOT_P90_OVER_P50 = 3.0
HOTSPOT_MAX_MEDIAN_GAIN = 2.0
GAIN_MIN_PIXELS = 50


@dataclass
class RetroResult:
    verdict: str                      # retroreflective | not_retroreflective | specular_hotspot | inconclusive
    ratio: float | None = None
    gain_plate: float | None = None
    gain_surround: float | None = None
    reasons: list[str] = field(default_factory=list)
    gain_spread: float | None = None  # p90/p10 of the per-pixel flash gain over the plate background (1.0 = uniform)
    hotspot: bool = False             # the gain is concentrated in a small bright spot (glare on glass), not plate-wide

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def retroreflection(off: np.ndarray, on: np.ndarray, plate_mask: np.ndarray, *,
                    hsrp_expected: bool = True, min_ratio: float = 2.5, max_ratio: float = 1.4,
                    ring_px: int | None = None, align: bool = True) -> RetroResult:
    """``off`` / ``on``: flash-off and flash-on RGB frames of the same scene (same exposure
    settings). ``plate_mask``: bool HxW of the plate's *white background* region (the detector's
    rectified quad, eroded so characters/border are mostly excluded is fine - medians are used)."""
    if not hsrp_expected:
        return RetroResult("inconclusive", reasons=["plate is not expected to be retroreflective (non-HSRP)"])
    if align:
        on = align_translation(off, on)
    lo, ln = _lum_linear(off), _lum_linear(on)
    pm = plate_mask.astype(bool)
    r = ring_px or max(6, int(0.5 * np.sqrt(pm.sum())))
    ring = ndi.binary_dilation(pm, iterations=r) & ~ndi.binary_dilation(pm, iterations=max(2, r // 4))
    if pm.sum() < 300 or ring.sum() < 300:
        return RetroResult("inconclusive", reasons=["plate or surround region too small to measure"])
    eps = 1e-3
    off_p, on_p = float(np.median(lo[pm])), float(np.median(ln[pm]))
    off_r, on_r = float(np.median(lo[ring])), float(np.median(ln[ring]))
    gp, gr = (on_p + eps) / (off_p + eps), (on_r + eps) / (off_r + eps)
    res = RetroResult("inconclusive", gain_plate=round(gp, 3), gain_surround=round(gr, 3))
    if max(gp, gr) < 1.03:
        res.reasons.append("flash does not appear to have fired (no frame brightened)")
        return res
    if off_p > 0.85:
        res.reasons.append("plate already saturated without flash (strong ambient light): not measurable")
        return res
    if off_p < 0.01 and off_r < 0.01:
        res.reasons.append("scene too dark in the flash-off frame")
        return res
    # Headroom: the sensor clips at 1.0, so the largest gain the plate can show is (1)/(off_p).
    # If that cannot reach min_ratio relative to the surround's gain, a retroreflective plate
    # would be MEASURED as "not retroreflective" - the worst failure here - so refuse to judge.
    headroom = (1.0 + eps) / (off_p + eps)
    if headroom / max(gr, 1.0) < min_ratio * 1.15:
        res.reasons.append(f"no headroom: plate is already {off_p:.2f} (linear) without flash, so it cannot show "
                           f"a {min_ratio}x gain. Meter exposure on the plate (aim <= ~0.3) and lock it for both frames.")
        return res
    res.ratio = round(gp / gr, 3)
    spread, hotspot = gain_uniformity(lo, ln, pm, eps)
    res.gain_spread = None if spread is None else round(spread, 3)
    res.hotspot = hotspot
    if hsrp_expected and hotspot:
        # A bright spot on an otherwise unchanged plate is a reflection of the flash on a glossy surface (the glass
        # of a screen, a laminated print), not the plate-wide return of retroreflective sheeting.
        res.verdict = "specular_hotspot"
        res.reasons.append("the flash lit a small hot-spot on the plate while the rest barely changed: that is glare on a "
                           "glossy surface (a screen or print), not the plate-wide return of retroreflective sheeting")
    elif res.ratio >= min_ratio:
        if spread is None or spread <= MAX_GAIN_SPREAD:
            res.verdict = "retroreflective"
        else:
            res.reasons.append(f"the flash gain is not uniform across the plate (p90/p10 = {spread:.1f} > {MAX_GAIN_SPREAD}): "
                               "retroreflective sheeting brightens evenly, so this is not accepted as proof")
    elif res.ratio <= max_ratio:
        res.verdict = "not_retroreflective"
        res.reasons.append(f"plate brightened {gp:.2f}x under flash vs {gr:.2f}x for its surround "
                           "(a screen emits its own light; a print reflects like its surroundings)")
    else:
        res.reasons.append("ratio in the grey zone between retroreflective and not")
    return res


def gain_uniformity(lo: np.ndarray, ln: np.ndarray, plate: np.ndarray, eps: float = 1e-3) -> tuple[float | None, bool]:
    """How evenly the flash brightened the plate. Retroreflective sheeting returns the light straight back, so
    the WHOLE white background gains about the same factor. A flash reflected off glass (a screen, a glossy print)
    lights a small spot: a few pixels gain a lot, the rest nothing. Over the plate-mask pixels that are bright
    without flash (the white background, not the dark characters): G = (on + eps) / (off + eps);
    returns (p90/p10, hotspot) with hotspot = p90/p50 >= 3 and median G < 2. (None, False) with too few pixels."""
    pm = plate.astype(bool)
    bg = pm & (lo >= 0.5 * float(np.median(lo[pm])))
    if int(bg.sum()) < GAIN_MIN_PIXELS:
        return None, False
    g = (ln[bg] + eps) / (lo[bg] + eps)
    p10, p50, p90 = (float(v) for v in np.percentile(g, [10, 50, 90]))
    return p90 / p10, bool(p90 / p50 >= HOTSPOT_P90_OVER_P50 and p50 < HOTSPOT_MAX_MEDIAN_GAIN)


# ------------------------------------------------------------------------- 2. planarity
@dataclass
class PlanarityResult:
    verdict: str                      # planar | non_planar | inconclusive
    residual_px: float | None = None
    baseline_px: float | None = None
    tiles: int = 0
    inlier_frac: float | None = None
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _dlt(p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    def norm(x):
        c = x.mean(0)
        s = np.sqrt(2) / (np.sqrt(((x - c) ** 2).sum(1)).mean() + 1e-9)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (np.c_[x, np.ones(len(x))] @ T.T)[:, :2], T

    pn, tp = norm(p)
    qn, tq = norm(q)
    A = []
    for (x, y), (u, v) in zip(pn, qn):
        A += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    try:
        h = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    except np.linalg.LinAlgError:
        return None
    H = np.linalg.inv(tq) @ h @ tp
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)


def _apply(H, p):
    x = np.c_[p, np.ones(len(p))] @ H.T
    return x[:, :2] / x[:, 2:3]


def planarity(a: np.ndarray, b: np.ndarray, *, work: int = 512, tile: int = 64,
              min_baseline_px: float = 1.5, seed: int = 0) -> PlanarityResult:
    """Is the motion between two frames explained by ONE plane? ``a``/``b``: RGB frames.
    Statistic: the fraction of tile motions the best single homography explains (RANSAC inlier
    fraction). A median residual would be dominated by whichever plane covers most tiles."""
    sc = work / max(a.shape[:2])
    ga = ndi.zoom(_lum_linear(a) ** 0.5, sc, order=1)
    gb = ndi.zoom(_lum_linear(b) ** 0.5, sc, order=1)
    ga, gb = ga - ndi.gaussian_filter(ga, 6), gb - ndi.gaussian_filter(gb, 6)   # texture only
    h, w = ga.shape
    P, Q = [], []
    for y in range(0, h - tile + 1, tile // 2):
        for x in range(0, w - tile + 1, tile // 2):
            ta, tb = ga[y : y + tile, x : x + tile], gb[y : y + tile, x : x + tile]
            if ta.std() < 0.004 or tb.std() < 0.004:
                continue
            dx, dy, pk = phase_shift(ta, tb)
            if pk < 0.18 or abs(dx) > tile / 3 or abs(dy) > tile / 3:
                continue
            c = np.array([x + tile / 2, y + tile / 2])
            P.append(c)
            Q.append(c + [dx, dy])
    res = PlanarityResult("inconclusive", tiles=len(P))
    if len(P) < 12:
        res.reasons.append("too few textured, matchable tiles")
        return res
    P, Q = np.array(P), np.array(Q)
    mot = np.linalg.norm(Q - P, axis=1)
    base = float(np.median(mot))
    res.baseline_px = round(base, 2)
    if base < min_baseline_px:
        res.reasons.append(f"camera moved only {base:.2f}px at {work}px width: not enough baseline")
        return res
    rng = np.random.default_rng(seed)
    tau = max(0.35, 0.06 * base)                     # px tolerance at the working resolution
    best, best_in = None, -1
    with np.errstate(all="ignore"):
        for _ in range(300):
            idx = rng.choice(len(P), 4, replace=False)
            H = _dlt(P[idx], Q[idx])
            if H is None:
                continue
            err = np.linalg.norm(_apply(H, P) - Q, axis=1)
            n_in = int(np.nansum(err <= tau))
            if n_in > best_in:
                best, best_in, best_err_all = H, n_in, err
    if best is None:
        return res
    frac = best_in / len(P)
    res.residual_px = round(float(np.nanmedian(best_err_all)), 3)
    res.inlier_frac = round(frac, 3)
    if frac >= 0.85:
        res.verdict = "planar"
        res.reasons.append(f"{frac:.0%} of tiles move as ONE plane (motion {base:.1f}px). Consistent with a screen/print - "
                           "but also with a flat, shallow scene or a pure camera rotation.")
    elif frac < 0.8:
        res.verdict = "non_planar"
    else:
        res.reasons.append("ambiguous planarity")
    return res
