"""Photometric enhancement of a plate / VIN crop: glare, shading and low-contrast handling.

Every function returns a uint8 HxWx3 image of EXACTLY the input size. That is deliberate: the
recogniser maps a crop to a fixed number of time steps proportional to its width, so variants of
one crop produce time-aligned posteriors that can be fused step by step
(``vinplate.decode.entropy_fuse``) - the single-frame analogue of burst fusion.

What this can and cannot do (the same honesty rule as the rest of the package):
  * It RECOVERS contrast that is still in the pixels: veiling glare and halos (bright but not
    clipped), cast shadows, uneven lighting, mild blur, a weak colour channel.
  * It cannot recover pixels that are clipped to pure white or painted over. ``deglare`` only
    removes the hard white blob so it stops being read as a stroke; whatever was underneath stays
    unknown and must come out of the decoder as ``?``, never as a guess.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CLIP_LEVEL = 247   # all three channels at or above this = specular clipping


def luminance(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def _rep3(g: np.ndarray) -> np.ndarray:
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    return np.repeat(g8[..., None], 3, axis=2)


def _stretch(g: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(g, [lo_p, hi_p])
    if hi - lo < 8:                      # flat: nothing to stretch, don't amplify noise
        return g
    return (g - lo) * (255.0 / (hi - lo))


def clip_mask(rgb: np.ndarray, level: int = CLIP_LEVEL) -> np.ndarray:
    """Pixels clipped to white in every channel (specular highlight / flash hot-spot), with
    isolated single pixels (sensor noise on a white plate) removed."""
    m = (rgb >= level).all(axis=2)
    return ndi.binary_opening(m, structure=np.ones((2, 2)))


def clip_stats(rgb: np.ndarray) -> dict[str, float]:
    m = clip_mask(rgb)
    frac = float(m.mean())
    lab, n = ndi.label(m)
    blob = float(np.bincount(lab.ravel())[1:].max() / m.size) if n else 0.0
    return {"clipped_frac": round(frac, 4), "largest_blob": round(blob, 4)}


# ------------------------------------------------------------------------------- variants
def stretch(rgb: np.ndarray) -> np.ndarray:
    """Luminance, percentile contrast stretch, replicated to 3 channels. Fixes a veil over the
    whole crop (haze, mild glare, backlight) and removes colour as a nuisance."""
    return _rep3(_stretch(luminance(rgb)))


def best_channel(rgb: np.ndarray) -> np.ndarray:
    """The single colour channel with the widest contrast. Yellow / green plates have almost no
    contrast in blue; a flash tinted by the paint changes which channel carries the strokes."""
    spread = [np.subtract(*np.percentile(rgb[..., c].astype(np.float32), [95, 5])) for c in range(3)]
    return _rep3(_stretch(rgb[..., int(np.argmax(spread))].astype(np.float32)))


def retinex(rgb: np.ndarray, sigma_frac: float = 0.5) -> np.ndarray:
    """Single-scale retinex on luminance: remove the slowly varying illumination (shadow edge,
    glare gradient, vignetting) and keep the reflectance, i.e. the strokes."""
    g = luminance(rgb) + 1.0
    sigma = max(3.0, sigma_frac * rgb.shape[0])
    r = np.log(g) - np.log(ndi.gaussian_filter(g, sigma) + 1.0)
    return _rep3(_stretch(r, 1.0, 99.0))


def sharpen(rgb: np.ndarray, amount: float = 1.4) -> np.ndarray:
    """Unsharp mask on the stretched luminance, for defocus / motion blur (soft recovery only)."""
    g = _stretch(luminance(rgb))
    sigma = max(1.0, rgb.shape[0] / 16.0)
    return _rep3(g + amount * (g - ndi.gaussian_filter(g, sigma)))


def deglare(rgb: np.ndarray) -> np.ndarray | None:
    """Remove hard-clipped hot-spots by diffusing in the surrounding (unclipped) values
    (normalised convolution), then stretch. Returns None when there is nothing to do - no
    clipped region, or the crop is so overexposed that the 'blob' is just the white plate."""
    m = clip_mask(rgb)
    frac = float(m.mean())
    if frac < 0.004 or frac > 0.45:
        return None
    hole = ndi.binary_dilation(m, iterations=2)
    g = luminance(rgb)
    w = (~hole).astype(np.float32)
    if w.sum() < 0.1 * w.size:
        return None
    sigma = max(3.0, rgb.shape[0] / 6.0)
    num = ndi.gaussian_filter(g * w, sigma)
    den = ndi.gaussian_filter(w, sigma)
    fill = num / np.maximum(den, 1e-3)
    out = np.where(hole, fill, g)
    return _rep3(_stretch(out))


def variants(rgb: np.ndarray, *, full: bool = True) -> list[tuple[str, np.ndarray]]:
    """The views the recogniser reads. ``orig`` is always first. With ``full=False`` only the
    cheap, broadly useful ones are produced."""
    out: list[tuple[str, np.ndarray]] = [("orig", rgb), ("stretch", stretch(rgb))]
    if full:
        out.append(("retinex", retinex(rgb)))
        out.append(("sharpen", sharpen(rgb)))
        out.append(("channel", best_channel(rgb)))
        dg = deglare(rgb)
        if dg is not None:
            out.append(("deglare", dg))
    return out
