"""Split a rectified plate into its printed rows."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

TWO_ROW_MAX_ASPECT = 2.6   # width/height of a rectified plate: 1-row ~4.2, 2-row ~1.6-2.2


def looks_two_row(h: int, w: int) -> bool:
    return w / h < TWO_ROW_MAX_ASPECT


def split_rows(rgb: np.ndarray, overlap: float = 0.12) -> list[np.ndarray]:
    """Cut a 2-row plate at the quietest horizontal band between the rows (smallest vertical
    gradient energy in the middle 30%), with ``overlap`` of the plate height shared by both
    crops so a character that straddles the cut is not lost. The recogniser is trained on
    crops with exactly this kind of neighbouring-row leakage."""
    h = rgb.shape[0]
    g = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32) if rgb.ndim == 3 else rgb.astype(np.float32)
    energy = ndi.uniform_filter1d(np.abs(np.diff(g, axis=0)).mean(axis=1), max(3, h // 20))
    lo, hi = int(0.35 * h), int(0.65 * h)
    cut = lo + int(np.argmin(energy[lo:hi]))
    m = int(overlap * h)
    return [rgb[: min(h, cut + m)], rgb[max(0, cut - m) :]]
