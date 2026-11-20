"""Text-line normalisation shared by training and inference.

The recogniser is a *line* model with a fixed input height. Plate rows fit a 48x256 canvas,
VIN lines a 48x512 one. Training data and the deployed pipeline must produce byte-identical
canvases from a given crop, so both call ``fit_line``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PLATE_LINE_SHAPE = (48, 256)   # (H, W): one plate row (a 1-row plate, or one row of a 2-row plate)
VIN_LINE_SHAPE = (48, 512)

_INTERP = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "area": Image.BOX, "lanczos": Image.LANCZOS}


def fit_line(img: np.ndarray, shape: tuple[int, int], interp: str = "bilinear") -> np.ndarray:
    """Resize a line crop to height H keeping aspect, centre it on a W-wide canvas padded with
    the crop's own border colour. If it is wider than the canvas it is squeezed to fit.
    ``img``: HxWx3 uint8 -> HxWx3 uint8 of exactly ``shape``."""
    H, W = shape
    h, w = img.shape[:2]
    new_w = max(8, int(round(w * H / h)))
    pil = Image.fromarray(img)
    if new_w > W:
        return np.asarray(pil.resize((W, H), _INTERP[interp]))
    body = np.asarray(pil.resize((new_w, H), _INTERP[interp]))
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).mean(axis=0)
    out = np.empty((H, W, 3), np.uint8)
    out[:] = border.astype(np.uint8)
    x0 = (W - new_w) // 2
    out[:, x0 : x0 + new_w] = body
    return out


def to_gray(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> HxW uint8, Rec.601 luma. The recogniser is single-channel: colour adds
    nothing to reading characters and costs input bandwidth. Training and inference share this."""
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114 + 0.5).astype(np.uint8)
