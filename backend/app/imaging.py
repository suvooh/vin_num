"""Image decoding: upright RGB pixels (longest side capped) and the EXIF capture time."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from PIL import ExifTags, Image, ImageOps

MAX_SIDE = 1600


def _exif_time(img: Image.Image) -> datetime | None:
    try:
        exif = img.getexif()
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        raw = sub.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def decode(payload: bytes, max_side: int = MAX_SIDE) -> tuple[np.ndarray, datetime | None]:
    """Upright RGB pixels (longest side capped at ``max_side``) and EXIF capture time."""
    try:
        img = Image.open(io.BytesIO(payload))
        taken = _exif_time(img)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError(f"could not decode image: {exc}") from exc
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return np.asarray(img), taken
