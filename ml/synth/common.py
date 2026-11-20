"""Fonts and small drawing helpers shared by the plate and VIN renderers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ImageFont

FONT_ROOT = Path(__file__).resolve().parent / "fonts"


class FontBank:
    """OFL fonts grouped by role: hsrp / painted / print / stamp (see scripts/fetch_fonts.py)."""

    def __init__(self, root: Path = FONT_ROOT):
        self.by_role: dict[str, list[Path]] = {}
        if not root.is_dir():
            raise FileNotFoundError(f"{root} missing: run `python scripts/fetch_fonts.py`")
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            role = (d / "ROLE").read_text().strip() if (d / "ROLE").exists() else "print"
            self.by_role.setdefault(role, []).extend(sorted(d.glob("*.ttf")))
        if not any(self.by_role.values()):
            raise FileNotFoundError(f"no .ttf files under {root}")

    def pick(self, role: str, rng: np.random.Generator) -> Path:
        files = self.by_role[role]
        return files[int(rng.integers(len(files)))]


_BANK: FontBank | None = None


def font_bank() -> FontBank:
    global _BANK
    if _BANK is None:
        _BANK = FontBank()
    return _BANK


@lru_cache(maxsize=1024)
def load_font(path: str, size: int, weight: int = 0) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    if weight and "[" in Path(path).name:
        try:
            axes = f.get_variation_axes()
            if len(axes) == 1:  # a lone wght axis; multi-axis fonts keep their default instance
                f.set_variation_by_axes([int(np.clip(weight, axes[0]["minimum"], axes[0]["maximum"]))])
        except Exception:
            pass
    return f


def size_for_cap_height(path: str, cap_px: float, weight: int = 0) -> int:
    """Font size whose capital-letter height is ``cap_px`` pixels."""
    f = load_font(path, 100, weight)
    b = f.getbbox("H", anchor="ls")
    cap100 = max(1, b[3] - b[1])
    return max(6, int(round(cap_px * 100 / cap100)))


def jitter(rng: np.random.Generator, c, amount: float):
    """Colour jitter, clipped to 0..255."""
    c = np.asarray(c, dtype=np.float32)
    return tuple(int(v) for v in np.clip(c + rng.normal(0, amount, 3), 0, 255))
