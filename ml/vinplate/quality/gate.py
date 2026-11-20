"""Capture-quality gate: cheap checks that decide "retake" before any recognition is attempted.

This is the most effective glare / blur / smudge defence there is, and it is not a model: a
live preview can run it every frame and tell the user *what to change* ("move closer", "tilt
to avoid glare") while a better photo is still obtainable. Reconstructing a destroyed
character afterwards is a last resort that can only ever be flagged, never trusted.

All thresholds below are PROVISIONAL (chosen from first principles, then eyeballed on the
synthetic set); they must be tuned on real captures against recognition success before the
gate is allowed to block users. ``calibrated`` on the report says so.

Cost: a few array passes over a ~48 px-high crop - sub-millisecond-scale in native code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

THRESHOLDS = dict(
    min_height_px=24,        # glyph rows below this cannot be read reliably
    min_sharpness=4.0,       # re-blur ratio on the 48-px-high normalised crop (sharp ~10-23, sigma=3 blur ~2-5)
    max_glare_frac=0.06,     # saturated pixels
    max_glare_blob=0.025,    # largest connected saturated blob, as fraction of crop area
    min_mean=40.0, max_mean=215.0,
    min_contrast=0.18,       # (p95 - p5) / 255
    max_clipped_dark=0.25,
)


@dataclass
class QualityReport:
    metrics: dict[str, float]
    issues: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)
    calibrated: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"ok": self.ok, "metrics": self.metrics, "issues": self.issues,
                "guidance": self.guidance, "calibrated": self.calibrated}


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def assess_crop(img: np.ndarray, thresholds: dict | None = None) -> QualityReport:
    """``img``: HxW or HxWx3 uint8 crop of one plate / VIN region (detector output)."""
    t = {**THRESHOLDS, **(thresholds or {})}
    g = _gray(img)
    h, w = g.shape
    # normalise scale so sharpness does not depend on how big the crop happens to be
    scale = 48.0 / h
    gn = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((max(8, int(w * scale)), 48), Image.BILINEAR), np.float32)

    # No-reference sharpness by re-blur: a sharp image loses most of its Laplacian energy when
    # blurred a little more, an already-blurry one barely changes. Unlike raw Laplacian
    # variance this does not depend on how much text/edge content the crop happens to hold.
    sharp = float(ndi.laplace(gn).var() / (ndi.laplace(ndi.gaussian_filter(gn, 1.2)).var() + 1e-6))
    sat = gn >= 250
    glare_frac = float(sat.mean())
    lab, n = ndi.label(sat)
    blob = float(np.bincount(lab.ravel())[1:].max() / sat.size) if n else 0.0
    p5, p95 = np.percentile(gn, [5, 95])
    contrast = float((p95 - p5) / 255)
    m = {
        "height_px": float(h), "sharpness": round(sharp, 2), "glare_frac": round(glare_frac, 4),
        "glare_blob": round(blob, 4), "mean": round(float(gn.mean()), 1), "contrast": round(contrast, 3),
        "dark_clipped": round(float((gn <= 5).mean()), 4),
    }

    issues, tips = [], []
    if h < t["min_height_px"]:
        issues.append("too small"); tips.append("Move closer so the plate/number fills more of the frame.")
    if sharp < t["min_sharpness"]:
        issues.append("blurred"); tips.append("Hold the phone steady and tap to focus.")
    if glare_frac > t["max_glare_frac"] or blob > t["max_glare_blob"]:
        issues.append("glare"); tips.append("Tilt the phone a little or step aside to move the reflection off the number.")
    if m["mean"] < t["min_mean"] or m["dark_clipped"] > t["max_clipped_dark"]:
        issues.append("too dark"); tips.append("Add light, or turn the torch on and move to avoid reflections.")
    if m["mean"] > t["max_mean"]:
        issues.append("overexposed"); tips.append("Reduce the light on the plate or avoid direct flash.")
    if contrast < t["min_contrast"] and "overexposed" not in issues and "too dark" not in issues:
        issues.append("low contrast"); tips.append("Clean the surface if possible and retake from a different angle.")
    return QualityReport(m, issues, tips)
