import numpy as np
from PIL import Image, ImageFilter

from synth.plates import render_plate
from vinplate.quality import assess_crop


def _plate(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(render_plate(rng, spec=dict(layout="standard", groups=["MH", "12", "AB", "1234"],
                                               text="MH12AB1234", kind="standard")).img, 0, 255).astype(np.uint8)


def test_sharp_plate_passes():
    r = assess_crop(_plate())
    assert r.ok, (r.issues, r.metrics)


def test_blur_is_flagged_with_guidance():
    blurred = np.asarray(Image.fromarray(_plate()).filter(ImageFilter.GaussianBlur(5)))
    r = assess_crop(blurred)
    assert "blurred" in r.issues and any("steady" in g for g in r.guidance)


def test_glare_blob_is_flagged():
    img = _plate().copy()
    h, w = img.shape[:2]
    img[int(0.1 * h):int(0.9 * h), int(0.3 * w):int(0.55 * w)] = 255
    assert "glare" in assess_crop(img).issues


def test_dark_and_tiny_are_flagged():
    dark = (_plate().astype(np.float32) * 0.1).astype(np.uint8)
    assert "too dark" in assess_crop(dark).issues
    tiny = _plate()[::10, ::10]
    assert "too small" in assess_crop(tiny).issues
