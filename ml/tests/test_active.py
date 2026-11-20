"""Physics-simulation tests for the active presentation-attack checks.

IMPORTANT: these prove the code computes what the physics says it should, on a toy scene model.
They do NOT prove the checks work on real phones, real flashes and real plates; that needs real
captures (docs/CAPTURE_PROTOCOL.md).
"""

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from vinplate.authenticity.active import phase_shift, planarity, retroreflection


def textured(h, w, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    g = sum(ndi.gaussian_filter(rng.standard_normal((h, w)), s) * a for s, a in ((1.5, 5), (5, 10), (16, 14), (40, 18)))
    g = (g - g.min()) / np.ptp(g)
    return np.clip(0.08 + 0.3 * g * scale, 0, 1)


def to_u8(lin):
    return (255 * np.clip(lin, 0, 1) ** (1 / 2.2)).astype(np.uint8)[..., None].repeat(3, -1)


H, W = 240, 400
PLATE = np.zeros((H, W), bool)
PLATE[90:150, 100:300] = True


def scenes(kind, seed=0):
    body = textured(H, W, seed)
    off = body.copy()
    off[PLATE] = 0.18 + 0.01 * np.random.default_rng(seed).standard_normal(PLATE.sum())      # white plate background
    lit = 0.9 + 0.1 * np.random.default_rng(seed + 1).random((H, W))
    if kind == "real_hsrp":       # diffuse paint brightens a little, retroreflective sheeting a lot
        on = off * (1 + 0.25 * lit)
        on[PLATE] = off[PLATE] * (1 + 0.25) + 1.4
    elif kind == "screen":        # emissive: flash adds only a faint veiling glare on the glass
        on = off + 0.015 * lit
    elif kind == "print":         # diffuse everywhere: plate and surround scale together
        on = off * (1 + 0.25 * lit)
    else:
        raise ValueError(kind)
    return to_u8(off), to_u8(on)


@pytest.mark.parametrize("kind,verdict", [("real_hsrp", "retroreflective"), ("screen", "not_retroreflective"),
                                          ("print", "not_retroreflective")])
def test_flash_gain_separates_real_screen_print(kind, verdict):
    off, on = scenes(kind)
    r = retroreflection(off, on, PLATE)
    assert r.verdict == verdict, r


def test_robust_to_handshake_and_exposure_change():
    off, on = scenes("real_hsrp", 3)
    shaken = np.roll(np.roll(on, 3, axis=0), -4, axis=1)
    dim = (shaken.astype(np.float32) * 0.85).astype(np.uint8)       # auto-exposure drift between frames
    assert retroreflection(off, dim, PLATE).verdict == "retroreflective"
    off, on = scenes("screen", 4)
    assert retroreflection(off, np.roll(on, 3, axis=1), PLATE).verdict == "not_retroreflective"


def test_never_a_fraud_signal_when_not_applicable_or_not_measurable():
    off, on = scenes("screen")
    assert retroreflection(off, on, PLATE, hsrp_expected=False).verdict == "inconclusive"   # old painted plate
    assert retroreflection(off, off.copy(), PLATE).verdict == "inconclusive"                  # flash did not fire
    tiny = np.zeros((H, W), bool)
    tiny[100:105, 100:105] = True
    assert retroreflection(off, on, tiny).verdict == "inconclusive"
    sat = np.full((H, W, 3), 250, np.uint8)
    assert retroreflection(sat, sat, PLATE).verdict == "inconclusive"                         # washed out by sun
    bright_off, bright_on = np.full((H, W, 3), 170, np.uint8), np.full((H, W, 3), 200, np.uint8)
    r = retroreflection(bright_off, bright_on, PLATE)                                          # plate with no headroom
    assert r.verdict == "inconclusive" and any("headroom" in x for x in r.reasons)


def test_phase_shift_is_subpixel_accurate():
    a = textured(128, 128, 7)
    b = ndi.shift(a, (2.4, -3.7), order=3, mode="wrap")
    dx, dy, pk = phase_shift(a, b)
    assert abs(dx + 3.7) < 0.15 and abs(dy - 2.4) < 0.15 and pk > 0.3


def warp(img, shift, rot_deg=0.0):
    h, w = img.shape[:2]
    a = np.deg2rad(rot_deg)
    c, s = np.cos(a), np.sin(a)
    # inverse affine for PIL: out(x,y) = in(M @ (x,y) + t)
    coeffs = (c, s, -shift[0] - c * w / 2 - s * h / 2 + w / 2, -s, c, -shift[1] + s * w / 2 - c * h / 2 + h / 2)
    return np.asarray(Image.fromarray(img).transform((w, h), Image.AFFINE, coeffs, Image.BICUBIC), np.float32)


def test_planarity_flat_vs_layered_scene():
    big = to_u8(textured(480, 640, 11)).astype(np.float32)[..., 0]
    a = np.stack([big] * 3, -1).astype(np.uint8)
    flat_b = np.stack([warp(big.astype(np.uint8), (7, 3), 1.0)] * 3, -1).astype(np.uint8)
    # layered: a foreground block moves 3x as far as the background (parallax)
    bg = warp(big.astype(np.uint8), (3, 1), 0.3)
    fg = warp(big.astype(np.uint8), (11, 4), 0.3)
    m = np.zeros(big.shape, bool)
    m[100:380, 140:520] = True
    lay = np.where(m, fg, bg)
    layered_b = np.stack([lay] * 3, -1).astype(np.uint8)
    assert planarity(a, flat_b).verdict == "planar"
    assert planarity(a, layered_b).verdict == "non_planar"


def test_planarity_is_inconclusive_without_baseline():
    big = to_u8(textured(480, 640, 12))
    assert planarity(big, big).verdict == "inconclusive"


# ------------------------------------------------------- integration: Authenticator
def test_authenticator_runs_what_it_can_and_reports_the_rest():
    from vinplate.authenticity.assess import Authenticator, pick_patches

    off, on = scenes("screen", 5)
    big = np.asarray(Image.fromarray(off).resize((640, 384), Image.BILINEAR))
    p = pick_patches(big, (0, 0, 640, 384))
    assert p.shape == (4, 96, 96, 3)                          # native crops, never resized
    assert pick_patches(big, (0, 0, 60, 60)) is None          # region smaller than a patch

    seen = {}
    auth = Authenticator(pad_model=lambda patches: seen.setdefault("n", len(patches)) and 0.97)
    d = auth.assess([off, on], roi=(0, 0, W, H), flash_on=on, plate_mask=PLATE, hsrp_expected=True)
    assert seen["n"] == 4
    # a screen: failed flash physics (+2.0) and a high learned score (+2.0) -> rejected
    assert d.status == "reject" and d.pa_score >= 3.0
    assert "retroreflection" not in d.not_run

    plain = Authenticator().assess([off])                     # nothing but one frame, no model
    assert plain.status == "no_flags"
    assert {"recapture", "retroreflection", "planarity"} <= set(plain.not_run)


# ---------------------------------------------------------------- flash uniformity / hot-spot
def _with_glare_patch(frac=0.25, gain=0.9):
    """A screen replay whose flash reflection washes out the right ``frac`` of the plate (a glare patch on the
    glass), as in the real attack: the rest of the plate barely changes."""
    off, on = scenes("screen")
    lin = (on[..., 0].astype(np.float32) / 255.0) ** 2.2
    xx = np.arange(W)[None, :].repeat(H, 0)
    x0 = 100 + (1 - frac) * 200                                         # the plate spans x 100..300
    edge = 1 / (1 + np.exp(-(xx - x0) / 6.0))
    rows = 0.7 + 0.3 * np.exp(-((np.arange(H)[:, None] - 120) / 30.0) ** 2)
    return off, to_u8(np.clip(lin + gain * edge * rows, 0, 1))


def test_uniform_hsrp_gain_stays_retroreflective_and_reports_its_spread():
    off, on = scenes("real_hsrp")
    r = retroreflection(off, on, PLATE)
    assert r.verdict == "retroreflective" and r.hotspot is False
    assert r.gain_spread is not None and r.gain_spread <= 1.5, r


@pytest.mark.parametrize("frac", [0.15, 0.25, 0.35])
def test_glare_patch_on_a_screen_is_a_specular_hotspot(frac):
    off, on = _with_glare_patch(frac)
    r = retroreflection(off, on, PLATE)
    assert r.verdict == "specular_hotspot" and r.hotspot is True, r
    assert r.gain_spread is not None and r.gain_spread > 2.5
    assert any("glare on a glossy surface" in x for x in r.reasons)


def test_hotspot_rule_needs_hsrp_expected_and_a_flashed_frame():
    off, on = _with_glare_patch()
    assert retroreflection(off, on, PLATE, hsrp_expected=False).verdict == "inconclusive"
    assert retroreflection(off, off.copy(), PLATE).verdict == "inconclusive"          # the flash did not fire


def test_strong_but_non_uniform_gain_is_not_accepted_as_retroreflective():
    """A big overall gain (ratio >= 2.5) whose spread exceeds 2.5 must not pass: it is inconclusive, never a pass."""
    off, on = scenes("real_hsrp")
    lin = (on[..., 0].astype(np.float32) / 255.0) ** 2.2
    lo = (off[..., 0].astype(np.float32) / 255.0) ** 2.2
    # keep the plate bright overall (most of it gains ~5x) but let its left 30% gain almost nothing
    boost = np.where(np.arange(W)[None, :] < 160, 0.05, 1.0)
    lin2 = np.where(PLATE, lo * 1.25 + (lin - lo * 1.25) * boost, lin)
    r = retroreflection(off, to_u8(np.clip(lin2, 0, 1)), PLATE)
    assert r.ratio >= 2.5 and r.hotspot is False                       # the plate-level gain alone WOULD pass ...
    assert r.verdict == "inconclusive" and any("not uniform" in x for x in r.reasons), r      # ... uniformity stops it


def test_gain_uniformity_ignores_dark_characters_and_needs_enough_pixels():
    from vinplate.authenticity.active import gain_uniformity
    lo = np.full((60, 200), 0.2, np.float32)
    ln = lo * 3.0
    lo2, ln2 = lo.copy(), ln.copy()
    lo2[:, 90:110], ln2[:, 90:110] = 0.01, 0.01 * 1.0                         # dark glyph columns that do not brighten
    mask = np.ones((60, 200), bool)
    spread, hot = gain_uniformity(lo2, ln2, mask)
    assert hot is False and spread == pytest.approx(1.0, abs=0.05)             # the dark pixels are excluded from the gain map
    assert gain_uniformity(lo[:5, :5], ln[:5, :5], np.ones((5, 5), bool)) == (None, False)
