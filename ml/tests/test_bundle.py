"""Capture-bundle tests: several frames of one vehicle -> status + liveness.

IMPORTANT, same caveat as test_active.py: the flash / parallax frames here come from a TOY physics
model (and recapture frames from ``synth/recapture.py``, a simulator). These tests prove the bundle
logic wires the checks together correctly and never turns missing evidence into a pass. They do NOT
show that the checks work on real phones, flashes and plates. No accuracy is asserted for PadNet.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from vinplate import runtime
from vinplate.authenticity import Authenticator, BundleDecision, Frame, assess_bundle, read_conflicts

ML = Path(__file__).resolve().parents[1]
PHOTOS = sorted((ML / "data" / "openimages" / "images").glob("*.jpg"))


# ------------------------------------------------------------------ toy scene (borrowed from test_active)
def textured(h, w, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    g = sum(ndi.gaussian_filter(rng.standard_normal((h, w)), s) * a for s, a in ((1.5, 5), (5, 10), (16, 14), (40, 18)))
    g = (g - g.min()) / np.ptp(g)
    return np.clip(0.08 + 0.3 * g * scale, 0, 1)


def to_u8(lin):
    return (255 * np.clip(lin, 0, 1) ** (1 / 2.2)).astype(np.uint8)[..., None].repeat(3, -1)


def warp(gray_u8, shift, rot_deg=0.0):
    h, w = gray_u8.shape
    a = np.deg2rad(rot_deg)
    c, s = np.cos(a), np.sin(a)
    coeffs = (c, s, -shift[0] - c * w / 2 - s * h / 2 + w / 2, -s, c, -shift[1] + s * w / 2 - c * h / 2 + h / 2)
    return np.asarray(Image.fromarray(gray_u8).transform((w, h), Image.AFFINE, coeffs, Image.BICUBIC), np.float32)


H, W = 480, 640
ROI = (200, 200, 440, 300)                       # x0, y0, x1, y1 : the "plate"
PLATE = np.zeros((H, W), bool)
PLATE[ROI[1] : ROI[3], ROI[0] : ROI[2]] = True


def flash_pair(kind, seed=0):
    body = textured(H, W, seed)
    off = body.copy()
    off[PLATE] = 0.18 + 0.01 * np.random.default_rng(seed).standard_normal(PLATE.sum())
    lit = 0.9 + 0.1 * np.random.default_rng(seed + 1).random((H, W))
    if kind == "real_hsrp":
        on = off * (1 + 0.25 * lit)
        on[PLATE] = off[PLATE] * 1.25 + 1.4
    elif kind == "screen":
        on = off + 0.015 * lit
    elif kind == "print":
        on = off * (1 + 0.25 * lit)
    else:
        raise ValueError(kind)
    return to_u8(off), to_u8(on)


def angle_frame(primary, layered):
    g = primary[..., 0]
    if layered:                                  # nearer block (the car) moves 3x as far as the background
        bg, fg = warp(g, (3, 1), 0.3), warp(g, (11, 4), 0.3)
        m = np.zeros(g.shape, bool)
        m[100:380, 140:520] = True
        out = np.where(m, fg, bg)
    else:                                        # a flat replay: everything moves as ONE plane
        out = warp(g, (7, 3), 1.0)
    return np.stack([out] * 3, -1).astype(np.uint8)


def jpeg(arr, **exif_tags):
    ex = Image.Exif()
    for k, v in exif_tags.items():
        ex[int(k, 16)] = v
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=92, exif=ex)
    return b.getvalue()


def bundle(kind="real_hsrp", angle=True, roi=ROI, seed=0, **frames_kw):
    off, on = flash_pair(kind, seed)
    fr = [Frame(off, "primary", roi=roi, name="front"), Frame(on, "flash_on", name="front_flash")]
    if angle:
        fr.append(Frame(angle_frame(off, layered=(kind == "real_hsrp")), "angle", name="left"))
    return fr


# ------------------------------------------------------------------------------- liveness
def test_genuine_hsrp_bundle_is_confirmed_live():
    d = assess_bundle(bundle("real_hsrp"), hsrp_expected=True)
    assert isinstance(d, BundleDecision)
    assert d.liveness == "confirmed" and d.status != "reject", d.reasons
    assert "retroreflection_passed" in d.positive or "retroreflection_passed" in d.evidence["liveness_basis"]
    assert "parallax_non_planar" in d.positive
    assert d.evidence["liveness_basis"] == ["retroreflection_passed"]      # parallax is listed, but is not the basis
    assert "retroreflection" not in d.not_run and "planarity" not in d.not_run
    assert d.evidence["calibrated"] is False


@pytest.mark.parametrize("kind", ["screen", "print"])
def test_replay_with_flash_pair_is_a_suspected_spoof(kind):
    d = assess_bundle(bundle(kind), hsrp_expected=True)
    assert d.liveness == "suspected_spoof", d.reasons
    assert d.status in ("review", "reject") and d.pa_score >= 2.0
    assert d.evidence["active"]["retroreflection"]["verdict"] == "not_retroreflective"
    assert "retroreflection_passed" not in d.positive


def test_screen_replay_plus_high_learned_score_is_rejected():
    auth = Authenticator(pad_model=lambda patches: 0.97)
    d = assess_bundle(bundle("screen"), authenticator=auth, hsrp_expected=True)
    assert d.status == "reject" and d.liveness == "suspected_spoof" and d.pa_score >= 3.0
    assert d.evidence["recapture_by_frame"]["front"] == pytest.approx(0.97)


def test_parallax_alone_does_not_confirm_liveness():
    """A screen held in front of a real background also shows depth, so depth alone is not liveness."""
    d = assess_bundle(bundle("real_hsrp"), hsrp_expected=False)      # painted plate: flash test does not apply
    assert d.liveness == "not_established" and d.evidence["liveness_basis"] == []
    assert "parallax_non_planar" in d.positive                        # still reported as supporting evidence
    assert any("depth alone does not establish liveness" in r for r in d.reasons)
    assert "retroreflection" in d.not_run


def test_parallax_with_a_clean_recapture_score_NEVER_confirms():
    """The real-attack regression: a photo of a monitor in front of an office has depth AND PadNet scored it
    0.003. Parallax + a clean PadNet score used to confirm liveness; neither is proof, alone or together."""
    auth = Authenticator(pad_model=lambda patches: 0.003)
    d = assess_bundle(bundle("real_hsrp"), authenticator=auth, hsrp_expected=False)
    assert d.liveness == "not_established" and d.evidence["liveness_basis"] == []
    assert "parallax_non_planar" in d.positive                       # supporting evidence is still reported
    assert any("clean score is never used as proof" in r for r in d.reasons)
    assert any("depth alone does not establish liveness" in r for r in d.reasons)


def test_parallax_with_a_suspicious_recapture_score_does_not_confirm():
    auth = Authenticator(pad_model=lambda patches: 0.6)
    d = assess_bundle(bundle("real_hsrp"), authenticator=auth, hsrp_expected=False)
    assert d.liveness != "confirmed"


def test_flat_angle_frames_alone_never_confirm():
    off, _ = flash_pair("screen")
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(angle_frame(off, layered=False), "angle")])
    assert d.liveness == "not_established"
    assert d.pa_score == pytest.approx(0.5)                          # the single weak planar cue, counted once


def test_two_flat_angles_count_the_planar_cue_once():
    off, _ = flash_pair("screen")
    fr = [Frame(off, "primary", roi=ROI), Frame(angle_frame(off, False), "angle", name="a"),
          Frame(angle_frame(off, False), "angle", name="b")]
    assert assess_bundle(fr).pa_score == pytest.approx(0.5)


def test_single_image_is_not_established_and_no_flags_is_not_liveness():
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI)])
    assert d.status == "no_flags" and d.liveness == "not_established" and not d.positive
    assert any("NO positive evidence" in r for r in d.reasons)
    assert {"retroreflection", "planarity", "recapture"} <= set(d.not_run)
    assert d.evidence["not_run_reasons"]["retroreflection"] == "no flash-on frame in the bundle"
    assert d.evidence["not_run_reasons"]["planarity"] == "no angle frames in the bundle"


def test_require_liveness_policy():
    off, _ = flash_pair("real_hsrp")
    single = [Frame(off, "primary", roi=ROI)]
    d = assess_bundle(single, require_liveness=True)
    assert d.status == "review" and any("liveness required" in r for r in d.reasons)
    ok = assess_bundle(bundle("real_hsrp"), hsrp_expected=True, require_liveness=True)
    assert ok.liveness == "confirmed" and ok.status != "reject"
    assert not any("liveness required" in r for r in ok.reasons)
    # policy never downgrades a reject
    auth = Authenticator(pad_model=lambda p: 0.97)
    assert assess_bundle(bundle("screen"), authenticator=auth, hsrp_expected=True, require_liveness=True).status == "reject"


def test_recapture_score_is_the_max_over_frames():
    scores = iter([0.1, 0.95, 0.1])
    auth = Authenticator(pad_model=lambda patches: next(scores))
    d = assess_bundle(bundle("real_hsrp"), authenticator=auth, hsrp_expected=True)
    assert d.evidence["recapture_by_frame"] == {"front": 0.1, "front_flash": 0.95, "left": 0.1}
    assert d.pa_score >= 2.0 and d.liveness == "suspected_spoof"     # one recaptured frame is an attack
    assert any("front_flash" in r and "0.95" in r for r in d.reasons)


# ------------------------------------------------------------------------ not_run, never a silent pass
def test_flash_frame_without_roi_is_not_run_with_a_reason():
    d = assess_bundle(bundle("real_hsrp", roi=None, angle=False), hsrp_expected=True)
    assert "retroreflection" in d.not_run
    assert "ROI" in d.evidence["not_run_reasons"]["retroreflection"]
    assert d.liveness == "not_established" and "retroreflection_passed" not in d.positive


def test_roi_outside_the_image_counts_as_missing():
    d = assess_bundle(bundle("real_hsrp", roi=(5000, 5000, 5100, 5100), angle=False), hsrp_expected=True)
    assert "retroreflection" in d.not_run and d.liveness == "not_established"


def test_flash_frame_with_wrong_aspect_ratio_is_not_run():
    off, on = flash_pair("real_hsrp")
    portrait = np.ascontiguousarray(np.rot90(on))
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(portrait, "flash_on")], hsrp_expected=True)
    assert "retroreflection" in d.not_run and "aspect" in d.evidence["not_run_reasons"]["retroreflection"]


def test_flash_frame_of_another_size_is_resized_not_rejected():
    off, on = flash_pair("real_hsrp")
    small = np.asarray(Image.fromarray(on).resize((W // 2, H // 2), Image.BILINEAR))
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(small, "flash_on")], hsrp_expected=True)
    assert d.evidence["active"]["retroreflection"]["verdict"] == "retroreflective"


def test_angle_frame_with_wrong_aspect_ratio_is_skipped():
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(np.ascontiguousarray(np.rot90(off)), "angle", name="turned")])
    assert "planarity" in d.not_run and "turned" in d.evidence["not_run_reasons"]["planarity"]


def test_no_flash_gain_at_all_stays_inconclusive_not_spoof():
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(off.copy(), "flash_on")], hsrp_expected=True)
    assert d.liveness == "not_established" and "retroreflection" in d.not_run
    assert "flash does not appear to have fired" in d.evidence["not_run_reasons"]["retroreflection"]


# ------------------------------------------------------------------------------ roles
def test_primary_is_the_first_primary_frame_else_the_first_frame():
    off, on = flash_pair("real_hsrp")
    d = assess_bundle([Frame(on, "flash_on", name="flash"), Frame(off, "primary", roi=ROI, name="main")], hsrp_expected=True)
    assert d.evidence["primary_frame"] == "main"
    assert d.evidence["active"]["retroreflection"]["verdict"] == "retroreflective"
    d2 = assess_bundle([Frame(off, "other", name="x"), Frame(on, "other", name="y")])
    assert d2.evidence["primary_frame"] == "x"


def test_unknown_role_and_empty_bundle_are_errors():
    off, _ = flash_pair("screen")
    with pytest.raises(ValueError, match="role"):
        assess_bundle([Frame(off, "selfie")])
    with pytest.raises(ValueError, match="at least one"):
        assess_bundle([])


def test_extra_flash_frames_are_noted_not_silently_used():
    off, on = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(on, "flash_on", name="f1"), Frame(on, "flash_on", name="f2")],
                      hsrp_expected=True)
    assert any("2 flash-on frames" in r and "f1" in r for r in d.reasons)


def test_per_frame_report_and_json_roundtrip():
    d = assess_bundle(bundle("real_hsrp"), hsrp_expected=True)
    assert [e["name"] for e in d.per_frame] == ["front", "front_flash", "left"]
    assert d.per_frame[0]["roi"] == list(ROI) and d.per_frame[1]["roi"] == list(ROI)   # flash falls back to primary's ROI
    assert d.per_frame[2]["roi"] is None
    back = json.loads(json.dumps(d.as_dict()))
    assert back["liveness"] == d.liveness and back["status"] == d.status


# --------------------------------------------------------------------------- provenance in the bundle
def test_hard_provenance_flag_in_any_frame_rejects_and_is_tagged():
    off, _ = flash_pair("real_hsrp")
    ai = jpeg(off, **{"0x0131": "Stable Diffusion XL"})
    clean = jpeg(off, **{"0x010F": "Samsung", "0x0110": "SM-A546E"})
    fr = [Frame(off, "primary", payload=clean, roi=ROI, name="main"),
          Frame(angle_frame(off, True), "angle", payload=ai, name="side")]
    d = assess_bundle(fr)
    assert d.status == "reject" and d.liveness == "suspected_spoof"
    hit = [r for r in d.reasons if "ai_software_exif" in r]
    assert hit and "[side]" in hit[0] and "[main]" not in hit[0]
    assert d.per_frame[1]["provenance"]["hard"] == ["ai_software_exif"] and d.per_frame[0]["provenance"]["hard"] == []


def test_same_finding_in_two_frames_is_reported_once_with_both_names():
    off, _ = flash_pair("real_hsrp")
    ai = jpeg(off, **{"0x0131": "Midjourney"})
    d = assess_bundle([Frame(off, "primary", payload=ai, name="a"), Frame(off, "other", payload=ai, name="b")])
    hit = [r for r in d.reasons if "ai_software_exif" in r]
    assert len(hit) == 1 and "[a, b]" in hit[0]


def test_undecodable_payload_rejects_but_is_not_a_spoof_claim():
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", payload=b"not an image", roi=ROI)])
    assert d.status == "reject" and d.liveness == "not_established"


def test_failed_attestation_rejects():
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI)], attestation={"verified": False, "nonce_ok": False})
    assert d.status == "reject" and d.liveness == "not_established"
    assert any("attestation failed" in r for r in d.reasons)


def test_verified_attestation_raises_assurance_but_not_liveness():
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI)], attestation={"verified": True, "nonce_ok": True})
    assert d.assurance == "attested" and d.liveness == "not_established"


# ------------------------------------------------------------------------ consistency across frames
@pytest.mark.parametrize("a,b,ok", [
    ("MH12AB1234", "MH12AB1234", True),
    ("MH12AB1234", "MH12??1234", True),            # '?' matches anything
    ("MH12AB?234", "MH12AB234", True),             # ...and may stand for a character the other read lacks
    ("mh 12 ab 1234", "MH12AB1234", True),         # case and spaces are not differences
    ("MH12AB1234", "MH12AB1284", False),           # a known character changed
    ("MH12AB1234", "MH12AB234", False),            # a known character dropped
    ("MH12AB1234", "MH12AB12345", False),          # a known character added
    ("??????????", "KA01XY9999", True),            # an all-unknown read contradicts nothing
])
def test_read_conflicts(a, b, ok):
    assert (read_conflicts([a, b]) == []) is ok


def test_read_conflicts_skips_empty_reads_and_names_frames():
    assert read_conflicts([None, "", "MH12AB1234"]) == []
    out = read_conflicts(["MH12AB1234", None, "MH12AB1284"], ["front", "flash", "left"])
    assert len(out) == 1 and "(front)" in out[0] and "(left)" in out[0]


def test_bundle_flags_text_that_differs_between_frames():
    fr = bundle("real_hsrp")
    d = assess_bundle(fr, hsrp_expected=True, readings=["MH12AB1234", None, "MH12AB1284"])
    assert d.status == "review" and any("plate text differs" in r and "left" in r for r in d.reasons)
    assert d.evidence["read_conflicts"]
    ok = assess_bundle(fr, hsrp_expected=True, readings=["MH12AB1234", "MH12??1234", "MH12AB1234"])
    assert ok.evidence["read_conflicts"] == [] and not any("plate text differs" in r for r in ok.reasons)


# ----------------------------------------------------------------------- PadNet is optional
def test_missing_recapture_model_is_reported_not_an_error(monkeypatch):
    monkeypatch.delenv("VINPLATE_MODELS", raising=False)
    monkeypatch.setattr(runtime, "MODEL_DIRS", ())
    auth = Authenticator.from_default()
    assert auth.pad is None and "not found" in auth.pad_error
    off, _ = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI)], authenticator=auth)
    assert "recapture" in d.not_run and "not found" in d.evidence["not_run_reasons"]["recapture"]


def test_corrupt_recapture_model_is_reported_not_an_error(monkeypatch, tmp_path):
    (tmp_path / "pad_int8.onnx").write_bytes(b"this is not an onnx file")
    monkeypatch.setenv("VINPLATE_MODELS", str(tmp_path))
    auth = Authenticator.from_default()
    assert auth.pad is None and "could not be loaded" in auth.pad_error


needs_pad = pytest.mark.skipif(not any((d / "pad_int8.onnx").is_file() for d in runtime.MODEL_DIRS),
                               reason="shipped PadNet model not present")
needs_photos = pytest.mark.skipif(not PHOTOS, reason="Open Images photos not present (ml/data is not in version control)")


@needs_pad
def test_shipped_padnet_loads_and_runs_in_a_bundle():
    auth = Authenticator.from_default()
    assert auth.pad is not None and auth.pad_error is None
    off, on = flash_pair("real_hsrp")
    d = assess_bundle([Frame(off, "primary", roi=ROI), Frame(on, "flash_on")], authenticator=auth, hsrp_expected=True)
    s = d.evidence["recapture_by_frame"]
    assert set(s) == {"frame0", "frame1"} and all(0.0 <= v <= 1.0 for v in s.values())
    assert "recapture" not in d.not_run


# ------------------------------------------------------------ simulated recapture of a REAL photo
def _roi_with_headroom(rgb, size=(220, 110)):
    """A plate-sized window whose linear luminance leaves room for a flash test (aim ~0.03-0.3)."""
    lin = (rgb.astype(np.float32) / 255.0) ** 2.2
    lum = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    h, w = lum.shape
    sw, sh = size
    for y in range(40, h - sh - 40, 20):
        for x in range(40, w - sw - 40, 20):
            if 0.05 < np.median(lum[y : y + sh, x : x + sw]) < 0.25:
                return (x, y, x + sw, y + sh)
    return None


@needs_photos
def test_screen_replay_of_a_real_photo_never_reaches_confirmed():
    from synth.recapture import screen_recapture

    rng = np.random.default_rng(3)
    tried = 0
    for p in PHOTOS[:25]:
        src = np.asarray(Image.open(p).convert("RGB"))
        if min(src.shape[:2]) < 300:
            continue
        shot, _ = screen_recapture(src, rng)
        roi = _roi_with_headroom(shot)
        if roi is None:
            continue
        tried += 1
        lin = (shot.astype(np.float32) / 255.0) ** 2.2
        veil = 0.015 * (0.9 + 0.1 * rng.random(lin.shape[:2]))[..., None]                # emissive: flash adds a faint veil
        on = (255 * np.clip(lin + veil, 0, 1) ** (1 / 2.2)).astype(np.uint8)
        angle = np.stack([warp(shot[..., 1], (7, 3), 1.0)] * 3, -1).astype(np.uint8)     # one flat plane
        d = assess_bundle([Frame(shot, "primary", roi=roi), Frame(on, "flash_on"), Frame(angle, "angle")], hsrp_expected=True)
        assert d.liveness != "confirmed", (p.name, d.reasons)
        retro = d.evidence["active"].get("retroreflection")
        if retro:   # may legitimately decline to judge (e.g. a near-black surround makes any veil a huge relative gain)
            assert retro["verdict"] in ("not_retroreflective", "inconclusive"), retro
        if tried == 2:
            break
    assert tried, "no photo had a usable plate-sized window"


@needs_photos
@needs_pad
def test_padnet_scores_recaptured_and_bona_fide_crops_in_range():
    from synth.recapture import bona_fide, screen_recapture

    rng = np.random.default_rng(5)
    auth = Authenticator.from_default()
    src = np.asarray(Image.open(PHOTOS[0]).convert("RGB"))
    assert min(src.shape[:2]) >= 200
    for shot in (screen_recapture(src, rng)[0], bona_fide(src, rng)[0]):
        score, note = auth.recapture_score(shot, None)
        assert note is None and 0.0 <= score <= 1.0                     # runs and returns a probability; accuracy NOT asserted


# ------------------------------------------------------------------ screen context + flash hot-spot (real attack)
ATTACK_CONTEXT = {"score": 3.0, "url": True, "ui_words": ["DETAILS", "IMAGES", "RELATED", "SEARCH"], "text_lines": 31,
                  "reasons": ["a web address (URL) is visible in the scene", "browser / app interface words are visible: DETAILS, IMAGES, RELATED, SEARCH",
                              "31 separate lines of text are visible, far more than a vehicle scene normally has"]}


def real_attack_frames():
    """The ingredients of the user's real attack: the flash reflected as a glare patch on glass, a left photo
    with depth (the monitor stands in front of an office), PadNet saying 'genuine'."""
    frames = hotspot_bundle()
    frames.append(Frame(angle_frame(frames[0].rgb, layered=True), "angle", name="left"))
    return frames


def test_real_attack_ingredients_are_never_confirmed_even_without_the_screen_text():
    """Glare patch + depth + a 'clean' PadNet score: the old policy confirmed this via parallax + PadNet."""
    auth = Authenticator(pad_model=lambda patches: 0.003)
    d = assess_bundle(real_attack_frames(), authenticator=auth, hsrp_expected=True)
    assert d.liveness != "confirmed" and d.evidence["liveness_basis"] == []
    assert "parallax_non_planar" in d.positive and d.evidence["recapture_by_frame"]["front"] == pytest.approx(0.003)
    assert d.evidence["active"]["retroreflection"]["verdict"] == "specular_hotspot"
    assert d.liveness == "suspected_spoof"                                     # the hot-spot alone already counts against it


def test_screen_photo_with_parallax_and_clean_padnet_is_not_confirmed_and_is_a_suspected_spoof():
    """Same, plus the browser UI the OCR read: flagged outright (score >= 3)."""
    auth = Authenticator(pad_model=lambda patches: 0.003)
    d = assess_bundle(real_attack_frames(), authenticator=auth, hsrp_expected=True, screen_context=ATTACK_CONTEXT)
    assert d.liveness == "suspected_spoof" and d.status == "reject"
    assert d.pa_score >= 4.0 and d.evidence["liveness_basis"] == []              # 3.0 context + 1.0 hot-spot
    assert d.evidence["screen_context"]["score"] == 3.0
    assert any("screen / interface context" in r for r in d.reasons)


def test_a_single_url_is_only_a_review_not_a_reject():
    """A dealer sticker with a web address must not condemn a genuine capture: URL alone weighs 1.0 -> review."""
    ctx = {"score": 1.0, "url": True, "ui_words": [], "text_lines": 3, "reasons": ["a web address (URL) is visible in the scene"]}
    d = assess_bundle(bundle("real_hsrp"), hsrp_expected=True, screen_context=ctx)
    assert d.status == "review" and d.liveness == "suspected_spoof" and d.pa_score == pytest.approx(1.0)


def test_weak_context_below_the_spoof_threshold_changes_nothing_else():
    ctx = {"score": 0.5, "url": False, "ui_words": [], "text_lines": 15, "reasons": ["15 separate lines of text ..."]}
    d = assess_bundle(bundle("real_hsrp"), hsrp_expected=True, screen_context=ctx)
    assert d.liveness == "confirmed" and d.status != "reject"           # genuine HSRP flash pair, 0.5 is only a hint
    assert d.pa_score == pytest.approx(0.5)


def hotspot_bundle(seed=0):
    """Screen replay with the flash reflected as a glare patch over the right quarter of the plate."""
    off, on = flash_pair("screen", seed)
    lin = (on[..., 0].astype(np.float32) / 255.0) ** 2.2
    xx = np.arange(W)[None, :].repeat(H, 0)
    x0 = ROI[0] + 0.75 * (ROI[2] - ROI[0])
    rows = 0.7 + 0.3 * np.exp(-((np.arange(H)[:, None] - (ROI[1] + ROI[3]) / 2) / 40.0) ** 2)
    on2 = to_u8(np.clip(lin + 0.9 * (1 / (1 + np.exp(-(xx - x0) / 6.0))) * rows, 0, 1))
    return [Frame(off, "primary", roi=ROI, name="front"), Frame(on2, "flash_on", name="front_flash")]


def test_specular_hotspot_under_flash_is_a_spoof_signal_not_confirmation():
    d = assess_bundle(hotspot_bundle(), hsrp_expected=True)
    assert d.evidence["active"]["retroreflection"]["verdict"] == "specular_hotspot"
    assert d.evidence["active"]["retroreflection"]["hotspot"] is True
    assert d.liveness == "suspected_spoof" and d.pa_score >= 1.0
    assert "retroreflection_passed" not in d.positive
    assert any("specular hot-spot" in r for r in d.reasons)


def test_confirmed_requires_a_uniform_retroreflective_gain_and_nothing_else():
    d = assess_bundle(bundle("real_hsrp"), hsrp_expected=True)
    assert d.liveness == "confirmed" and d.evidence["liveness_basis"] == ["retroreflection_passed"]
    assert d.evidence["active"]["retroreflection"]["gain_spread"] <= 2.5
