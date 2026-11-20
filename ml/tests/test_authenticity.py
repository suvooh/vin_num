import io
import struct

import numpy as np
import pytest
from PIL import Image, PngImagePlugin
from scipy import ndimage as ndi

from vinplate.authenticity import analyze_pixels, analyze_provenance, decide
from vinplate.authenticity.forensics import jpeg_grid, moire_score, noise_inconsistency
from vinplate.authenticity.provenance import dst_tokens


def scene(seed=0, size=256) -> np.ndarray:
    """Textured, natural-ish test image (multi-scale smooth noise + shapes + sensor noise)."""
    rng = np.random.default_rng(seed)
    g = sum(ndi.gaussian_filter(rng.standard_normal((size, size)), s) * a for s, a in ((2, 6), (8, 18), (24, 40)))
    g = 128 + g * 4
    for _ in range(14):
        x, y = rng.integers(0, size - 40, 2)
        g[y : y + rng.integers(10, 40), x : x + rng.integers(10, 40)] += rng.uniform(-60, 60)
    g = g + rng.standard_normal((size, size)) * 2.0
    rgb = np.stack([g * 1.0, g * 0.95 + 5, g * 0.9 + 10], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def jpeg_bytes(arr, q=90, **kw) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=q, **kw)
    return b.getvalue()


def insert_app1(jpeg: bytes, payload: bytes) -> bytes:
    return jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]


def codes(rep):
    return {f.code for f in rep.findings}


# ---------------------------------------------------------------- provenance
def test_clean_camera_like_jpeg_has_no_flags():
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "Samsung", "SM-A546E"
    rep = analyze_provenance(jpeg_bytes(scene(), exif=ex))
    assert not rep.hard and not rep.soft and rep.exif["make"] == "Samsung"
    assert len(rep.sha256) == 64 and len(rep.dhash) == 16


def test_missing_exif_is_information_not_a_flag():
    rep = analyze_provenance(jpeg_bytes(scene()))
    assert "no_camera_exif" in codes(rep) and not rep.hard and not rep.soft


@pytest.mark.parametrize("software,field", [("Adobe Photoshop 25.1", "soft"), ("Stable Diffusion XL", "hard")])
def test_exif_software(software, field):
    ex = Image.Exif()
    ex[0x0131] = software
    rep = analyze_provenance(jpeg_bytes(scene(), exif=ex))
    assert getattr(rep, field), rep.findings


def test_png_text_chunks_from_ai_tools_are_hard_flags():
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a plate, Steps: 20, Sampler: Euler a, CFG scale: 7")
    b = io.BytesIO()
    Image.fromarray(scene()).save(b, "PNG", pnginfo=info)
    rep = analyze_provenance(b.getvalue())
    assert "ai_png_chunk" in codes(rep) and rep.hard


XMP = (b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
       b"<rdf:Description xmlns:Iptc4xmpExt=\"http://iptc.org/std/Iptc4xmpExt/2008-02-29/\" %s/></rdf:RDF></x:xmpmeta>")


@pytest.mark.parametrize("value,hard", [
    ("trainedAlgorithmicMedia", True), ("compositeWithTrainedAlgorithmicMedia", True),
    ("screenCapture", True), ("digitalCapture", False), ("humanEdits", False),
])
def test_xmp_digital_source_type(value, hard):
    attr = f'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/{value}"'.encode()
    rep = analyze_provenance(insert_app1(jpeg_bytes(scene()), XMP % attr))
    assert bool(rep.hard) is hard, rep.findings


def test_dst_tokens_ignore_unrelated_words():
    assert dst_tokens("<x>fingerprint print</x>") == set()
    assert dst_tokens('<Iptc4xmpExt:DigitalSourceType>trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>') == {"trainedalgorithmicmedia"}


def test_garbage_c2pa_markers_are_a_hard_flag_not_trust():
    # bytes that look like C2PA but are not a valid manifest must never be read as provenance
    rep = analyze_provenance(insert_app1(jpeg_bytes(scene()), b"JUMBF....jumb....c2pa....not a manifest"))
    assert rep.c2pa["present"] and rep.c2pa["verified"] is False and "c2pa_unreadable" in codes(rep)


def test_no_c2pa_in_ordinary_jpeg():
    assert analyze_provenance(jpeg_bytes(scene())).c2pa == {"present": False, "verified": None}


def _exif_with_thumbnail(thumb_jpeg: bytes) -> bytes:
    end = "<"
    ifd0 = struct.pack(end + "H", 0) + struct.pack(end + "I", 14)            # 0 entries, next IFD at 14
    entries = struct.pack(end + "HHII", 0x0201, 4, 1, 14 + 2 + 24 + 4) + struct.pack(end + "HHII", 0x0202, 4, 1, len(thumb_jpeg))
    ifd1 = struct.pack(end + "H", 2) + entries + struct.pack(end + "I", 0)
    tiff = b"II*\x00" + struct.pack(end + "I", 8) + ifd0 + ifd1 + thumb_jpeg
    return b"Exif\x00\x00" + tiff


def test_stale_exif_thumbnail_flags_pixel_edit():
    # NB hand-built TIFF: offsets are relative to the TIFF header. IFD0 at 8 (0 entries), IFD1 at 14.
    img = scene(1)
    edited = img.copy()
    edited[60:140, 60:200] = 255 - edited[60:140, 60:200]               # the edit
    thumb_of_original = io.BytesIO()
    Image.fromarray(img).resize((64, 64)).save(thumb_of_original, "JPEG")
    seg = _exif_with_thumbnail(thumb_of_original.getvalue())
    ok_thumb = io.BytesIO()
    Image.fromarray(edited).resize((64, 64)).save(ok_thumb, "JPEG")
    bad = analyze_provenance(insert_app1(jpeg_bytes(edited), seg))
    good = analyze_provenance(insert_app1(jpeg_bytes(edited), _exif_with_thumbnail(ok_thumb.getvalue())))
    assert "thumbnail_mismatch" in codes(bad), (bad.exif, bad.findings)
    assert "thumbnail_mismatch" not in codes(good), (good.exif, good.findings)


# ----------------------------------------------------------------- forensics
def test_moire_overlay_is_detected_and_clean_is_not():
    base = scene(2)
    yy, xx = np.mgrid[:256, :256]
    grid = 14 * np.sin(2 * np.pi * xx / 3.1) * np.sin(2 * np.pi * yy / 3.4)
    screen = np.clip(base.astype(np.float32) + grid[..., None], 0, 255).astype(np.uint8)
    g = lambda a: (a[..., 1]).astype(np.float32)  # noqa: E731
    assert moire_score(g(screen)) > 2 * moire_score(g(base))
    assert "periodic_pattern" in analyze_pixels(screen).flags
    assert "periodic_pattern" not in analyze_pixels(base).flags


def test_noise_splice_is_detected():
    rng = np.random.default_rng(3)
    base = scene(3).astype(np.float32)
    clean = np.clip(base + rng.standard_normal(base.shape) * 2, 0, 255)
    spliced = clean.copy()
    spliced[96:160, 80:200] = np.clip(base[96:160, 80:200] + rng.standard_normal((64, 120, 3)) * 14, 0, 255)
    gy = lambda a: a[..., 1]  # noqa: E731
    assert noise_inconsistency(gy(spliced)) > 1.6 * noise_inconsistency(gy(clean))


def test_jpeg_grid_shift_after_crop_is_detected():
    arr = np.asarray(Image.open(io.BytesIO(jpeg_bytes(scene(4), q=80))).convert("RGB"))
    gray = lambda a: a[..., 1].astype(np.float32)  # noqa: E731
    assert jpeg_grid(gray(arr))["phase"] == (0, 0) and jpeg_grid(gray(arr))["strength"] >= 1.25
    cropped = arr[3:, 5:]
    assert jpeg_grid(gray(cropped))["phase"] == ((0 - 5) % 8, (0 - 3) % 8)
    assert "jpeg_grid_shifted" in analyze_pixels(np.ascontiguousarray(cropped)).flags


def test_resized_image_makes_grid_checks_uninformative_not_alarming():
    arr = np.asarray(Image.fromarray(scene(5)).resize((190, 190), Image.BILINEAR))
    rep = analyze_pixels(arr)
    assert "jpeg_grid_shifted" not in rep.flags and any("grid not visible" in n for n in rep.notes)


# ---------------------------------------------------------------------- fusion
def test_fusion_policy():
    clean = analyze_provenance(jpeg_bytes(scene()))
    d = decide(clean, analyze_pixels(scene()))
    assert d.status == "no_flags" and d.assurance == "none"           # "no flags" is not "authentic"
    assert set(d.not_run) == {"recapture", "tamper", "ai_generated", "retroreflection", "planarity"}   # absent checks are reported

    ex = Image.Exif(); ex[0x0131] = "Midjourney"
    assert decide(analyze_provenance(jpeg_bytes(scene(), exif=ex))).status == "reject"

    ex = Image.Exif(); ex[0x0131] = "Snapseed 2.0"
    assert decide(analyze_provenance(jpeg_bytes(scene(), exif=ex))).status == "review"

    att_ok = dict(verified=True, device_integrity=True, nonce_ok=True, hash_matches=True)
    assert decide(clean, attestation=att_ok).assurance == "attested"
    assert decide(clean, attestation={**att_ok, "hash_matches": False}).status == "reject"
    assert decide(clean, require_assurance="attested").status == "review"   # policy hook
    assert decide(clean, model_scores={"recapture": 0.6}).status == "review"


def test_undecodable_file_is_rejected():
    assert decide(analyze_provenance(b"not an image")).status == "reject"


# ------------------------------------------------- presentation-attack fusion
def _clean():
    return analyze_provenance(jpeg_bytes(scene()))


def test_one_weak_pa_cue_never_rejects_but_strong_agreement_does():
    from vinplate.authenticity.active import RetroResult
    from vinplate.authenticity.forensics import ForensicsReport

    moire = ForensicsReport(flags=["periodic_pattern"])
    d = decide(_clean(), moire)
    assert d.status == "review" and d.pa_score == 1.0          # a single cue: look, don't reject
    assert decide(_clean(), ForensicsReport(flags=["horizontal_banding"])).status == "no_flags"   # weak hint alone

    failed_flash = {"retroreflection": RetroResult("not_retroreflective")}
    assert decide(_clean(), active=failed_flash).status == "review"                      # physics alone: review
    both = decide(_clean(), moire, active=failed_flash, model_scores={"recapture": 0.95})
    assert both.status == "reject" and both.pa_score >= 3.0
    assert decide(_clean(), moire, active=failed_flash).status == "reject"              # 1.0 + 2.0


def test_retroreflection_pass_is_recorded_as_positive_evidence_only():
    from vinplate.authenticity.active import RetroResult

    d = decide(_clean(), active={"retroreflection": RetroResult("retroreflective", ratio=4.0)})
    assert d.positive == ["retroreflection_passed"] and d.assurance == "none" and d.status == "no_flags"
    assert "retroreflection" not in d.not_run


def test_inconclusive_active_checks_are_reported_not_passed():
    from vinplate.authenticity.active import PlanarityResult, RetroResult

    d = decide(_clean(), active={"retroreflection": RetroResult("inconclusive"), "planarity": PlanarityResult("inconclusive")})
    assert {"retroreflection", "planarity"} <= set(d.not_run) and d.status == "no_flags"


# ---------------------------------------------------------------- screen context + hot-spot weights (decide)
def _decide(**kw):
    from vinplate.authenticity.fusion import decide
    from vinplate.authenticity.provenance import ProvenanceReport
    return decide(ProvenanceReport(), **kw)


def test_screen_context_score_is_added_to_the_presentation_attack_score():
    ctx = lambda score: {"score": score, "reasons": ["interface text"]}
    assert _decide(context=ctx(0.5)).pa_score == pytest.approx(0.5) and _decide(context=ctx(0.5)).status == "no_flags"
    assert _decide(context=ctx(1.0)).status == "review"
    assert _decide(context=ctx(2.5)).status == "review"
    d = _decide(context=ctx(3.0))
    assert d.status == "reject" and d.pa_score == pytest.approx(3.0)
    assert d.evidence["screen_context"]["score"] == 3.0
    assert _decide().evidence["screen_context"] is None


def test_specular_hotspot_adds_one_point_and_is_not_a_positive():
    from vinplate.authenticity.active import RetroResult
    d = _decide(active={"retroreflection": RetroResult("specular_hotspot", hotspot=True)})
    assert d.pa_score == pytest.approx(1.0) and d.status == "review"
    assert "retroreflection_passed" not in d.positive
    assert "retroreflection" not in d.not_run                                  # a definite verdict, not "did not run"


def test_context_stacks_with_the_other_signals_into_a_reject():
    from vinplate.authenticity.active import RetroResult
    d = _decide(context={"score": 2.0, "reasons": []}, active={"retroreflection": RetroResult("specular_hotspot", hotspot=True)})
    assert d.pa_score == pytest.approx(3.0) and d.status == "reject"
