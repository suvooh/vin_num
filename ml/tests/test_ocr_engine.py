"""PP-OCR engine: geometry (no model needed) and real detect / deskew / read on rendered plates
(skipped when the ONNX models are not present)."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from vinplate.ocr.ppocr import _homography_coeffs, _vocab_matrix, min_area_rect, warp_quad
from vinplate.runtime import REPO_ROOT, find_model
from vinplate.vocab import BLANK, CHARS, UNK, VOCAB_SIZE


# ------------------------------------------------------------------------------- geometry
def _rotated_rect_points(cx, cy, w, h, deg, n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = np.c_[rng.uniform(-w / 2, w / 2, n), rng.uniform(-h / 2, h / 2, n)]
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return p @ np.array([[c, s], [-s, c]]) + [cx, cy]


@pytest.mark.parametrize("deg", [0, 7, -12, 25])
def test_min_area_rect_recovers_size_and_order(deg):
    q, long_, short = min_area_rect(_rotated_rect_points(200, 100, 120, 30, deg))
    assert long_ == pytest.approx(120, rel=0.04) and short == pytest.approx(30, rel=0.08)
    top = q[1] - q[0]                                               # TL -> TR runs along the long side
    assert np.degrees(np.arctan2(top[1], top[0])) == pytest.approx(deg, abs=2.5)
    assert q[3][1] > q[0][1]                                        # BL is below TL: rows read top-down


def test_min_area_rect_handles_collinear_points():
    q, long_, short = min_area_rect(np.c_[np.arange(30.0), np.zeros(30)])
    assert long_ == pytest.approx(29) and short == pytest.approx(0, abs=1e-6)


def test_homography_maps_the_corners():
    dst = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], float)
    src = np.array([[10, 12], [112, 22], [105, 70], [3, 58]], float)
    a, b, c, d, e, f, g, h = _homography_coeffs(dst, src)
    for (x, y), (u, v) in zip(dst, src):
        den = g * x + h * y + 1
        assert (a * x + b * y + c) / den == pytest.approx(u, abs=1e-6)
        assert (d * x + e * y + f) / den == pytest.approx(v, abs=1e-6)


def test_warp_quad_straightens_a_tilted_plate():
    img = Image.new("RGB", (400, 300), (90, 90, 90))
    plate = Image.new("RGB", (200, 50), (240, 240, 240))
    ImageDraw.Draw(plate).rectangle((0, 0, 199, 6), fill=(0, 0, 0))             # black stripe along the TOP edge
    img.paste(plate.rotate(15, expand=True, fillcolor=(90, 90, 90), resample=Image.BICUBIC), (100, 80))
    rgb = np.asarray(img)
    q, _, _ = min_area_rect(np.argwhere((rgb != 90).any(axis=2))[:, ::-1])       # everything that is not background
    crop = warp_quad(rgb, q)
    h, w = crop.shape[:2]
    assert w / h == pytest.approx(4.0, rel=0.12)
    g = crop.mean(axis=2)
    assert g[: h // 5].mean() < g[h // 2 :].mean() - 60                         # the stripe is at the top, upright


def test_warp_quad_rotates_vertical_text_upright():
    rgb = np.zeros((300, 300, 3), np.uint8)
    quad = np.array([[100, 50], [140, 50], [140, 250], [100, 250]], float)       # tall: 40 x 200
    crop = warp_quad(rgb, quad)
    assert crop.shape[1] > crop.shape[0]


# ---------------------------------------------------------------------------- vocabulary map
class _Chars:
    chars = ["", "A", "b", "0", "l", "中", "-", "Q", " "]


def test_vocab_matrix_projects_onto_the_plate_alphabet():
    m = _vocab_matrix(_Chars.chars)
    assert m.shape == (len(_Chars.chars), VOCAB_SIZE) and np.allclose(m.sum(axis=1), 1.0)
    cls = lambda c: _Chars.chars.index(c)
    ix = lambda c: CHARS.index(c) + 1
    assert m[cls(""), BLANK] == 1 and m[cls(" "), BLANK] == 1                    # blank and space are CTC blank
    assert m[cls("b"), ix("B")] == 1                                             # case folds
    assert m[cls("中"), ix(UNK)] == 1 and m[cls("-"), ix(UNK)] == 1              # CJK / punctuation = unreadable
    assert m[cls("l"), ix("L")] + m[cls("l"), ix("1")] + m[cls("l"), ix("I")] == pytest.approx(1)
    assert m[cls("l"), ix("1")] > 0.3                                            # a lowercase l may be a 1


# -------------------------------------------------------------------------- real engine
def _models_present() -> bool:
    try:
        for n in ("ocr_det.onnx", "ocr_rec.onnx", "ocr_charset.txt"):
            find_model(n)
        return True
    except FileNotFoundError:
        return False


needs_models = pytest.mark.skipif(not _models_present(), reason="PP-OCR ONNX models not found (app/assets/models)")


@pytest.fixture(scope="module")
def engine():
    from vinplate.ocr import PpOcrEngine
    return PpOcrEngine()


@pytest.fixture(scope="module")
def font():
    fonts = sorted((REPO_ROOT / "ml" / "synth" / "fonts" / "barlowcondensed").glob("*Bold.ttf"))
    if not fonts:
        pytest.skip("no font available")
    return str(fonts[0])


def _scene(font, rows, angle=0.0, size=(1000, 700)):
    """A grey 'car' with a light plate carrying one or two rows of text, optionally rotated."""
    h = 120 if len(rows) == 1 else 230
    plate = Image.new("RGB", (560, h), (242, 242, 238))
    d = ImageDraw.Draw(plate)
    d.rounded_rectangle((2, 2, 557, h - 3), 10, outline=(20, 20, 20), width=5)
    for k, t in enumerate(rows):
        d.text((280, h * (k + 0.5) / len(rows)), t, font=ImageFont.truetype(font, 80 if len(rows) == 1 else 82), fill=(15, 15, 15), anchor="mm")
    img = Image.new("RGB", size, (110, 114, 120))
    pl = plate.rotate(angle, expand=True, fillcolor=(110, 114, 120), resample=Image.BICUBIC)
    img.paste(pl, ((size[0] - pl.width) // 2, (size[1] - pl.height) // 2))
    return np.asarray(img)


@needs_models
def test_detector_deskews_a_rotated_plate(engine, font):
    rgb = _scene(font, ["MH 12 AB 1234"], angle=12)
    boxes = engine.detect(rgb)
    assert boxes, "no text detected"
    big = max(boxes, key=lambda b: b.width * b.height)
    assert big.angle == pytest.approx(-12, abs=4)                    # PIL rotates counter-clockwise
    from vinplate.decode import greedy
    text = greedy(engine.rec_logp(warp_quad(rgb, big.quad)))[0]
    assert text == "MH12AB1234", text


@needs_models
@pytest.mark.parametrize("angle", [0, 10, -8])
def test_scene_reader_reads_one_row_plates(engine, font, angle):
    from vinplate.scene import SceneReader
    res = SceneReader(engine).read(_scene(font, ["KA 05 MN 4821"], angle), domains=("plate",))
    assert res.plate is not None and res.plate.reading.text == "KA05MN4821", (res.candidates, res.lines)
    assert res.plate.status == "accept"


@needs_models
def test_scene_reader_reads_a_two_row_plate(engine, font):
    from vinplate.scene import SceneReader
    res = SceneReader(engine).read(_scene(font, ["MH 12", "AB 1234"]), domains=("plate",))
    assert res.plate is not None and res.plate.reading.text == "MH12AB1234", (res.candidates, res.lines)


@needs_models
def test_scene_reader_does_not_report_a_plate_in_an_empty_scene(engine):
    from vinplate.scene import SceneReader
    rng = np.random.default_rng(3)
    res = SceneReader(engine).read((rng.random((480, 640, 3)) * 255).astype(np.uint8))
    assert res.plate is None and res.vin is None


@needs_models
def test_read_lines_is_rapidocr_shaped(engine, font):
    lines = engine.read_lines(_scene(font, ["TN 71 JE 5250"]))
    assert lines and {"text", "confidence", "cy", "x", "x1", "y0", "h"} <= set(lines[0])


def test_fast_projection_equals_the_dense_matrix_product():
    """to_logp gathers the few meaningful classes and takes '?' as the remainder; it must agree with
    the plain probs @ matrix projection."""
    from vinplate.ocr.ppocr import PpOcrEngine
    chars = [""] + [chr(c) for c in range(33, 120)] + ["中", "文", "é", " "]
    eng = PpOcrEngine.__new__(PpOcrEngine)
    eng.chars = chars
    eng._vmat = _vocab_matrix(chars)
    eng._unk = CHARS.index(UNK) + 1
    eng._keep = np.nonzero(eng._vmat[:, eng._unk] < 1.0)[0]
    eng._vsmall = np.ascontiguousarray(eng._vmat[eng._keep])
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.full(len(chars), 0.3), size=40).astype(np.float32)
    dense = np.clip(probs @ eng._vmat, 1e-9, None)
    dense = np.log(dense / dense.sum(axis=1, keepdims=True))
    assert np.allclose(eng.to_logp(probs), dense, atol=2e-3)
    t2 = eng.to_logp(probs, temperature=2.0)
    assert np.allclose(np.exp(t2).sum(axis=1), 1.0, atol=1e-6)
