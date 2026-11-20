from pathlib import Path

import numpy as np
import pytest

from synth.plates import render_plate
from vinplate.decode.lines import concat_lines, trim_blank
from vinplate.geometry.rows import looks_two_row, split_rows
from vinplate.pipeline import OnnxRecognizer, Reader
from vinplate.testing import make_logp
from vinplate.vocab import BLANK, VOCAB_SIZE


class Fake:
    """Returns pre-baked posteriors in call order (what a perfect recogniser would emit)."""

    def __init__(self, *lps):
        self.q = list(lps)
        self.calls = []

    def logp(self, domain, gray):
        self.calls.append((domain, gray.shape))
        return self.q.pop(0)


def two_row_plate(seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        r = render_plate(rng, spec=dict(layout="standard", groups=["MH", "12", "AB", "1234"], text="MH12AB1234", kind="standard"))
        if len(r.rows) == 2:
            return r
    raise AssertionError("no 2-row plate drawn")


def test_row_split_lands_between_the_rows():
    r = two_row_plate()
    img = np.clip(r.img, 0, 255).astype(np.uint8)
    assert looks_two_row(*img.shape[:2])
    top, bottom = split_rows(img)
    gap_lo, gap_hi = r.row_y[0][1], r.row_y[1][0]
    cut = (img.shape[0] - bottom.shape[0] + top.shape[0]) / 2          # centre of the shared overlap
    assert abs(cut - (gap_lo + gap_hi) / 2) < 0.12 * img.shape[0]
    assert top.shape[0] > r.row_y[0][1] and bottom.shape[0] > img.shape[0] - r.row_y[1][0]   # overlap kept


def test_trim_and_concat_do_not_create_a_false_gap():
    a, b = make_logp("MH12"), make_logp("AB1234")
    t = trim_blank(a)
    assert t.shape[0] < a.shape[0] and np.exp(t[:, BLANK]).min() < 0.9
    from vinplate.decode import read_line
    from vinplate.grammar import PLATE_GRAMMAR
    r = read_line(concat_lines([a, b]), PLATE_GRAMMAR)
    assert r.text == "MH12AB1234" and r.suspected_gaps == [] and r.status == "accept"


def test_two_row_plate_is_read_row_by_row_and_joined():
    r = two_row_plate()
    img = np.clip(r.img, 0, 255).astype(np.uint8)
    fake = Fake(make_logp("MH12"), make_logp("AB1234"))
    res = Reader(fake).read_plate(img)
    assert res.rows == 2 and [c[0] for c in fake.calls] == ["plate", "plate"]
    assert res.reading.text == "MH12AB1234" and res.parsed["display"] == "MH 12 AB 1234"
    assert res.status in ("accept", "review")   # quality gate may downgrade a clean read of a poor crop


def test_one_row_plate_is_one_call():
    rng = np.random.default_rng(1)
    for _ in range(100):
        r = render_plate(rng, spec=dict(layout="standard", groups=["KA", "01", "AB", "1234"], text="KA01AB1234", kind="standard"))
        if len(r.rows) == 1:
            break
    fake = Fake(make_logp("KA01AB1234"))
    res = Reader(fake).read_plate(np.clip(r.img, 0, 255).astype(np.uint8))
    assert res.rows == 1 and len(fake.calls) == 1 and res.reading.text == "KA01AB1234"


def test_failed_quality_gate_downgrades_a_clean_reading():
    black = np.zeros((60, 240, 3), np.uint8)
    res = Reader(Fake(make_logp("KA01AB1234"))).read_plate(black, rows=1)
    assert res.reading.status == "accept" and res.status == "review" and "too dark" in res.quality.issues


def test_vin_frames_fusion_through_the_reader():
    vin = "MA3EWB22SPA123456"
    a, b = make_logp(vin, flat=(5,)), make_logp(vin, flat=(12,))
    img = np.full((60, 600, 3), 200, np.uint8)
    res = Reader(Fake(a, b), gate=False).read_vin_frames([img, img])
    assert res.reading.text == vin and res.parsed["wmi"] == "MA3" and res.status == "accept"


def test_garbage_posteriors_are_rejected_not_parsed():
    res = Reader(Fake(make_logp("QQQQQQQQQQQQQQQQQQQQQQQQ")), gate=False).read_plate(np.full((50, 220, 3), 255, np.uint8), rows=1)
    assert res.status == "reject"
    assert res.parsed is None                  # no structured fields for a rejected read


@pytest.mark.skipif(not Path("build/rec_plate_int8.onnx").exists(), reason="run export.export_onnx first")
def test_onnx_recogniser_shapes_and_provider():
    rec = OnnxRecognizer("build/rec_plate_int8.onnx", "build/rec_vin_int8.onnx", provider="cpu")
    lp = rec.logp("plate", np.random.default_rng(0).integers(0, 255, (48, 256), dtype=np.uint8))
    assert lp.shape == (64, VOCAB_SIZE) and np.allclose(np.exp(lp).sum(-1), 1, atol=1e-3)
    assert rec.logp("vin", np.zeros((48, 512), np.uint8)).shape == (128, VOCAB_SIZE)
