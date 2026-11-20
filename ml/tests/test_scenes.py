"""Tests for the synthetic scene benchmark (eval/scenes.py, eval/e2e_eval.py).

Fast and model-free: scene generation (a handful of scenes), the scoring rules, aggregation and
the oracle ``fake`` reader. Scene tests that need the Open Images photos skip when the data
folder is absent; the VIN-scene and scoring tests never need it.
"""

import re

import numpy as np
import pytest
from PIL import Image

from eval import e2e_eval as ev
from eval.scenes import MAX_SIDE, Scene, load_backgrounds, make_plate_scene, make_scenes, make_vin_scene


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def backgrounds():
    bgs = load_backgrounds()
    if not bgs:
        pytest.skip("Open Images backgrounds not present (ml/data/openimages)")
    return bgs


def _plate(bgs, seed):
    path, boxes = bgs[0]
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return make_plate_scene(np.random.default_rng(seed), rgb, boxes[0])


@pytest.fixture(scope="module")
def plate_scene(backgrounds):
    return _plate(backgrounds, 11)


@pytest.fixture(scope="module")
def vin_scene():
    return make_vin_scene(np.random.default_rng(21))


# --------------------------------------------------------------------------------- scenes
def _check_scene(sc: Scene, domain: str):
    assert sc.domain == domain
    assert sc.img.dtype == np.uint8 and sc.img.ndim == 3 and sc.img.shape[2] == 3
    assert max(sc.img.shape[:2]) <= MAX_SIDE
    assert sc.profile in ("clean", "mild", "moderate", "severe")
    assert re.fullmatch(r"[0-9A-Z?]+", sc.truth)
    assert sc.quad.shape == (4, 2)
    h, w = sc.img.shape[:2]
    x0, y0, x1, y1 = sc.bbox
    assert -3 <= x0 < x1 <= w + 3 and -3 <= y0 < y1 <= h + 3, "region must lie in the frame"
    assert (x1 - x0) >= 20 and (y1 - y0) >= 8
    # corner order TL TR BR BL: the top edge is above the bottom edge, left is left of right
    assert sc.quad[0, 1] < sc.quad[3, 1] and sc.quad[1, 1] < sc.quad[2, 1]
    assert sc.quad[0, 0] < sc.quad[1, 0] and sc.quad[3, 0] < sc.quad[2, 0]


def test_plate_scene_is_sane(plate_scene):
    _check_scene(plate_scene, "plate")
    assert 4 <= len(plate_scene.truth) <= 12
    assert 140 <= plate_scene.meta["plate_w_px"] <= 600     # target 150-420, 2-row / jitter widen it a little


def test_vin_scene_is_sane(vin_scene):
    _check_scene(vin_scene, "vin")
    # the "VIN" / "CHASSIS NO" label prefix is not part of the value to read
    assert 6 <= len(vin_scene.truth) <= 17
    assert 10 <= vin_scene.meta["cap_px"] <= 70


def test_vin_scene_is_deterministic(vin_scene):
    again = make_vin_scene(np.random.default_rng(21))
    assert np.array_equal(again.img, vin_scene.img)
    assert again.truth == vin_scene.truth and np.allclose(again.quad, vin_scene.quad)
    other = make_vin_scene(np.random.default_rng(22))
    assert not (other.img.shape == vin_scene.img.shape and np.array_equal(other.img, vin_scene.img))


def test_plate_scene_is_deterministic(backgrounds, plate_scene):
    again = _plate(backgrounds, 11)
    assert np.array_equal(again.img, plate_scene.img)
    assert again.truth == plate_scene.truth and np.allclose(again.quad, plate_scene.quad)


def test_plate_scene_changes_only_near_the_plate(backgrounds, plate_scene):
    """The photo outside the (painted-out + pasted) plate neighbourhood is the real photo."""
    path, boxes = backgrounds[0]
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    s = plate_scene.meta["bg_scale"]
    ref = np.asarray(rgb.resize(plate_scene.img.shape[1::-1], Image.LANCZOS if s >= 1 else Image.BICUBIC), np.float32)
    diff = np.abs(plate_scene.img.astype(np.float32) - ref).mean(axis=2)
    x0, y0, x1, y1 = plate_scene.bbox
    mx, my = 0.6 * (x1 - x0), 1.0 * (y1 - y0)
    far = np.ones(diff.shape, bool)
    far[max(0, int(y0 - my)): int(y1 + my), max(0, int(x0 - mx)): int(x1 + mx)] = False
    near_mean = diff[int(y0): int(y1), int(x0): int(x1)].mean()
    # scene-level JPEG / blur and removal of other plates allow small drift far away, but the
    # plate region itself must have been replaced
    assert near_mean > 10 * max(diff[far].mean(), 0.5) or near_mean > 15


def test_make_scenes_deterministic_and_mixed(backgrounds):
    a = list(make_scenes(3, seed=7, plate_frac=0.5, backgrounds=backgrounds))
    b = list(make_scenes(3, seed=7, plate_frac=0.5, backgrounds=backgrounds))
    assert len(a) == len(b) == 3
    for x, y in zip(a, b):
        assert x.domain == y.domain and x.truth == y.truth and np.array_equal(x.img, y.img)


def test_make_scenes_without_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        next(make_scenes(1, seed=0, plate_frac=1.0, root=tmp_path))


# --------------------------------------------------------------------------------- scoring
def S(truth, text, masked, status):
    return ev.score_reading("plate", truth, text, masked, status)


def test_exact_accepted():
    s = S("MH12AB1234", "MH12AB1234", "MH12AB1234", "accept")
    assert s["exact"] and s["faithful"] and not s["accepted_wrong"] and s["asserted_wrong_chars"] == 0


def test_honest_unknown_on_a_hidden_character_is_faithful_not_exact():
    truth = "MH12AB?234"                                     # synthetic damage hid the 7th char
    s = S(truth, "MH12AB0234", "MH12AB?234", "review")
    assert s["faithful"] and not s["exact"] and s["asserted_wrong_chars"] == 0
    assert not s["accepted_with_hidden"] and not s["accepted_wrong"]


def test_accept_with_a_hidden_truth_is_reported_separately_not_as_wrong():
    s = S("MH12AB?234", "MH12AB?234", "MH12AB?234", "accept")
    assert s["accepted_with_hidden"] and not s["accepted_wrong"] and not s["exact"]
    # ...but if it filled the hidden character with a letter it hallucinated:
    h = S("MH12AB?234", "MH12AB0234", "MH12AB0234", "accept")
    assert h["asserted_wrong_chars"] == 1 and not h["faithful"] and h["accepted_with_hidden"]
    assert not h["accepted_wrong"]


def test_abstaining_on_a_readable_character_still_faithful():
    s = S("MH12AB1234", "MH12AB1234", "MH12AB?234", "review")
    assert s["faithful"] and s["exact"] and s["shown_chars"] == 9 and s["shown_correct"] == 9


def test_wrong_asserted_character():
    s = S("MH12AB1234", "MH12AB1235", "MH12AB1235", "review")
    assert s["asserted_wrong_chars"] == 1 and not s["faithful"] and not s["exact"]
    assert not s["accepted_wrong"]                           # only 'accept' can be accepted-wrong


def test_accepted_wrong_is_the_safety_metric():
    s = S("MH12AB1234", "MH12AB1235", "MH12AB1235", "accept")
    assert s["accepted_wrong"] and not s["exact"] and not s["faithful"]


def test_length_mismatch_counts_every_shown_character_wrong():
    s = S("MH12AB1234", "MH12AB1234", "?MH12AB1234", "review")     # inserted '?' for a "dropped" char
    assert s["length_mismatch"] and s["asserted_wrong_chars"] == 10 and not s["faithful"]
    assert s["masked_len_delta"] == 1 and s["exact"]          # text itself was right
    short = S("MH12AB1234", "MH12AB123", "MH12AB123", "review")
    assert short["length_mismatch"] and short["asserted_wrong_chars"] == 9


def test_reject_and_none_assert_nothing():
    for status, text in (("reject", "XX"), ("none", "")):
        s = S("MH12AB1234", text, text, status)
        assert not s["found"] and s["asserted_wrong_chars"] == 0 and not s["faithful"]
        assert not s["accepted_wrong"] and not s["exact"]


def test_wilson_interval():
    lo, hi = ev.wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    lo, hi = ev.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert ev.wilson(0, 0) == (0.0, 1.0)


def test_located():
    quad_box = (100, 100, 300, 160)
    assert ev.located((110, 105, 290, 150), quad_box) is True
    assert ev.located((120, 110, 180, 150), quad_box) is True       # a sub-region inside it
    assert ev.located((500, 400, 700, 460), quad_box) is False
    assert ev.located((0, 0, 1600, 1200), quad_box) is False        # the whole frame is not a location
    assert ev.located(None, quad_box) is None


# ------------------------------------------------------------------------------ aggregation
def row(reader, truth, text, masked, status, tags=(), profile="mild", domain="plate", **kw):
    return dict(reader=reader, truth=truth, text=text, masked=masked, status=status, tags=list(tags),
                profile=profile, domain=domain, **kw)


def test_summarise_hand_made_rows():
    rows = [
        row("x", "MH12AB1234", "MH12AB1234", "MH12AB1234", "accept", ["glare"], ms=10, located=True),
        row("x", "KA01AB0001", "KA01AB0007", "KA01AB0007", "accept", ["glare", "blur"], ms=30, located=False),
        row("x", "DL1CAB?234", "DL1CAB0234", "DL1CAB?234", "review", ["dirt"], "severe", ms=20, located=True),
        row("x", "GJ05CD4321", "", "", "none", [], "clean", "vin", ms=40),
        row("y", "MH12AB1234", "MH12AB1234", "MH12AB1234", "accept", ["glare"]),
    ]
    sm = ev.summarise(rows)
    a = sm["x"]["all"]
    assert a["n"] == 4 and a["n_readable"] == 3
    assert (a["accept"], a["review"], a["none"]) == (0.5, 0.25, 0.25)
    assert a["exact"] == pytest.approx(1 / 3)                 # over the 3 readable scenes
    assert a["faithful"] == pytest.approx(0.5)                # first and third; the wrong digit and the miss are not
    assert a["accepted_wrong"] == pytest.approx(1 / 3) and a["accepted_wrong_n"] == 1
    lo, hi = a["accepted_wrong_ci95"]
    assert lo < 1 / 3 < hi
    assert a["accept_precision"] == pytest.approx(0.5)        # 2 accepted readable scenes, 1 wrong
    assert a["wrong_chars_per_scene"] == pytest.approx(0.25) and a["wrong_chars_per_found"] == pytest.approx(1 / 3)
    assert a["located"] == pytest.approx(2 / 3) and a["ms_median"] == 25
    g = sm["x"]["tag=glare"]                                  # a scene counts in every tag it carries
    assert g["n"] == 2 and g["accepted_wrong_n"] == 1
    assert sm["x"]["tag=blur"]["n"] == 1 and sm["x"]["tag=dirt"]["n"] == 1
    assert sm["x"]["domain=vin"]["n"] == 1 and sm["x"]["domain=plate"]["n"] == 3
    assert sm["x"]["profile=severe"]["n"] == 1
    assert sm["y"]["all"]["exact"] == 1.0 and sm["y"]["all"]["located"] is None
    table = ev.format_table(sm)
    assert table.count("\n") >= 6 and "| all | x | 4 |" in table and "| all | y | 1 |" in table


def test_summarise_slice_with_only_hidden_truths_has_no_exact_rate():
    sm = ev.summarise([row("x", "MH12?B1234", "MH12AB1234", "MH12?B1234", "review", ["glare"])])
    s = sm["x"]["tag=glare"]
    assert s["n_readable"] == 0 and s["exact"] is None and s["accepted_wrong"] is None and s["faithful"] == 1.0


# ---------------------------------------------------------------------------------- readers
def test_fake_oracle_reader_is_perfect(vin_scene, plate_scene):
    scenes = [plate_scene, vin_scene]
    rows = ev.run({"fake": ev._make_fake()}, scenes)
    assert len(rows) == 2 and all(r["located"] is True for r in rows)
    s = ev.summarise(rows)["fake"]["all"]
    assert s["faithful"] == 1.0 and s["accepted_wrong"] in (0.0, None) and s["wrong_chars_per_scene"] == 0.0
    if s["n_readable"]:
        assert s["exact"] == 1.0


def test_run_records_reader_errors_as_misses(vin_scene):
    def boom(rgb, domain):
        raise RuntimeError("model exploded")

    rows = ev.run({"boom": boom}, [vin_scene])
    assert rows[0]["status"] == "none" and "model exploded" in rows[0]["error"]
    assert ev.summarise(rows)["boom"]["all"]["none"] == 1.0


def test_unavailable_readers_are_skipped_not_fatal(monkeypatch, capsys):
    def missing():
        raise ImportError("no module named vinplate.scene")

    monkeypatch.setitem(ev.READER_FACTORIES, "ghost", missing)
    readers = ev.build_readers(["fake", "ghost", "nope"])
    assert list(readers) == ["fake"]
    err = capsys.readouterr().err
    assert "ghost" in err and "nope" in err


def test_vin_label_is_stripped_from_new_reader_output():
    assert ev._strip_vin_label("VINMD2A3A1J9A2834023", "VINMD2A3A1J9A28?4023") == ("MD2A3A1J9A2834023", "MD2A3A1J9A28?4023")
    assert ev._strip_vin_label("CHASSISNO12345678", "CHASSISNO12345678") == ("12345678", "12345678")
    assert ev._strip_vin_label("MD2A3A1J9A2834023", "MD2A3A1J9A2834023")[0] == "MD2A3A1J9A2834023"
