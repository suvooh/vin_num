"""Vehicle category from plate colours + text layout (``vinplate.category``).

Colour tests use hand-made crops with known colours; the accuracy tests use the repo's synthetic plate renderer
in every scheme (``synth.category_plates``) under several lighting variants. Those numbers are SYNTHETIC: the test
plates take their colours from the same table the rules do, so they prove the pipeline, not that real plates look
like this (see ``eval/category_eval.py`` for the full report)."""

import json

import numpy as np
import pytest

from eval.category_eval import CONFIDENT, accuracy, evaluate, summarise
from synth.category_plates import LIGHTING, SCHEMES, degrade, render_scheme
from vinplate import category as cat
from vinplate.category import (ALTERNATIVES, COLOUR_RULES, LABELS, LAYOUT_RULES, classify_plate, name_colour,
                               two_means)
from vinplate.decode import Reading
from vinplate.grammar import parse_plate
from vinplate.ocr.ppocr import TextBox
from vinplate.scene import FieldReading, LineObs, SceneReader, _category_of
from vinplate.testing import make_logp

RESULT_KEYS = {"key", "label", "confidence", "bg", "ink", "reasons", "alternatives", "ev", "bh"}


def strip_plate(bg, ink, h=48, w=240, bar=4, period=16):
    """A plate-like crop: ``bg`` everywhere, ink bars (about 25 % of the width) over the middle 60 % of the rows."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    y0, y1 = int(0.2 * h), int(0.8 * h)
    for x in range(0, w, period):
        img[y0:y1, x:x + bar] = ink
    return img


WHITE, BLACK, YELLOW, GREEN, BLUE, RED = (248, 248, 246), (18, 18, 18), (255, 206, 22), (22, 150, 72), (0, 72, 172), (196, 28, 36)


# ================================================================================ constants
def test_labels_are_the_exact_strings_of_the_spec():
    assert LABELS == {
        "private": "Private vehicle", "commercial": "Commercial / transport vehicle",
        "rental": "Rental / self-drive vehicle", "ev_private": "Electric vehicle (private)",
        "ev_commercial": "Electric vehicle (commercial)", "temporary": "Temporary registration",
        "trade": "Trade certificate / dealer vehicle", "diplomatic": "Diplomatic / consular vehicle",
        "military": "Military / defence vehicle", "unknown": "Type not determined"}


def test_rule_tables_only_name_known_categories_and_colours():
    colours = {"black", "white", "grey", "red", "yellow", "green", "blue", "other"}
    for bg, ink, key, factor in COLOUR_RULES:
        assert bg in colours and (ink is None or ink in colours) and key in LABELS and 0 < factor <= 1
    for layout, (key, bgs) in LAYOUT_RULES.items():
        assert key in LABELS and bgs <= colours
    for key, alts in ALTERNATIVES.items():
        assert key in LABELS and len(alts) <= 2 and key not in alts and all(a in LABELS for a in alts)


# ================================================================================ colour naming
@pytest.mark.parametrize("rgb,name", [
    ((18, 18, 18), "black"), ((0, 0, 0), "black"), ((60, 60, 70), "black"),
    ((248, 248, 246), "white"), ((255, 255, 255), "white"), ((200, 200, 200), "white"),
    ((128, 128, 128), "grey"), ((110, 112, 115), "grey"),
    ((196, 28, 36), "red"), ((255, 0, 10), "red"), ((255, 10, 0), "red"), ((230, 60, 60), "red"),
    ((255, 206, 22), "yellow"), ((240, 190, 40), "yellow"), ((200, 160, 20), "yellow"),
    ((22, 150, 72), "green"), ((40, 170, 90), "green"),
    ((0, 72, 172), "blue"), ((24, 54, 150), "blue"), ((70, 120, 230), "blue"), ((0, 120, 220), "blue"),
    # the REAL Indian EV plate is a teal (hue 174-175 on a clean photo, 184-190 washed out through a monitor): green, never blue
    ((24, 181, 166), "green"), ((30, 162, 149), "green"), ((132, 189, 193), "green"), ((151, 199, 208), "green"), ((0, 166, 140), "green"),
    ((255, 244, 214), "white"),                  # a pale cream is a white plate under warm light, not yellow
    ((230, 30, 200), "other"),                   # magenta
])
def test_name_colour(rgb, name):
    assert name_colour(rgb)[0] == name


def test_name_colour_confidence_is_half_at_a_boundary_and_full_far_from_one():
    far = name_colour((255, 255, 255))[1]
    assert far == 1.0 or far > 0.95
    assert name_colour((18, 18, 18))[1] > 0.95
    edge = name_colour((72, 71, 72))[1]                                    # v = 0.282, one hair above the black boundary
    assert 0.5 <= edge < 0.52
    for rgb in [(200, 120, 40), (90, 90, 90), (250, 250, 100), (17, 71, 171)]:
        assert 0.5 <= name_colour(rgb)[1] <= 1.0


def test_hue_wraps_at_red_and_ties_resolve_r_then_g_then_b():
    assert name_colour((255, 0, 40))[0] == "red" and name_colour((255, 40, 0))[0] == "red"
    # r == g maximum: hue = 60 * ((g - b) / d) -> 60 degrees = yellow, not green
    assert name_colour((200, 200, 40))[0] == "yellow"
    # g == b maximum: hue = 60 * ((b - r) / d + 2) -> 180 degrees: cyan / teal is on the GREEN side (a real EV plate is teal)
    assert name_colour((40, 200, 200))[0] == "green"


# ================================================================================ 2-means
def test_two_means_background_is_the_bigger_cluster():
    bg, ink, dist = two_means(strip_plate(YELLOW, BLACK))
    assert np.allclose(bg, YELLOW, atol=1) and np.allclose(ink, BLACK, atol=1) and dist > 300
    bg, ink, _ = two_means(strip_plate(BLACK, YELLOW))                      # a dark plate: bg is still the bigger cluster
    assert np.allclose(bg, BLACK, atol=1) and np.allclose(ink, YELLOW, atol=1)


def test_two_means_one_colour_has_no_ink_and_is_deterministic():
    flat = np.full((30, 90, 3), 200, np.uint8)
    bg, ink, dist = two_means(flat)
    assert ink is None and dist == 0.0 and np.allclose(bg, 200)
    near = strip_plate((200, 200, 200), (215, 205, 200))                    # two colours 22 apart: below the 40 threshold
    assert two_means(near)[1] is None
    a = two_means(strip_plate(GREEN, WHITE))
    b = two_means(strip_plate(GREEN, WHITE))
    assert all(np.array_equal(x, y) for x, y in zip(a[:2], b[:2])) and a[2] == b[2]


def test_two_means_samples_a_strided_grid():
    big = strip_plate(WHITE, BLACK, h=480, w=2400, bar=40, period=160)       # 1.15 M px: step = 16, still the right answer
    bg, ink, _ = two_means(big)
    assert np.allclose(bg, WHITE, atol=1) and np.allclose(ink, BLACK, atol=1)


# ================================================================================ result shape + rules
def test_result_shape_is_exact_and_json_serialisable():
    r = classify_plate(strip_plate(YELLOW, BLACK), parse_plate("MH12AB1234"))
    assert set(r) == RESULT_KEYS
    assert r["key"] == "commercial" and r["label"] == LABELS["commercial"] and r["ev"] is False and r["bh"] is False
    assert isinstance(r["confidence"], float) and 0 <= r["confidence"] <= 1 and r["confidence"] == round(r["confidence"], 3)
    assert set(r["bg"]) == {"name", "rgb"} and r["bg"]["name"] == "yellow" and r["bg"]["rgb"] == [255, 206, 22]
    assert set(r["ink"]) == {"name", "rgb"} and r["ink"]["name"] == "black" and all(type(v) is int for v in r["ink"]["rgb"])
    assert r["reasons"][:2] == ["plate background looks yellow", "text looks black"]
    json.dumps(r)


@pytest.mark.parametrize("bg,ink,key", [
    (WHITE, BLACK, "private"), (YELLOW, BLACK, "commercial"), (BLACK, YELLOW, "rental"),
    (GREEN, WHITE, "ev_private"), (GREEN, YELLOW, "ev_commercial"), (YELLOW, RED, "temporary"),
    (RED, WHITE, "trade"), (BLUE, WHITE, "diplomatic"),
])
def test_colour_rules(bg, ink, key):
    text = "123CD45" if key == "diplomatic" else "MH12AB1234"        # a diplomatic plate carries a CD / CC / UN code
    r = classify_plate(strip_plate(bg, ink), parse_plate(text))
    assert r["key"] == key and r["label"] == LABELS[key]
    assert r["ev"] is key.startswith("ev_")


def test_a_blue_plate_with_an_ordinary_registration_is_never_called_diplomatic():
    """TEXT-PATTERN GUARD. A real photo of a green-plate EV was called 'diplomatic' (its teal colour was named blue). A
    diplomatic plate carries a CD / CC / UN code, so an ordinary registration on a 'blue' plate means the colour is wrong."""
    for text in ("MH12AB1234", "KA01MN4120", "22BH1234AA"):
        r = classify_plate(strip_plate(BLUE, WHITE), parse_plate(text))
        assert r["key"] == "unknown" and cat.R_DIP_FORMAT in r["reasons"], text
        assert [a["key"] for a in r["alternatives"]] == ["ev_private", "private"]
        assert r["ev"] is False and r["confidence"] < 0.5
    # yellow text on it: the commercial EV comes first
    y = classify_plate(strip_plate(BLUE, (255, 214, 40)), parse_plate("MH12AB1234"))
    assert y["key"] == "unknown" and [a["key"] for a in y["alternatives"]] == ["ev_commercial", "ev_private"]
    # the bh flag survives, in the same place as on the other unknown path
    bh = classify_plate(strip_plate(BLUE, WHITE), parse_plate("22BH1234AA"))
    assert bh["bh"] is True and bh["reasons"][2] == cat.R_BH


def test_blue_stays_diplomatic_when_the_text_is_a_diplomatic_code_or_unreadable():
    dip = classify_plate(strip_plate(BLUE, WHITE), parse_plate("123CD45"))
    assert dip["key"] == "diplomatic" and cat.R_BLUE not in dip["reasons"] and cat.R_DIP_FORMAT not in dip["reasons"]
    none = classify_plate(strip_plate(BLUE, WHITE), None)               # no text to contradict the colour
    assert none["key"] == "diplomatic" and cat.R_BLUE in none["reasons"] and none["confidence"] < dip["confidence"]


# measured on the user's photo of a green-plate Toyota Camry (KA01MN4120): clean, and photographed off a monitor
REAL_EV_PLATES = [((24, 181, 166), (246, 246, 246)), ((30, 162, 149), (246, 246, 246)),
                  ((132, 189, 193), (250, 250, 250)), ((151, 199, 208), (240, 240, 240))]


@pytest.mark.parametrize("bg,ink", REAL_EV_PLATES)
def test_the_real_teal_ev_plate_is_an_ev_not_a_diplomatic_plate(bg, ink):
    for text in ("KA01MN4120", None):
        r = classify_plate(strip_plate(bg, ink), parse_plate(text) if text else None)
        assert r["key"] == "ev_private" and r["ev"] is True and r["bg"]["name"] == "green", (bg, text, r)
        assert "diplomatic" not in [a["key"] for a in r["alternatives"]] or text is None


def test_an_ordinary_registration_never_gets_diplomatic_as_an_alternative():
    for bg, ink in [(GREEN, WHITE), (BLACK, YELLOW), *REAL_EV_PLATES, ((90, 160, 110), (240, 240, 240)), ((40, 90, 40), (250, 250, 250))]:
        r = classify_plate(strip_plate(bg, ink), parse_plate("KA01MN4120"))
        assert "diplomatic" not in [a["key"] for a in r["alternatives"]] and r["key"] != "diplomatic", (bg, ink, r["key"])


def test_unmatched_colour_pairs_are_unknown_with_the_dark_alternatives():
    r = classify_plate(strip_plate(BLACK, WHITE), parse_plate("MH12AB1234"))
    assert r["key"] == "unknown" and r["label"] == LABELS["unknown"] and cat.R_NO_RULE in r["reasons"]
    assert [a["key"] for a in r["alternatives"]] == ["military", "rental"]
    assert r["ev"] is False
    r = classify_plate(strip_plate((140, 60, 160), WHITE), None)           # a purple plate
    assert r["key"] == "unknown" and [a["key"] for a in r["alternatives"]] == ["private", "commercial"]


def test_one_colour_crop_is_named_but_barely_trusted():
    r = classify_plate(np.full((30, 90, 3), 250, np.uint8), None)
    assert r["ink"] is None and cat.R_ONE_COLOUR in r["reasons"]
    assert r["confidence"] < 0.34 and r["alternatives"]


@pytest.mark.parametrize("crop", [None, np.zeros((0, 10, 3), np.uint8), np.zeros((10, 0, 3), np.uint8), np.zeros((10, 10), np.uint8)])
def test_unusable_crops_are_unknown_not_an_exception(crop):
    r = classify_plate(crop, None)
    assert set(r) == RESULT_KEYS and r["key"] == "unknown" and r["confidence"] == 0.0 and r["bg"] is None and r["ink"] is None


def test_a_one_pixel_crop_does_not_crash():
    assert classify_plate(np.zeros((1, 1, 3), np.uint8), None)["key"] == "unknown"


# ================================================================================ layout vs colour
def _base(crop):
    """The colour-only confidence of a white / blue / red / dark crop (every rule factor but blue is 1)."""
    return classify_plate(crop, None)["confidence"]


def test_text_layout_wins_over_colour_and_agreement_adds_a_tenth():
    dip_blue = classify_plate(strip_plate(BLUE, WHITE), parse_plate("123CD45"))
    assert dip_blue["key"] == "diplomatic" and cat.R_LAYOUT["diplomatic"] in dip_blue["reasons"] and cat.R_AGREE in dip_blue["reasons"]
    base = _base(strip_plate(BLUE, WHITE)) / 0.7                           # undo the blue rule's x0.7
    assert dip_blue["confidence"] == pytest.approx(min(1.0, base + 0.1), abs=0.003)


def test_a_disagreeing_colour_multiplies_the_confidence_by_06_and_says_so():
    white = strip_plate(WHITE, BLACK)
    r = classify_plate(white, parse_plate("123CD45"))
    assert r["key"] == "diplomatic"                                        # the text pattern decides
    assert cat.R_DISAGREE.format("white", "diplomatic") in r["reasons"] and cat.R_AGREE not in r["reasons"]
    assert r["confidence"] == pytest.approx(_base(white) * 0.6, abs=0.003)
    assert [a["key"] for a in r["alternatives"]][0] == "private"           # what the colour alone says comes first


def test_military_and_dealer_layouts():
    mil = parse_plate("12A123456B")
    assert mil["layout"] == "military"
    assert classify_plate(strip_plate((26, 26, 30), WHITE), mil)["key"] == "military"
    assert classify_plate(strip_plate(WHITE, BLACK), mil)["key"] == "military"
    dealer = parse_plate("DLR012345")
    assert classify_plate(strip_plate(RED, WHITE), dealer)["key"] == "trade"
    r = classify_plate(strip_plate(WHITE, BLACK), dealer)
    assert r["key"] == "trade" and cat.R_DISAGREE.format("white", "dealer") in r["reasons"]
    assert r["ev"] is False and r["bh"] is False


def test_bharat_series_is_a_flag_not_a_category():
    bh = parse_plate("22BH1234AA")
    white = classify_plate(strip_plate(WHITE, BLACK), bh)
    assert white["key"] == "private" and white["bh"] is True and cat.R_BH in white["reasons"] and cat.R_AGREE in white["reasons"]
    plain = classify_plate(strip_plate(WHITE, BLACK), parse_plate("MH12AB1234"))
    assert white["confidence"] >= plain["confidence"] and plain["bh"] is False
    yellow = classify_plate(strip_plate(YELLOW, BLACK), bh)                 # a BH plate that is not white: still a flag, less sure
    assert yellow["key"] == "commercial" and yellow["bh"] is True and cat.R_BH_COLOUR in yellow["reasons"]
    assert yellow["confidence"] < classify_plate(strip_plate(YELLOW, BLACK), None)["confidence"]
    unk = classify_plate(strip_plate(BLACK, WHITE), bh)
    assert unk["key"] == "unknown" and unk["bh"] is True and cat.R_BH in unk["reasons"]


# ================================================================================ confidence + alternatives
def test_alternatives_appear_only_when_the_confidence_is_low_and_never_repeat_the_answer():
    clean = classify_plate(strip_plate(YELLOW, BLACK), parse_plate("MH12AB1234"))
    assert clean["confidence"] >= cat.ALT_BELOW and clean["alternatives"] == []
    dim = classify_plate(strip_plate((230, 100, 40), (10, 10, 10)), parse_plate("MH12AB1234"))        # orange-ish: a yellow near its edge
    assert dim["key"] == "commercial" and dim["confidence"] < cat.ALT_BELOW
    assert [a["key"] for a in dim["alternatives"]] == ALTERNATIVES["commercial"][:2]
    for a in dim["alternatives"]:
        assert set(a) == {"key", "label"} and a["label"] == LABELS[a["key"]] and a["key"] != dim["key"]


def test_confidence_falls_with_the_separation_between_text_and_background():
    strong = classify_plate(strip_plate(WHITE, BLACK), None)["confidence"]
    weak = classify_plate(strip_plate(WHITE, (150, 150, 150)), None)["confidence"]                    # grey text on white: only ~140 apart
    assert weak < strong


def test_classification_is_deterministic():
    crop = render_scheme(np.random.default_rng(3), "ev_commercial").crop
    assert classify_plate(crop, None) == classify_plate(crop.copy(), None)


# ================================================================================ synthetic accuracy
@pytest.fixture(scope="module")
def synthetic():
    return evaluate(n_per=8, seed=7)


def test_every_scheme_is_classified_on_clean_synthetic_plates(synthetic):
    ok, n = accuracy(synthetic["rows"], light="clean")
    assert n == 8 * len(SCHEMES) and ok / n >= 0.95, (ok, n)
    for scheme, (a, b) in summarise(synthetic["rows"])["clean_per_scheme"].items():
        assert a / b >= 0.9, (scheme, a, b)


def test_degraded_lighting_stays_useful_and_wrong_answers_are_rarely_confident(synthetic):
    """Honest floors, not targets: the synthetic degradations are mild and the plate colours are the rule table's own."""
    s = summarise(synthetic["rows"])
    degraded = [r for r in synthetic["rows"] if r["light"] != "clean"]
    ok = sum(r["got"] == r["expected"] for r in degraded)
    assert ok / len(degraded) >= 0.85, ok / len(degraded)
    for light, d in s["per_light"].items():
        floor = 0.60 if light == "dim" else 0.85         # dim: a white plate turns grey and is (rightly) not called private
        assert d["correct"] / d["n"] >= floor, (light, d)
    confident_wrong = sum(d["confident_wrong"] for d in s["per_light"].values())
    assert confident_wrong / len(degraded) <= 0.01, confident_wrong
    assert CONFIDENT == cat.ALT_BELOW


def test_no_synthetic_plate_with_an_ordinary_registration_is_ever_called_diplomatic_or_military(synthetic):
    """The text-pattern guard as an invariant over every scheme and lighting: only the diplomatic / military text layouts may
    produce those categories. (Military is layout-only by construction; diplomatic used to leak in through the blue colour.)"""
    rows = synthetic["rows"]
    assert rows and all(set(r) >= {"scheme", "light", "expected", "got"} for r in rows)
    bad = [(r["scheme"], r["light"], r["got"]) for r in rows
           if r["got"] in ("diplomatic", "military") and r["expected"] not in ("diplomatic", "military")]
    assert not bad, bad[:10]


def test_a_dim_white_plate_is_not_called_private_it_abstains(synthetic):
    dim_private = [r for r in synthetic["rows"] if r["light"] == "dim" and r["expected"] == "private"]
    assert dim_private and all(r["got"] in ("private", "unknown") for r in dim_private)
    assert sum(r["got"] == "unknown" for r in dim_private) > len(dim_private) // 2


def test_lighting_variants_are_all_exercised():
    rng = np.random.default_rng(0)
    p = render_scheme(rng, "private")
    outs = {k: degrade(p.crop, k, rng) for k in LIGHTING}
    assert outs["clean"] is p.crop and all(o.shape == p.crop.shape and o.dtype == np.uint8 for o in outs.values())
    assert len({o.tobytes() for o in outs.values()}) == len(LIGHTING)


def test_schemes_render_the_text_layout_their_rule_needs():
    rng = np.random.default_rng(5)
    for scheme in ("trade", "diplomatic", "military", "bharat", "private"):
        p = render_scheme(rng, scheme)
        assert parse_plate(p.text)["layout"] == p.layout, (scheme, p.text)
    assert render_scheme(rng, "trade").layout == "dealer" and render_scheme(rng, "military").layout == "military"


# ================================================================================ scene integration
class FakeEngine:
    """Detector + recogniser stand-in: fixed boxes and a fixed log-posterior."""

    def __init__(self, boxes, lp, raw):
        self.boxes, self.lp, self.raw = list(boxes), lp, raw

    def detect(self, rgb):
        return self.boxes

    def rec_probs_many(self, crops):
        return [tuple(c.shape[:2]) for c in crops]

    def rec_probs(self, crop):
        return self.rec_probs_many([crop])[0]

    def to_logp(self, probs, temperature=1.0):
        return self.lp

    def greedy(self, probs):
        return self.raw, 1.0


def _scene_with_plate(crop, x=60, y=80, size=(420, 760)):
    img = np.full((*size, 3), (112, 112, 118), np.uint8)
    h, w = crop.shape[:2]
    img[y:y + h, x:x + w] = crop
    q = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    return img, TextBox(q, 0.95)


@pytest.mark.parametrize("scheme", ["commercial", "ev_private", "private"])
def test_scene_reader_attaches_the_category_of_the_plate_crop(scheme):
    rng = np.random.default_rng(11)
    p = render_scheme(rng, scheme)
    img, box = _scene_with_plate(p.crop)
    res = SceneReader(FakeEngine([box], make_logp(p.text), p.text), multi_view=False).read(img, domains=("plate",))
    assert res.plate is not None and res.plate.category is not None
    assert res.plate.category["key"] == p.expected
    d = res.plate.as_dict()
    assert d["category"] == res.plate.category and set(d["category"]) == RESULT_KEYS
    json.dumps(d["category"])


def test_a_vin_has_no_category_and_the_key_is_still_present():
    class _Ln:
        box = TextBox(np.array([[0, 0], [40, 0], [40, 10], [0, 10]], float), 0.9)
        crop = strip_plate(WHITE, BLACK)
    assert _category_of("vin", [_Ln()], None) is None and _category_of("plate", [], None) is None
    r = Reading("MH12AB1234", True, "accept", 0.9, 0.9, [0.9] * 10, [], [], 0.0, "MH12AB1234")
    fr = FieldReading("vin", r, None, (0, 0, 1, 1), [])
    assert "category" in fr.as_dict() and fr.as_dict()["category"] is None


def test_a_tiny_crop_gives_no_category():
    class _Ln:
        box = TextBox(np.array([[0, 0], [40, 0], [40, 10], [0, 10]], float), 0.9)
        crop = np.zeros((3, 40, 3), np.uint8)
    assert _category_of("plate", [_Ln()], None) is None


def test_the_category_uses_the_biggest_line_crop_of_the_field():
    small = TextBox(np.array([[0, 0], [30, 0], [30, 8], [0, 8]], float), 0.9)
    large = TextBox(np.array([[0, 0], [200, 0], [200, 50], [0, 50]], float), 0.9)
    lns = [LineObs(small, strip_plate(YELLOW, BLACK, h=8, w=30), None, make_logp("MH12"), "MH12"),
           LineObs(large, strip_plate(GREEN, WHITE), None, make_logp("AB1234"), "AB1234")]
    assert _category_of("plate", lns, parse_plate("MH12AB1234"))["key"] == "ev_private"


def test_fused_photos_take_the_category_from_the_best_photo():
    rng = np.random.default_rng(13)
    p = render_scheme(rng, "commercial")
    img, box = _scene_with_plate(p.crop)
    res = SceneReader(FakeEngine([box], make_logp(p.text), p.text), multi_view=False).read_frames([img, img], domains=("plate",))
    assert res.plate.views == 2 and "fused 2 photos" in res.plate.notes[0]
    assert res.plate.category is not None and res.plate.category["key"] == "commercial"
    assert res.plate.as_dict()["category"]["key"] == "commercial"
