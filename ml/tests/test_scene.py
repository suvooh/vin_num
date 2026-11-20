"""Scene reader logic that does not need a model: decode choices on simulated posteriors,
VIN folding, label-clutter rejection, window search, line ownership and the box geometry.
(Tests that run the real PP-OCR engine are in test_ocr_engine.py.)"""

import numpy as np
import pytest

from vinplate.decode import Reading, read_line
from vinplate.grammar import PLATE_GRAMMAR, VIN_GRAMMAR, parse_plate, vin_check_digit_ok
from vinplate.ocr.ppocr import TextBox
from vinplate.scene import (MAX_STRETCH, MIN_ASPECT, NO_SCREEN_CONTEXT, FieldReading, LineObs, SceneReader, _compatible,
                            _fold_vin, _near, _side_by_side, _stacked, normalise_aspect, screen_context, slot_candidates)
from vinplate.testing import make_logp
from vinplate.vocab import CHARS, VOCAB_SIZE


@pytest.fixture(scope="module")
def reader():
    return SceneReader(engine=object())          # decode paths never touch the engine


def _box(x, y, w, h, angle=0.0):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    u, v = np.array([c, s]) * w, np.array([-s, c]) * h
    o = np.array([x, y], float)
    return TextBox(np.array([o, o + u, o + u + v, o + v]), 0.9)


def _field(domain, text, box, status="accept", evidence=None):
    r = Reading(text, True, status, 0.9, 0.9, [0.9] * len(text), [], [], 0.0, text)
    return FieldReading(domain, r, None, box, [], evidence=len(text) if evidence is None else evidence)


# ------------------------------------------------------------------------------ VIN folding
def test_fold_vin_moves_iOQ_onto_the_digits():
    lp = np.log(np.full((1, VOCAB_SIZE), 1e-6))
    for ch, p in (("I", 0.6), ("O", 0.2), ("Q", 0.1)):
        lp[0, CHARS.index(ch) + 1] = np.log(p)
    out = np.exp(_fold_vin(lp))[0]
    assert out[CHARS.index("1") + 1] > 0.55 and out[CHARS.index("0") + 1] > 0.28
    assert max(out[CHARS.index(c) + 1] for c in "IOQ") < 1e-6
    assert out.sum() == pytest.approx(1.0)


def test_vin_read_with_an_I_where_a_1_belongs_is_recovered(reader):
    # the network is sure it saw 'I' at the check-digit-less serial; ISO forbids I, so it must be a 1
    lp = make_logp("MAL907KE1E6602791", confuse={13: ("I", 0.9)})
    r, _ = reader._decode("vin", [lp], allow_legacy=False)
    assert r.text == "MAL907KE1E6602791" or r.text[13] == "1", r.text


# ------------------------------------------------------------------------- VIN false positives
def test_label_text_that_looks_like_a_vin_is_rejected(reader):
    # "TYRE PRESSURE 32 PSI" is exactly 17 legal VIN characters once I is folded to 1
    lp = make_logp("TYREPRESSURE32PSI")
    assert reader._decode("vin", [lp], allow_legacy=False) is None
    assert reader._decode("vin", [lp], allow_legacy=True) is None


def test_clutter_filter_never_fires_when_the_check_digit_passes():
    assert SceneReader._is_clutter("TYREPRESSURE32PS1") is True
    assert SceneReader._is_clutter("1M8GDM9AXKP042788") is False          # valid ISO check digit


def test_legacy_chassis_needs_permission_and_is_never_auto_accepted(reader):
    lp = make_logp("81GZ16D60VVY9")                                       # 13 chars: not a 17-char VIN
    assert reader._decode("vin", [lp], allow_legacy=False) is None
    got = reader._decode("vin", [lp], allow_legacy=True)
    assert got is not None
    r, _ = got
    assert r.status != "accept" and any("legacy" in x for x in r.reasons)


def test_a_word_is_not_a_chassis_number(reader):
    assert reader._decode("vin", [make_logp("ENGINEFRONTWEIGHT")], allow_legacy=True) is None


def test_strict_vin_read_is_accepted(reader):
    got = reader._decode("vin", [make_logp("MAL907KE1E6602791")], allow_legacy=False)
    assert got is not None and got[0].text == "MAL907KE1E6602791" and got[0].status in ("accept", "review")


# ------------------------------------------------------------------------------- plate windows
@pytest.mark.parametrize("noisy,plate", [
    ("INDMH12AB1234", "MH12AB1234"),               # hologram text before the plate
    ("UP16C0059TC20", "UP16C0059"),                # temporary-registration mark after it
    ("IND22BH1234AAX", "22BH1234AA"),
])
def test_plate_inside_surrounding_text_is_found(reader, noisy, plate):
    got = reader._decode("plate", [make_logp(noisy)], allow_legacy=False)
    assert got is not None
    r, note = got
    assert r.text == plate and "sub-window" in note, (noisy, r.text, r.status)
    assert parse_plate(r.text) is not None


def test_clean_plate_is_decoded_whole(reader):
    r, note = reader._decode("plate", [make_logp("TN71JE5250")], allow_legacy=False)
    assert r.text == "TN71JE5250" and r.status == "accept" and note == ""


def test_unreadable_character_is_flagged_not_invented(reader):
    r, _ = reader._decode("plate", [make_logp("KA01AB1234", unreadable=(4,))], allow_legacy=False)
    assert r.status == "review" and "?" in r.masked


def test_a_hallucinated_short_read_is_not_a_plate(reader):
    r = reader._decode("plate", [make_logp("MH12")], allow_legacy=False)
    assert r is None or r[0].status != "accept"


# --------------------------------------------------------------------------- line ownership
def _assign(*fields):
    return SceneReader._assign(list(fields))


def test_a_text_line_is_never_both_the_plate_and_the_vin():
    plate = _field("plate", "MH12AB1234", (100, 100, 300, 140))
    vin = _field("vin", "MAL907KE1E6602791", (95, 98, 310, 142), status="review", evidence=12)
    p, v = _assign(plate, vin)
    assert (p is None) != (v is None) or (p is not None and v is not None and p.box != v.box)
    assert not (p is plate and v is vin)


def test_separate_lines_give_both_fields():
    plate = _field("plate", "MH12AB1234", (100, 100, 300, 140))
    vin = _field("vin", "MAL907KE1E6602791", (100, 400, 500, 440))
    p, v = _assign(plate, vin)
    assert p is plate and v is vin


def test_weak_or_rejected_candidates_are_not_reported():
    weak = _field("plate", "MH12AB1234", (0, 0, 10, 10), evidence=3)
    rejected = _field("plate", "MH12AB1234", (0, 0, 10, 10), status="reject")
    low = _field("plate", "MH12AB1234", (0, 0, 10, 10))
    low.reading.confidence = 0.05
    assert _assign(weak, rejected, low) == (None, None)


# ------------------------------------------------------------------------------- geometry
def test_two_row_plate_rows_are_stacked_not_side_by_side():
    top, bottom = _box(100, 100, 90, 40), _box(100, 150, 200, 44)
    assert _stacked(top, bottom) and not _side_by_side(top, bottom)
    assert not _stacked(bottom, top)


def test_split_plate_pieces_are_side_by_side():
    a, b = _box(100, 100, 90, 40), _box(215, 102, 200, 40)
    assert _side_by_side(a, b) and not _stacked(a, b)


def test_tilted_rows_still_stack():
    assert _stacked(_box(100, 100, 90, 40, 12), _box(100 - 9, 100 + 50, 200, 42, 12))


def test_far_apart_boxes_are_not_one_plate():
    assert not _near(_box(0, 0, 100, 30), _box(0, 400, 100, 30))
    assert _near(_box(0, 0, 100, 30), _box(0, 40, 100, 30))


def test_masked_readings_compatible_across_photos():
    assert _compatible("MH12AB1234", "MH12AB1234")
    assert _compatible("MH12??1234", "MH12AB1234")
    assert _compatible("MH12AB1234", "MH12AB1284")          # one misread digit tolerated
    assert not _compatible("MH12AB1234", "KA01XY9876")
    assert not _compatible("MH12AB1234", "MH12AB123")


def test_read_line_on_vocab_posteriors_still_matches_plain_decode():
    lp = make_logp("DL1CAB1234")
    assert read_line(lp, PLATE_GRAMMAR).text == "DL1CAB1234"


# ------------------------------------------------------------------- decode-time safety policy
def test_lookalike_ambiguity_is_shown_as_unknown_not_read(reader):
    """Series letter Q vs O: the net says Q (68%) but O has 25% and the grammar allows both."""
    lp = make_logp("WB97MQ3156", confuse={5: ("O", 0.25)})
    r, _ = reader._decode("plate", [lp], allow_legacy=False)
    assert r.status == "review" and 5 in r.inferred and r.masked[5] == "?"
    assert any("look-alike" in x for x in r.reasons)


def test_lookalike_that_the_grammar_forbids_is_not_flagged(reader):
    """A digit slot cannot be the letter O, so an 'O' alternative there is not an ambiguity."""
    lp = make_logp("TN71JE5250", confuse={9: ("O", 0.25)})
    r, _ = reader._decode("plate", [lp], allow_legacy=False)
    assert r.text == "TN71JE5250" and r.status == "accept"


def test_series_less_plate_is_never_auto_accepted(reader):
    """CG 65 Q 6426 read as CG 65 6426: the Q was dropped, and a series-less read looks legal."""
    r, _ = reader._decode("plate", [make_logp("CG656426")], allow_legacy=False)
    assert r.status == "review" and any("series" in x for x in r.reasons)


def test_short_number_is_never_auto_accepted(reader):
    r, _ = reader._decode("plate", [make_logp("MH12AB123")], allow_legacy=False)
    assert r.status != "accept"


def test_series_letter_I_O_Q_is_never_auto_accepted(reader):
    """HSRP series skip I and O (and often Q): a read containing one is probably a misread look-alike."""
    for plate in ("WB97MO3156", "AR87QC5837"):
        r, _ = reader._decode("plate", [make_logp(plate)], allow_legacy=False)
        assert r.status == "review" and any("I/O/Q" in x for x in r.reasons), (plate, r.status, r.reasons)
    r, _ = reader._decode("plate", [make_logp("WB97MD3156")], allow_legacy=False)
    assert r.status == "accept"


# ============================================================== aspect normalisation (CTC time-step budget)
def test_normalise_aspect_stretches_a_condensed_plate_crop_to_the_budget():
    out = normalise_aspect(np.zeros((99, 258, 3), np.uint8))     # the real HR88B8888 box: 2.6 : 1 -> only 16 CTC steps
    assert out.shape == (99, 396, 3) and out.dtype == np.uint8   # 4.0 * 99


def test_normalise_aspect_never_touches_wide_text():
    for h, w in ((48, 200), (50, 200), (48, 600), (30, 1200)):   # (50, 200) is exactly 4.0
        crop = np.zeros((h, w, 3), np.uint8)
        assert normalise_aspect(crop) is crop                     # the very same array, not a resampled copy


def test_normalise_aspect_is_capped_at_max_stretch():
    assert normalise_aspect(np.zeros((40, 40, 3), np.uint8)).shape == (40, 100, 3)      # 1:1 would need x4, capped at x2.5


def test_normalise_aspect_rounds_half_up_not_to_even():
    assert normalise_aspect(np.zeros((4, 5, 3), np.uint8)).shape[1] == 13               # 5 * 2.5 = 12.5 -> 13 (round-half-even: 12)


def test_normalise_aspect_constants_are_the_ones_the_dart_port_must_match():
    assert (MIN_ASPECT, MAX_STRETCH) == (4.0, 2.5)


class FakeEngine:
    """Detector + recogniser stand-in. A crop's 'posterior' is just its (h, w), so the log-posterior and the raw
    greedy text can be chosen per crop SIZE and the sizes the reader asks the recogniser about are recorded."""

    def __init__(self, boxes=(), lp=None, raw="", lps=None, raws=None):
        self.boxes, self.lp, self.raw = list(boxes), lp, raw
        self.lps, self.raws = lps or {}, raws or {}
        self.seen: list[tuple[int, int]] = []

    def detect(self, rgb):
        return self.boxes

    def rec_probs_many(self, crops):
        self.seen += [tuple(c.shape[:2]) for c in crops]
        return [tuple(c.shape[:2]) for c in crops]

    def rec_probs(self, crop):
        return self.rec_probs_many([crop])[0]

    def to_logp(self, probs, temperature=1.0):
        return self.lps.get(probs, self.lp)

    def greedy(self, probs):
        return self.raws.get(probs, self.raw), 1.0


class FakeEngineNoGreedy(FakeEngine):
    greedy = None                                               # an engine without the RapidOCR-style raw text


def _scene_img(w=500, h=400):
    return (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)


def test_every_line_crop_the_recogniser_sees_meets_the_aspect_budget():
    # two condensed boxes (aspect 2.1 and 2.3) that are also near enough to form a region
    eng = FakeEngine([_box(150, 100, 110, 52), _box(140, 160, 130, 56)], make_logp("MH12AB1234"), raw="MH12AB1234")
    SceneReader(eng, multi_view=False).read(_scene_img(), domains=("plate",))
    assert (52, 208) in eng.seen and (56, 224) in eng.seen                    # the line crops, stretched to exactly 4 : 1
    assert (52, 110) not in eng.seen and (56, 130) not in eng.seen            # ... and never recognised unstretched
    assert all(w / h >= 3.9 for h, w in eng.seen), eng.seen                   # the region crop hit the x2.5 cap (3.94)


def test_region_rows_are_stretched_before_they_are_recognised():
    lp = make_logp("MH12")
    lines = [LineObs(_box(100, 100, 120, 40), np.zeros((40, 120, 3), np.uint8), None, lp, "MH12"),
             LineObs(_box(100, 150, 120, 40), np.zeros((40, 120, 3), np.uint8), None, lp, "AB12")]
    eng = FakeEngine(lp=lp)
    regions = SceneReader(eng, multi_view=False)._regions(lines, lines, _scene_img())
    assert regions and len(regions[0]) == 2                                   # a plate-shaped group is split into two rows
    assert len(eng.seen) == 2 and all(w / h >= MIN_ASPECT - 0.05 for h, w in eng.seen), eng.seen
    for row in regions[0]:
        assert row.crop.shape[1] / row.crop.shape[0] >= MIN_ASPECT - 0.05     # the stored crop is the stretched one


def test_fused_photos_re_read_stretched_crops_too():
    eng = FakeEngine([_box(100, 100, 120, 40)], make_logp("TN71JE5250"), raw="TN71JE5250")      # 3.0 : 1
    res = SceneReader(eng, multi_view=False).read_frames([_scene_img(), _scene_img()], domains=("plate",))
    assert res.plate is not None and res.plate.views == 2 and "fused 2 photos" in res.plate.notes[0]
    assert (40, 160) in eng.seen and (40, 120) not in eng.seen                # observe AND the fusion re-read are stretched
    assert res.plate.posterior is not None and isinstance(res.plate.as_dict()["unknown_candidates"], list)


# ============================================================== unknown-slot candidates (the 'innate ability')
def _lp_with_slot(text, i, masses, fpc=3):
    """make_logp(text) with the spike frame of character i replaced by an explicit class distribution."""
    lp = make_logp(text, frames_per_char=fpc)
    p = np.full(VOCAB_SIZE, 0.002)
    for ch, v in masses.items():
        p[CHARS.index(ch) + 1] = v
    lp[(fpc + 1) * i] = np.log(p / p.sum())
    return lp


def test_unknown_letter_slot_lists_only_legal_letters_ranked():
    # HR88?8888: the character after the RTO digits MUST be a letter. The network half-saw a B (30%) and a digit 8 (10%).
    lp = _lp_with_slot("HR88B8888", 4, {"?": 0.55, "B": 0.30, "8": 0.10, "R": 0.03, "D": 0.02})
    out = slot_candidates("HR88?8888", lp, PLATE_GRAMMAR)
    assert [e["position"] for e in out] == [4]
    opts = out[0]["options"]
    assert [o["char"] for o in opts] == ["B", "R", "D"]                       # ranked; the digit 8 is never offered
    assert [o["p"] for o in opts] == sorted((o["p"] for o in opts), reverse=True)
    assert opts[0]["p"] == pytest.approx(0.768, abs=0.01)                     # softmax over the LEGAL characters only
    assert out[0]["suggested"] == "B"                                         # dominant: >= 0.6 and 0.3 ahead of the next


def test_two_close_letters_give_options_but_no_suggestion():
    lp = _lp_with_slot("HR88B8888", 4, {"?": 0.5, "B": 0.25, "D": 0.22})
    (e,) = slot_candidates("HR88?8888", lp, PLATE_GRAMMAR)
    assert [o["char"] for o in e["options"]] == ["B", "D"] and e["suggested"] is None


def test_options_below_five_percent_are_dropped():
    lp = _lp_with_slot("HR88B8888", 4, {"?": 0.7, "B": 0.2})
    (e,) = slot_candidates("HR88?8888", lp, PLATE_GRAMMAR)
    assert [o["char"] for o in e["options"]] == ["B"] and e["suggested"] == "B"      # the other 25 letters are < 5% each


def test_a_digit_only_slot_offers_only_digits():
    lp = _lp_with_slot("HR88B8888", 7, {"?": 0.5, "8": 0.3, "3": 0.1, "B": 0.05})
    (e,) = slot_candidates("HR88B88?8", lp, PLATE_GRAMMAR)                    # inside the 4-digit number: no letter is legal
    assert e["position"] == 7 and [o["char"] for o in e["options"]][:2] == ["8", "3"]
    assert all(o["char"].isdigit() for o in e["options"])


def test_a_letter_that_is_legal_but_improbable_in_a_digit_slot_is_not_offered():
    lp = _lp_with_slot("HR88B8888", 5, {"?": 0.5, "8": 0.3, "3": 0.1, "B": 0.05})
    (e,) = slot_candidates("HR88B?888", lp, PLATE_GRAMMAR)        # 'HR88BB888' (series BB, 3-digit number) parses, but is a far less likely plate
    assert [o["char"] for o in e["options"]] == ["8", "3"] and e["suggested"] == "8"


def test_inferred_but_known_characters_get_candidates_too():
    lp = _lp_with_slot("HR88B8888", 4, {"B": 0.5, "R": 0.2, "D": 0.1})
    out = slot_candidates("HR88B8888", lp, PLATE_GRAMMAR, inferred=[4])
    assert [e["position"] for e in out] == [4] and [o["char"] for o in out[0]["options"]][:2] == ["B", "R"]
    assert slot_candidates("HR88B8888", lp, PLATE_GRAMMAR) == []              # nothing unknown, nothing weak: nothing to offer


def test_vin_slots_offer_the_folded_posterior_so_iOQ_never_appear():
    # the reader folds I/O/Q onto 1/0 before it decodes a VIN, and that folded posterior is what slot_candidates gets
    lp = _fold_vin(_lp_with_slot("MAL907KE1E6602791", 3, {"?": 0.5, "9": 0.2, "O": 0.15, "Q": 0.1, "8": 0.05}))
    (e,) = slot_candidates("MAL?07KE1E6602791", lp, VIN_GRAMMAR)
    got = [o["char"] for o in e["options"]]
    assert set(got) >= {"9", "0"} and not set(got) & set("IOQ")


def test_no_candidates_without_an_alignment():
    assert slot_candidates("?" * 40, make_logp("ABC"), PLATE_GRAMMAR) == []             # more characters than frames
    assert slot_candidates("HR88?8888", None, PLATE_GRAMMAR) == []                      # no posterior kept
    assert slot_candidates("", make_logp("ABC"), PLATE_GRAMMAR) == []


def test_the_unknown_slot_reaches_the_reading_dict_through_the_scene_reader():
    lp = _lp_with_slot("HR88B8888", 4, {"?": 0.55, "B": 0.30, "8": 0.10, "R": 0.03, "D": 0.02})
    eng = FakeEngine([_box(100, 100, 300, 60)], lp, raw="HR88B8888")
    res = SceneReader(eng, multi_view=False).read(_scene_img(), domains=("plate",))
    d = res.plate.as_dict()
    assert d["value"] == "HR88?8888" and d["status"] != "accept"
    (e,) = d["unknown_candidates"]
    assert e["position"] == 4 and d["value"][e["position"]] == "?"
    assert [o["char"] for o in e["options"]][0] == "B" and not any(o["char"].isdigit() for o in e["options"])
    assert e["suggested"] == "B"
    clean = SceneReader(FakeEngine([_box(100, 100, 300, 60)], make_logp("TN71JE5250"), raw="TN71JE5250"),
                        multi_view=False).read(_scene_img(), domains=("plate",))
    assert clean.plate.as_dict()["value"] == "TN71JE5250" and clean.plate.as_dict()["unknown_candidates"] == []


# ============================================================== screen context
ATTACK_LINES = [
    "DE?CX?1", "E-Ringi", "Kafka R", "Quest27", "AskGartner Home", "e-Ringi",
    "https://www.bing.com/images/search?view=detailV2&ccid=3SUMWcMG&id=6770EFC1A91614CEA936",
    "HR88B8888 Becomes India's Most Expensive Car Registration Number Sold ...",
    "newsable.asianetnews.com 1280 x 720 jpeg 11 mos ago", "Search", "HR8888", "IND", "HR88BS", "REGISTR",
    "Related searc", "California", "See more details", "Car Number Plate in Mumbai", "Car Number Plate in Gurgaon",
    "Renewal of RC for 15-year-old cars to", "4444 Number Plate Price and", "Procedure of renewing vehicle registration",
    "Registration Process in India", "How to Che",
]


def test_attack_like_screen_text_scores_the_maximum():
    ctx = screen_context(ATTACK_LINES, excluded={10, 11})                    # the plate lines themselves are not context
    assert ctx["url"] is True and ctx["text_lines"] >= 14
    assert ctx["ui_words"] == ["DETAILS", "IMAGES", "RELATED", "SEARCH"]
    assert ctx["score"] == 3.0 and len(ctx["reasons"]) == 3
    assert set(ctx) == {"score", "url", "ui_words", "text_lines", "reasons"}


def test_a_plain_plate_scene_has_no_screen_context():
    assert screen_context(["MARUTI SUZUKI", "MH12AB1234"], excluded={1}) == {**NO_SCREEN_CONTEXT, "text_lines": 1}
    assert screen_context([]) == NO_SCREEN_CONTEXT


def test_a_dealer_sticker_url_alone_weighs_one_point():
    ctx = screen_context(["www.sharmamotors.com", "MH12AB1234"], excluded={1})
    assert ctx["url"] is True and ctx["ui_words"] == [] and ctx["score"] == 1.0


def test_many_lines_alone_weigh_half_a_point_and_short_lines_do_not_count():
    ctx = screen_context(["SHOP %d" % i for i in range(20)] + ["AB", "1", "?!"])
    assert ctx["text_lines"] == 20 and ctx["score"] == 0.5 and not ctx["url"]
    assert screen_context(["SHOP %d" % i for i in range(13)])["score"] == 0.0           # 13 < 14


def test_three_distinct_interface_words_are_needed():
    assert screen_context(["Search", "Settings"])["score"] == 0.0
    assert screen_context(["Search", "Settings", "Download"])["score"] == 1.5
    assert screen_context(["search search search"])["ui_words"] == ["SEARCH"]           # a repeated word is ONE word


def test_url_detection_rules():
    for t in ("http://x", "HTTPS://A.B", "www.example", "visit example.com now", "a.in", "b.org/x", "c.net", "d.co"):
        assert screen_context([t])["url"] is True, t
    for t in ("INDIA", "CONTACT 98765", "x.INDIA", "SECOND", "A.COMPANY"):
        assert screen_context([t])["url"] is False, t


def test_excluded_lines_and_non_ascii_text_do_not_count():
    assert screen_context(["https://a.b"], excluded={0})["score"] == 0.0
    assert screen_context(["汉字汉字汉字汉字"])["text_lines"] == 0   # only ASCII letters / digits count


def _context_engine(side_raw):
    """A plate line (300x60) and a side line (120x20) whose raw OCR text is ``side_raw``."""
    return FakeEngine([_box(100, 100, 300, 60), _box(20, 330, 120, 20)], make_logp("TN71JE5250"),
                      lps={(20, 120): make_logp("IMAGESBING")},
                      raws={(60, 300): "TN71JE5250 https://www.x.com", (20, 120): side_raw})


def test_scene_reader_attaches_the_context_and_excludes_the_chosen_plate_lines():
    res = SceneReader(_context_engine("MARUTI SUZUKI"), multi_view=False).read(_scene_img(), domains=("plate",))
    assert res.plate.as_dict()["value"] == "TN71JE5250"
    # the URL is on the PLATE's own line: not scene context. Only the badge line counts, as text.
    assert res.screen_context == {**NO_SCREEN_CONTEXT, "text_lines": 1}
    assert res.as_dict()["screen_context"] == res.screen_context
    res2 = SceneReader(_context_engine("https://www.bing.com/images"), multi_view=False).read(_scene_img(), domains=("plate",))
    assert res2.screen_context["url"] is True and res2.screen_context["score"] == 1.0
    assert res2.screen_context["reasons"] == ["a web address (URL) is visible in the scene"]


def test_read_frames_keeps_the_strongest_context_and_first_of_equals():
    reader = SceneReader(_context_engine("https://www.bing.com/images"), multi_view=False)
    res = reader.read_frames([_scene_img(), _scene_img()], domains=("plate",))
    assert res.screen_context["url"] is True and len(res.per_image) == 2
    single = reader.read_frames([_scene_img()], domains=("plate",))
    assert single.screen_context == res.screen_context


def test_observe_falls_back_to_the_decoded_text_when_the_engine_has_no_raw_text():
    eng = FakeEngineNoGreedy([_box(100, 100, 300, 60)], make_logp("TN71JE5250"))
    (ln,), _ = SceneReader(eng, multi_view=False).observe(_scene_img())
    assert ln.raw0 == ln.text0 == "TN71JE5250"


# ============================================================== pattern-aware ranking (amendment to the slot candidates)
# score_c = log(peak_c + 1e-9) + grammar.log_prior(text with c in the slot) over the characters the grammar allows there
def _opts(entries):
    (e,) = entries
    return [o["char"] for o in e["options"]], e


def test_nine_character_plate_slot_after_the_rto_offers_only_letters():
    # 'HR88?8888' is 9 characters: index 4 is THE series letter. Equal evidence for the digit 8 and the letter B.
    lp = _lp_with_slot("HR88B8888", 4, {"?": 0.5, "B": 0.2, "8": 0.2, "D": 0.1})
    chars, e = _opts(slot_candidates("HR88?8888", lp, PLATE_GRAMMAR))
    assert e["position"] == 4 and chars == ["B", "D"] and all(c.isalpha() for c in chars)


def test_ten_character_plate_series_slots_offer_only_letters():
    lp = _lp_with_slot("MH12AB1234", 4, {"?": 0.5, "A": 0.2, "4": 0.2, "O": 0.1})
    chars, e = _opts(slot_candidates("MH12?B1234", lp, PLATE_GRAMMAR))              # index 4 of 10 characters: a letter
    assert e["position"] == 4 and chars[0] == "A" and all(c.isalpha() for c in chars)
    lp = _lp_with_slot("MH12AB1234", 5, {"?": 0.5, "B": 0.2, "8": 0.2, "I": 0.1})
    chars, e = _opts(slot_candidates("MH12A?1234", lp, PLATE_GRAMMAR))              # index 5 of 10 characters: a letter
    assert e["position"] == 5 and chars[0] == "B" and all(c.isalpha() for c in chars)


def test_ten_character_plate_number_slot_offers_only_digits():
    lp = _lp_with_slot("MH12AB1234", 8, {"?": 0.5, "3": 0.2, "B": 0.2, "8": 0.1})
    chars, e = _opts(slot_candidates("MH12AB12?4", lp, PLATE_GRAMMAR))
    assert e["position"] == 8 and chars[:2] == ["3", "8"] and all(c.isdigit() for c in chars)


def test_series_letters_I_and_O_are_penalised_by_the_pattern():
    # equal evidence for A / O / I: HSRP series letters skip I and O, the grammar prior says so
    lp = _lp_with_slot("MH12AB1234", 4, {"?": 0.2, "A": 0.3, "O": 0.3, "I": 0.3})
    chars, e = _opts(slot_candidates("MH12?B1234", lp, PLATE_GRAMMAR))
    assert chars[0] == "A" and e["options"][0]["p"] > 0.6 and e["suggested"] == "A"


VIN_BASE = "MAL907KE1E6602791"            # index 9 = model-year code, 11.. = serial digits


def test_vin_year_slot_prefers_a_valid_model_year_code_on_equal_evidence():
    # position 10 (index 9) is a model-year code: U, Z and 0 are not year codes, K is. Equal evidence for all four.
    assert not any(vin_check_digit_ok(VIN_BASE[:9] + c + VIN_BASE[10:]) for c in "KUZ0")        # no check-digit bonus muddies this
    lp = _lp_with_slot(VIN_BASE, 9, {"?": 0.3, "K": 0.2, "U": 0.2, "Z": 0.2, "0": 0.2})
    chars, e = _opts(slot_candidates(VIN_BASE[:9] + "?" + VIN_BASE[10:], lp, VIN_GRAMMAR))
    assert e["position"] == 9 and chars[0] == "K" and e["options"][0]["p"] > 0.6 and e["suggested"] == "K"
    assert all(o["p"] < 0.06 for o in e["options"][1:])                                        # the invalid year codes trail far behind


def test_vin_serial_slot_prefers_a_digit_on_equal_evidence():
    # index 14 of the serial: the rest of the serial is digits, and no check digit decides it either way
    assert not any(vin_check_digit_ok(VIN_BASE[:14] + c + VIN_BASE[15:]) for c in "7T")
    lp = _lp_with_slot(VIN_BASE, 14, {"?": 0.3, "7": 0.3, "T": 0.3})
    chars, e = _opts(slot_candidates(VIN_BASE[:14] + "?" + VIN_BASE[15:], lp, VIN_GRAMMAR))
    assert chars[:2] == ["7", "T"] and e["options"][0]["p"] > e["options"][1]["p"] > 0.3


def test_vin_check_digit_rewards_the_character_that_makes_it_pass():
    # same equal evidence at index 12, but only 'T' makes the check digit (index 8) valid: it outranks the preferred digit
    assert vin_check_digit_ok(VIN_BASE[:12] + "T" + VIN_BASE[13:]) and not vin_check_digit_ok(VIN_BASE[:12] + "7" + VIN_BASE[13:])
    lp = _lp_with_slot(VIN_BASE, 12, {"?": 0.3, "7": 0.3, "T": 0.3})
    chars, e = _opts(slot_candidates(VIN_BASE[:12] + "?" + VIN_BASE[13:], lp, VIN_GRAMMAR))
    assert chars[:2] == ["T", "7"]


def test_look_alike_digit_is_not_even_offered_in_a_letter_slot_and_the_letter_is_suggested():
    lp = _lp_with_slot("HR88B8888", 4, {"?": 0.15, "B": 0.40, "8": 0.40})              # B and 8 are equally likely to the network
    chars, e = _opts(slot_candidates("HR88?8888", lp, PLATE_GRAMMAR))
    assert chars == ["B"] and "8" not in chars and e["suggested"] == "B" and e["options"][0]["p"] > 0.85


def test_slot_scores_are_the_documented_formula():
    """p_c = softmax over the legal characters of log(peak_c + 1e-9) + grammar.log_prior(string with c in the slot)."""
    import math
    from vinplate.vocab import ALPHABET
    lp = _lp_with_slot("MH12AB1234", 4, {"?": 0.2, "A": 0.3, "O": 0.3, "I": 0.2, "B": 0.1})
    (e,) = slot_candidates("MH12?B1234", lp, PLATE_GRAMMAR)
    spike = np.exp(lp[(3 + 1) * 4])                                                    # the frame carrying the slot's evidence
    sc = {}
    for k, ch in enumerate(ALPHABET):
        s = "MH12" + ch + "B1234"
        if PLATE_GRAMMAR.is_valid(s) and PLATE_GRAMMAR.log_prior(s) is not None:
            sc[ch] = math.log(float(spike[k + 1]) + 1e-9) + PLATE_GRAMMAR.log_prior(s)
    top = max(sc.values())
    z = sum(math.exp(v - top) for v in sc.values())
    want = sorted(((math.exp(v - top) / z, ch) for ch, v in sc.items()), key=lambda t: (-t[0], ALPHABET.index(t[1])))[:3]
    assert [(o["char"], o["p"]) for o in e["options"]] == [(ch, round(p, 3)) for p, ch in want if p >= 0.05]
