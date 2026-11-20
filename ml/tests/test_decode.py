import numpy as np
import pytest

from vinplate.decode import ctc_align, ctc_loglik, fuse_frames, greedy, prefix_beam_search, read_line
from vinplate.grammar import PLATE_GRAMMAR, VIN_GRAMMAR
from vinplate.grammar import vin as vin_mod
from vinplate.testing import make_logp
from vinplate.vocab import encode_text


def test_greedy_and_loglik_agree_on_clean_input():
    lp = make_logp("MH12AB1234")
    text, conf = greedy(lp)
    assert text == "MH12AB1234" and min(conf) > 0.9
    assert ctc_loglik(lp, encode_text("MH12AB1234")) > ctc_loglik(lp, encode_text("MH12AB1235")) + 5


def test_repeated_characters_need_a_blank():
    # "AA" must survive CTC collapse because the simulator inserts a blank between chars
    lp = make_logp("MH12AA1100")
    assert greedy(lp)[0] == "MH12AA1100"
    assert read_line(lp, PLATE_GRAMMAR).text == "MH12AA1100"


def test_alignment_confidence_tracks_evidence():
    lp = make_logp("MH12AB1234", flat=(4,))
    spans = ctc_align(lp, encode_text("MH12AB1234"))
    confs = [s[2] for s in spans]
    assert confs[4] < 0.1 and min(c for i, c in enumerate(confs) if i != 4) > 0.9


def test_clean_plate_is_accepted_without_review():
    r = read_line(make_logp("KA01AB1234"), PLATE_GRAMMAR)
    assert r.text == "KA01AB1234" and r.valid and r.status == "accept"
    assert r.inferred == [] and r.suspected_gaps == [] and r.grammar_cost < 0.5


@pytest.mark.parametrize("truth,pos,wrong", [
    ("MH12AB1234", 8, "B"),    # 3 read as B   -> digit slot cannot hold B
    ("MH12AB1034", 7, "O"),    # 0 read as O   -> digit slot cannot hold O
    ("MH12AB1234", 2, "I"),    # 1 read as I   -> RTO slot cannot hold I
    ("DL1CAB4521", 9, "I"),    # 1 read as I
    ("MH12AB5678", 6, "S"),    # 5 read as S
])
def test_grammar_repairs_letter_digit_confusions(truth, pos, wrong):
    lp = make_logp(truth, confuse={pos: (wrong, 0.55)})
    assert greedy(lp)[0] != truth                       # the raw network is wrong ...
    r = read_line(lp, PLATE_GRAMMAR)
    assert r.text == truth and r.valid                  # ... the constrained decode is not


def test_smeared_stress_case_does_not_hallucinate_an_extra_series_letter():
    # Harsher posteriors where the wrong and right character both span the same frames:
    # CTC can legally emit both ("ABS5678"). The 3-letter-series prior (non-Delhi) must
    # keep the decoder from taking that insertion.
    lp = make_logp("MH12AB5678", confuse={6: ("S", 0.55)}, smeared=True)
    assert read_line(lp, PLATE_GRAMMAR).text == "MH12AB5678"


def test_unknown_state_code_resolves_to_real_state():
    lp = make_logp("MH12AB1234", confuse={1: ("X", 0.55)})  # "MX" is not a state code
    assert greedy(lp)[0].startswith("MX")
    assert read_line(lp, PLATE_GRAMMAR).text == "MH12AB1234"


def test_confident_legal_misread_is_not_overridden():
    # If the network is *sure* about a legal but different string ("MP"), the grammar must
    # not "fix" it. Constraints resolve ambiguity; they do not overrule strong evidence.
    lp = make_logp("MH12AB1234", confuse={1: ("P", 0.9)})
    assert read_line(lp, PLATE_GRAMMAR).text == "MP12AB1234"


def test_confident_illegal_read_is_repaired_but_never_silently_accepted():
    # "KH" is not a state code. The nearest legal reading is "MH", but the network was
    # confident about K, so the repaired character must be reported as inferred and the
    # reading must not be auto-accepted.
    lp = make_logp("MH12AB1234", confuse={0: ("K", 0.9)})
    r = read_line(lp, PLATE_GRAMMAR)
    assert r.text == "MH12AB1234" and 0 in r.inferred and r.status != "accept"


@pytest.mark.parametrize("pos", [4, 5, 6, 7, 8, 9])
def test_washed_out_character_is_never_silently_dropped_or_accepted(pos):
    # The dangerous failure: CTC drops an unreadable character and the grammar happily
    # accepts the shorter, legal-looking plate. It must surface as inferred or as a
    # suspected gap, and the reading must go to review - never to accept.
    truth = "MH12AB1234"
    r = read_line(make_logp(truth, flat=(pos,)), PLATE_GRAMMAR)
    assert r.status != "accept"
    assert pos in r.inferred or r.suspected_gaps, (r.text, r.inferred, r.suspected_gaps)
    if len(r.text) == len(truth):                        # reconstructed: the rest must be right
        assert all(a == b for i, (a, b) in enumerate(zip(r.text, truth)) if i != pos)


def test_modern_layout_is_reconstructed_for_a_washed_out_digit():
    # A 4-digit number is near-certain on an HSRP plate, so a washed-out digit is *inferred
    # in place* (flagged) rather than dropped.
    r = read_line(make_logp("MH12AB1234", flat=(7,)), PLATE_GRAMMAR)
    assert len(r.text) == 10 and r.text[:7] == "MH12AB1" and r.text[8:] == "34"
    assert 7 in r.inferred and r.status == "review"


def test_vin_io_q_is_resolved_by_alphabet():
    truth = "MA3EWB22SPA123450"
    lp = make_logp(truth, confuse={16: ("O", 0.55)})
    assert greedy(lp)[0].endswith("O")
    assert read_line(lp, VIN_GRAMMAR).text == truth


def test_vin_check_digit_recovers_one_unreadable_char_when_mandated(monkeypatch):
    # Where the check digit is mandatory (e.g. North-American VINs), one washed-out
    # character is recoverable: the check pins its value (up to look-alike letters).
    truth = "1M8GDM9AXKP042788"
    lp = make_logp(truth, flat=(11,))
    monkeypatch.setattr(vin_mod, "CHECK_DIGIT_BONUS", 6.0)
    r = read_line(lp, VIN_GRAMMAR)
    assert r.text == truth
    assert r.inferred == [11]          # still reported as inferred, never passed off as read
    assert not r.needs_review          # one inferred char is within budget and the check pins it


def test_vin_check_digit_is_only_weak_evidence_by_default():
    # Indian OEM VINs often fail/ignore the ISO check digit, so by default an unreadable char
    # is not presented as a confident reading even when the check happens to pin it.
    truth = "1M8GDM9AXKP042788"
    r = read_line(make_logp(truth, flat=(11,)), VIN_GRAMMAR)
    assert r.needs_review and r.inferred == [11]
    assert truth in [t for t, _ in r.n_best]


def test_two_unreadable_chars_are_never_enumerated():
    truth = "1M8GDM9AXKP042788"
    r = read_line(make_logp(truth, flat=(11, 14)), VIN_GRAMMAR)
    assert r.status != "accept"
    assert len(r.inferred) + len(r.suspected_gaps) >= 2, (r.text, r.inferred, r.suspected_gaps)


def test_multi_frame_fusion_recovers_what_no_single_frame_has():
    truth = "MH12AB1234"
    a = make_logp(truth, flat=(4,))                     # glare on frame A hides char 4
    b = make_logp(truth, flat=(7,))                     # glare on frame B hides char 7
    for single in (a, b):
        assert read_line(single, PLATE_GRAMMAR).status != "accept"
    r = fuse_frames([a, b], PLATE_GRAMMAR)
    assert r.text == truth and r.valid
    assert r.inferred == [] and r.suspected_gaps == [] and r.status == "accept"


def test_fusion_survives_one_destroyed_frame():
    truth = "KA05MN9876"
    good = [make_logp(truth), make_logp(truth, confuse={3: ("S", 0.4)})]
    junk = make_logp("ZZ99ZZ9999")
    r = fuse_frames(good + [junk], PLATE_GRAMMAR)
    assert r.text == truth


def test_noise_is_rejected_not_forced_into_a_plausible_plate():
    lp = make_logp("QQQQQQQQQQQQQQQQQQQQQQQQ")          # nothing a plate grammar can accept
    r = read_line(lp, PLATE_GRAMMAR)
    # The grammar can always force *some* legal string out of noise (e.g. KL0QQQ0000). What
    # protects us is the grammar cost: how far it had to bend the network's own reading.
    assert r.status == "reject" and r.grammar_cost > 12


def test_beam_search_without_grammar_matches_greedy_on_clean_input():
    lp = make_logp("MH12AB1234")
    h = prefix_beam_search(lp, None)
    assert h[0].text == "MH12AB1234"


# ------------------------------------------------- explicit unreadable ('?') class
def test_network_abstention_is_reported_and_masked_not_hallucinated():
    r = read_line(make_logp("MH12AB1234", unreadable=(7,)), PLATE_GRAMMAR)
    assert r.text == "MH12AB1?34" and r.inferred == [7]
    assert r.masked == "MH12AB1?34" and r.status == "review"


def test_abstained_plate_digit_is_not_guessed():
    # nothing in a plate's grammar can pin one digit, so the '?' must survive to the output
    r = read_line(make_logp("DL1CAB4521", unreadable=(8,)), PLATE_GRAMMAR)
    assert "?" in r.text and r.status != "accept"
    assert len(r.n_best) > 1                       # candidate fills are visible to a reviewer


def test_two_abstentions_are_never_filled():
    r = read_line(make_logp("MH12AB1234", unreadable=(6, 8)), PLATE_GRAMMAR)
    assert r.text.count("?") == 2 and r.status == "review"


def test_abstained_state_code_is_resolved_only_as_far_as_legality_allows():
    # "?H" can be MH only (no other state code ends in H except... check) - the grammar
    # keeps it as '?' unless the choice is unique, and always flags it.
    r = read_line(make_logp("MH12AB1234", unreadable=(0,)), PLATE_GRAMMAR)
    assert r.status != "accept" and 0 in r.inferred


def test_vin_abstention_is_filled_only_when_the_check_digit_is_mandatory(monkeypatch):
    truth = "1M8GDM9AXKP042788"
    lp = make_logp(truth, unreadable=(11,))
    default = read_line(lp, VIN_GRAMMAR)
    assert default.text[11] == "?" and default.status != "accept"       # weak evidence: stay honest
    assert truth in [t for t, _ in default.n_best]                        # ...but the answer is on the list
    monkeypatch.setattr(vin_mod, "CHECK_DIGIT_BONUS", 6.0)
    pinned = read_line(lp, VIN_GRAMMAR)
    assert pinned.text == truth and pinned.inferred == [11]               # still flagged as inferred
    assert pinned.masked[11] == "?"                                       # and masked for display


# --------------------------------------------- gaps: a group gap is not a dropped character
def test_group_gaps_are_not_flagged_but_a_missing_character_is():
    from vinplate.decode import suspected_gaps
    pitch = 4
    group_gap = [(s, s, 0.9) for s in (0, 4, 8, 14, 18, 22)]            # 1.5 pitches at the group boundary
    assert suspected_gaps(group_gap, 28) == []
    wide_group_gap = [(s, s, 0.9) for s in (0, 4, 8, 15, 19, 23)]       # 1.75 pitches: still a printed space
    assert suspected_gaps(wide_group_gap, 28) == []
    dropped = [(s, s, 0.9) for s in (0, 4, 8, 16, 20, 24)]              # 2.0 pitches: a character is missing
    assert suspected_gaps(dropped, 28) == [3]
