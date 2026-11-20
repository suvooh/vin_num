import random

import pytest

from vinplate.grammar import (
    PLATE_GRAMMAR,
    VIN_GRAMMAR,
    parse_plate,
    parse_vin,
    vin_check_digit_ok,
    vin_year_candidates,
)
from vinplate.vocab import CHARS


@pytest.mark.parametrize("plate", [
    "MH12AB1234", "KA01A1234", "DL1CAB1234", "DL3CAB1234", "TN07CF1234", "UP16BT0001",
    "22BH1234AA", "21BH0001A", "MH021234", "199CD1", "03E123456X", "TS09EA1234", "TG07AB1234",
])
def test_valid_plates(plate):
    assert PLATE_GRAMMAR.is_valid(plate), plate
    assert parse_plate(plate) is not None


@pytest.mark.parametrize("plate", [
    "XX12AB1234",      # no such state code
    "MH12AB12345",     # 5-digit number
    "MH12ABCD1234",    # 4 series letters
    "MH12AB", "1234", "MH", "", "MH12AB123A",
    "20BH1234AA",      # parses structurally, but BH series did not exist before 2021 (soft prior, still valid)
])
def test_invalid_plates(plate):
    if plate == "20BH1234AA":
        assert PLATE_GRAMMAR.is_valid(plate)  # layout is legal; the prior penalises it
        assert parse_plate(plate)["log_prior"] < parse_plate("22BH1234AA")["log_prior"]
        return
    assert not PLATE_GRAMMAR.is_valid(plate), plate


def test_plate_display_and_fields():
    assert parse_plate("MH12AB1234")["display"] == "MH 12 AB 1234"
    assert parse_plate("DL1CAB1234")["display"] == "DL 1C AB 1234"
    p = parse_plate("22BH1234AA")
    assert p["layout"] == "bharat" and p["display"] == "22 BH 1234 AA"
    p = parse_plate("KA01A1234")
    assert (p["state_code"], p["rto"], p["series"], p["number"]) == ("KA", "01", "A", "1234")
    assert parse_plate("TS09EA1234")["legacy_state_code"] is True


@pytest.mark.parametrize("plate,display", [
    ("DLR012345", "DLR 01 2345"),    # the format on the user's infographic (illustrative, low prior)
    ("DLR12345", "DLR 12 345"),      # longest-first segmentation: 2 + 3 digits
    ("DLR1234", "DLR 1 234"),
    ("DLR0123", "DLR 0 123"),
])
def test_dealer_template(plate, display):
    p = parse_plate(plate)
    assert PLATE_GRAMMAR.is_valid(plate)
    assert p["layout"] == "dealer" and p["display"] == display
    assert p["series"] == "DLR" and p["rto"] + p["number"] == plate[3:]
    assert p["log_prior"] < parse_plate("MH12AB1234")["log_prior"]       # low prior: never preferred over the real formats


@pytest.mark.parametrize("plate", [
    "DLR", "DLR1", "DLR12", "DLR123",          # fewer than 4 digits after DLR
    "DLR1234567",                              # more than 6
    "012345", "01DLR2345", "DL012345X",        # a bare digit string / DLR not at the front
    "DLRO12345", "D1R012345",                  # look-alikes are not DLR
])
def test_dealer_template_rejects(plate):
    p = parse_plate(plate)
    assert p is None or p["layout"] != "dealer", plate


def test_dealer_template_does_not_disturb_standard_plates():
    assert parse_plate("DL12345")["layout"] == "standard"            # Delhi, 1 + 4 digits: still a normal plate
    assert parse_plate("MH12AB1234")["layout"] == "standard"
    assert parse_plate("DL1CAB1234")["display"] == "DL 1C AB 1234"


def test_modern_layout_preferred_over_legacy():
    modern = parse_plate("MH12AB1234")["log_prior"]
    assert modern > parse_plate("MH021234")["log_prior"]            # no series letters
    assert modern > parse_plate("MH12AB123")["log_prior"]           # 3-digit number
    assert modern > parse_plate("MH12IO1234")["log_prior"]          # I/O in series


def test_dfa_export_matches_nfa():
    dfa = PLATE_GRAMMAR.to_dfa(CHARS)
    acc = set(dfa["accepting"])
    trans = dfa["transitions"]

    def walk(s):
        q = dfa["start"]
        for ch in s:
            q = trans[str(q)].get(ch)
            if q is None:
                return False
        return q in acc

    rng = random.Random(0)
    samples = ["MH12AB1234", "22BH1234AA", "DL1CAB1234", "MH12AB", "XX1234"]
    for _ in range(3000):
        n = rng.randint(1, 12)
        pool = rng.choice([CHARS, "MHKADL0123456789AB"])
        samples.append("".join(rng.choice(pool) for _ in range(n)))
    for s in samples:
        assert walk(s) == PLATE_GRAMMAR.is_valid(s), s
    assert len(trans) < 2000  # stays tiny enough to ship as a table


# ------------------------------------------------------------------------------ VIN
def test_check_digit_known_vector():
    assert vin_check_digit_ok("1M8GDM9AXKP042788")        # canonical ISO/FMVSS example
    assert not vin_check_digit_ok("1M8GDM9A1KP042788")
    assert not vin_check_digit_ok("1M8GDM9AXKP04278")     # wrong length
    assert not vin_check_digit_ok("1M8GDM9AXKP04278I")    # I is not a VIN character


def test_year_codes():
    assert 2023 in vin_year_candidates("MA3EWB22SPA123456")      # position 10 = P
    assert vin_year_candidates("A" * 9 + "S" + "0" * 7) == [1995, 2025]
    assert vin_year_candidates("A" * 9 + "T" + "0" * 7) == [1996, 2026]
    assert vin_year_candidates("A" * 9 + "Z" + "0" * 7) == []   # Z is not a year code
    assert vin_year_candidates("A" * 9 + "0" + "0" * 7) == []   # nor is 0
    assert vin_year_candidates("A" * 9 + "5" + "0" * 7) == [2005]


@pytest.mark.parametrize("vin", ["MA3EWB22SPA123456", "1M8GDM9AXKP042788", "ME4KC091XPA123456"])
def test_valid_vins(vin):
    assert VIN_GRAMMAR.is_valid(vin), vin


@pytest.mark.parametrize("vin", ["MA3EWB22SPA1234567", "", "MA3EW", "MA3EWB22SPA12345!"])
def test_invalid_vins(vin):
    assert not VIN_GRAMMAR.is_valid(vin)


def test_short_chassis_is_legacy_not_a_vin():
    # Old Indian vehicles carry 6-16 character chassis numbers: accepted, but never as a
    # strict VIN, and with a heavy prior so a real 17-char reading always wins.
    p = parse_vin("MA3EWB22SPA12345")
    assert p["layout"] == "chassis_legacy" and p["short_chassis"]


def test_vin_with_ioq_falls_to_legacy_with_penalty():
    strict = parse_vin("MA3EWB22SPA123456")
    odd = parse_vin("MA3EWB2OSPA123456")  # contains the letter O
    assert strict["layout"] == "vin17" and odd["layout"] == "chassis_legacy"
    assert odd["log_prior"] < strict["log_prior"] - 3


def test_label_prefix_is_stripped():
    p = parse_vin("VINMA3EWB22SPA123456")
    assert p["label"] == "VIN" and p["vin"] == "MA3EWB22SPA123456"


def test_india_wmi_and_known_maker():
    p = parse_vin("MA3EWB22SPA123456")
    assert p["india_wmi"] and p["known_maker"] == "Maruti Suzuki"
    p = parse_vin("1M8GDM9AXKP042788")
    assert not p["india_wmi"] and p["check_digit_ok"]


# ----------------------------------------------------------- unreadable '?' wildcard
def test_wildcard_keeps_prefixes_alive_only_while_a_completion_is_legal():
    assert PLATE_GRAMMAR.is_valid("MH12AB?234")
    assert PLATE_GRAMMAR.is_valid("??12AB1234")        # some state code fits
    assert PLATE_GRAMMAR.is_valid("MH??AB1234")
    assert not PLATE_GRAMMAR.is_valid("MX12AB?234")    # no completion of MX is a state
    assert not PLATE_GRAMMAR.is_valid("MH12AB?23456")  # number would be 5 digits


def test_parse_with_unreadable_characters_does_not_crash():
    p = parse_plate("MH12AB?234")
    assert p["number"] == "?234" and p["state_code"] == "MH"
    assert parse_plate("22BH?234AA")["layout"] == "bharat"
    v = parse_vin("MA3EWB22S?A123456")
    assert v["layout"] == "vin17" and v["check_digit_ok"] is None


def test_unreadable_character_is_neutral_for_vin_checks():
    clean = parse_vin("MA3EWB22SPA123456")["log_prior"]
    with_unk = parse_vin("MA3EWB22S?A123456")["log_prior"]
    assert abs(with_unk - clean) < 2.0   # no year penalty / check bonus swing from the unknown
