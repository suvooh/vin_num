"""LEGACY extractor (the old backend): regression cases from real RapidOCR output. Kept because the
benchmark uses it as the baseline; the cases also document what a text-only approach gets right."""

from eval.legacy_extract import extract_fields, vin_check_digit_ok


def _lines(*pairs):
    return [{"text": t, "confidence": c} for t, c in pairs]


CLUSTER = _lines(("120 160", 0.742), ("60", 0.663), ("180", 0.704), ("15.010", 0.653), ("1099", 0.572))
VIN_STICKER = _lines(("7J3ZZ56T7834500003", 0.871))
PLATE = _lines(("UP16C0059TC/20", 0.846))


def _merge(best, cand, i):
    if cand and (best is None or cand["score"] > best["score"]):
        return {**cand, "image": i}
    return best


def _capture(*photos):
    fields = {"vin": None, "plate": None}
    for i, lines in enumerate(photos):
        found = extract_fields(lines)
        for key in fields:
            fields[key] = _merge(fields[key], found[key], i)
    return fields


def test_capture_picks_each_field_from_the_right_photo():
    fields = _capture(CLUSTER, VIN_STICKER, PLATE)
    assert fields["plate"]["value"] == "UP16C0059"
    assert fields["plate"]["image"] == 2
    assert fields["vin"]["value"] == "7J3ZZ56T783450000"
    assert fields["vin"]["read_length"] == 18


def test_vin_text_is_not_read_as_a_plate():
    assert extract_fields(VIN_STICKER)["plate"] is None


def test_dial_numbers_are_not_read_as_a_plate():
    assert extract_fields(CLUSTER)["plate"] is None


def test_valid_plates_read():
    assert extract_fields(_lines(("22BH1234AA", 0.8)))["plate"]["value"] == "22BH1234AA"
    plate = extract_fields(_lines(("MH12AB1234", 0.8)))["plate"]
    assert plate["value"] == "MH12AB1234" and plate["display"] == "MH 12 AB 1234"


def test_labelled_vin_with_valid_check_digit():
    vin = "1M8GDM9AXKP042788"
    assert vin_check_digit_ok(vin)
    found = extract_fields(_lines(("VIN: " + vin, 0.9)))["vin"]
    assert found["value"] == vin and found["check_digit_ok"]
