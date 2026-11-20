"""LEGACY baseline: the regex extractor the backend used on RapidOCR text lines (kept only so the
benchmark can compare the new pipeline against what it replaced - nothing imports it at runtime)."""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------- VIN (ISO 3779)

VIN_CHARS = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
_VIN_LABEL = re.compile(r"\b(VIN|V\.I\.N\.?|CHASSIS(\s*NO\.?)?)\b\s*[:#.-]?")
_VIN_FIX = str.maketrans({"O": "0", "Q": "0", "I": "1"})
_VIN_VALUES = {
    **{str(d): d for d in range(10)},
    **dict(zip("ABCDEFGH", range(1, 9))),
    **dict(zip("JKLMN", range(1, 6))),
    "P": 7,
    "R": 9,
    **dict(zip("STUVWXYZ", range(2, 10))),
}
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def vin_check_digit_ok(vin: str) -> bool:
    total = sum(_VIN_VALUES[c] * w for c, w in zip(vin, _VIN_WEIGHTS))
    expected = "X" if total % 11 == 10 else str(total % 11)
    return vin[8] == expected


def find_vin(lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    # VIN stickers wrap, so allow runs of 1-3 adjacent lines; joining everything would
    # let a VIN be stitched together out of fragments of unrelated lines.
    for start in range(len(lines)):
        for n in (1, 2, 3):
            chunk = lines[start : start + n]
            if len(chunk) < n:
                break
            conf = min(ln["confidence"] for ln in chunk)
            text = " ".join(ln["text"] for ln in chunk).upper()
            labelled = bool(_VIN_LABEL.search(text))
            cleaned = _VIN_LABEL.sub(" ", text)
            compact = re.sub(r"[^A-Z0-9]", "", cleaned).translate(_VIN_FIX)
            for m in re.finditer(r"[A-HJ-NPR-Z0-9]{17,}", compact):
                run = m.group(0)
                for offset in range(len(run) - 16):
                    vin = run[offset : offset + 17]
                    check_ok = vin_check_digit_ok(vin)
                    # Across lines, only trust a read the sticker label or the check digit backs up.
                    if n > 1 and not labelled and not check_ok:
                        continue
                    score = conf + (0.2 if check_ok else 0)
                    score -= 0.1 if len(run) > 17 else 0
                    score -= 0.3 * (n - 1)
                    if best is None or score > best[1]:
                        best = (vin, score, conf, len(run))
    if best is None:
        return None
    return {
        "value": best[0],
        "confidence": round(best[2], 3),
        "score": round(best[1], 3),
        "check_digit_ok": vin_check_digit_ok(best[0]),
        # A real VIN is exactly 17; a longer run means a character was added or misread.
        "read_length": best[3],
    }


# ------------------------------------------------------- Registration (India)

# State code, RTO number, series letters, 4-digit number; plus Bharat (BH) series.
PLATE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")
PLATE_BH = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")
_TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})

# RTO state / UT codes. Every Indian plate starts with one, so this is the strongest
# signal we have for telling a real plate from a coincidentally plate-shaped token.
STATE_CODES = frozenset(
    "AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TG TN TR TS UK UP WB".split()
)

_PLATE_MIN = 7   # XX 1 1234
_PLATE_MAX = 11  # XX 11 ABC 1234


def _coerce_plate(s: str) -> tuple[str, int] | None:
    """Repair one compact token into a legal plate; returns (plate, characters changed)."""
    if PLATE_BH.match(s):
        return s, 0
    # Positional repair: 2 letters, 1-2 digits, 0-3 letters, 4 digits.
    head = s[:2].translate(_TO_ALPHA)
    tail = s[-4:].translate(_TO_DIGIT)
    mid = s[2:-4]
    best = None
    for rto_len in (2, 1):
        rto = mid[:rto_len].translate(_TO_DIGIT)
        series = mid[rto_len:].translate(_TO_ALPHA)
        cand = head + rto + series + tail
        if not PLATE.match(cand):
            continue
        repairs = sum(a != b for a, b in zip(cand, s))
        if best is None or repairs < best[1]:
            best = (cand, repairs)
    return best


def _best_plate(raw: str) -> tuple[str, float] | None:
    """Highest-scoring plate inside `raw`, with a score for how plausible the read is.

    Plates are read with clutter around them ("IND", a "TC/20" temporary marking, the
    hologram line), so the token as a whole rarely matches - scan every window and
    prefer the one that needed the fewest character repairs.
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper()).removeprefix("IND")
    best = None
    for length in range(min(_PLATE_MAX, len(s)), _PLATE_MIN - 1, -1):
        for start in range(len(s) - length + 1):
            found = _coerce_plate(s[start : start + length])
            if found is None:
                continue
            plate, repairs = found
            state_ok = plate[:2] in STATE_CODES or plate[2:4] == "BH"
            # Without a state code, only trust a token that is a plate end to end as read;
            # a slice of longer text (a VIN) or digits bent into shape (dial numbers) is
            # a coincidence.
            if not state_ok and (length != len(s) or repairs):
                continue
            score = (1.0 if state_ok else 0.0) - 0.3 * repairs + 0.02 * length
            if best is None or score > best[1]:
                best = (plate, score)
    return best


def format_plate(plate: str) -> str:
    if m := PLATE.match(plate):
        return " ".join(g for g in m.groups() if g)
    if m := PLATE_BH.match(plate):
        return " ".join(m.groups())
    return plate


def normalize_plate(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def find_plate(lines: list[dict[str, Any]], vin: str | None = None) -> dict[str, Any] | None:
    # Plates are often split over two OCR boxes/rows, so try runs of 1-3 lines.
    best = None
    for i in range(len(lines)):
        for n in (1, 2, 3):
            chunk = lines[i : i + n]
            if len(chunk) < n:
                break
            text = "".join(ln["text"] for ln in chunk)
            # Text already read as the VIN can't also be the plate.
            if vin and vin in normalize_plate(text).translate(_VIN_FIX):
                continue
            found = _best_plate(text)
            if found is None:
                continue
            plate, shape = found
            conf = min(ln["confidence"] for ln in chunk)
            score = shape + conf
            if best is None or score > best[1]:
                best = (plate, score, conf)
    if best is None:
        return None
    return {
        "value": best[0],
        "display": format_plate(best[0]),
        "confidence": round(best[2], 3),
        "score": round(best[1], 3),
    }


def extract_fields(lines: list[dict[str, Any]]) -> dict[str, Any]:
    vin = find_vin(lines)
    plate = find_plate(lines, vin["value"] if vin else None)
    return {"vin": vin, "plate": plate}
