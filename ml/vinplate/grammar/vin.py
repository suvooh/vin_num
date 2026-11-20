"""VIN / chassis-number grammar (ISO 3779 structure, tuned for Indian vehicles).

Facts this module relies on, and how confident each one is:

* 17 characters, alphabet A-Z 0-9 minus I, O, Q (ISO 3779). Certain.
* Position 10 is the model-year code from a fixed 30-letter cycle. Certain for the code
  table; *whether every Indian OEM follows it* is not guaranteed -> soft prior only.
* Position 9 is a check digit ONLY where mandated (North America, China). Indian OEM VINs
  frequently do not satisfy it, so a failed check is never a rejection. A *passing* check
  is weak positive evidence (``CHECK_DIGIT_BONUS``, derived below), nothing more.
* India-made vehicles carry a WMI starting MA..ME (ISO 3780 allocation). Imported
  vehicles legitimately start elsewhere, so this is a bonus, not a gate.
* Older Indian vehicles can carry short (6-16 character) chassis numbers: a heavily
  down-weighted legacy template handles them.

Check-digit bonus derivation: a wrong read passes the ISO check ~1/11 of the time; a
correct read of a check-digit VIN passes always. If a fraction f of the fleet uses check
digits, LR = (f + (1-f)/11) / (1/11); at f=0.2 that is ~3.0 -> +1.1 nats. We use +1.0. It
must be recalibrated on a labelled sample of the real fleet (``eval/`` does this).
"""

from __future__ import annotations

from datetime import date

from vinplate.grammar.base import Choice, Cls, Grammar, Template
from vinplate.vocab import UNK

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # no I, O, Q
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LABELS = ("", "VIN", "VINNO", "CHASSIS", "CHASSISNO", "CHASISNO", "CHASISSNO")

CHECK_DIGIT_BONUS = 1.0
INDIA_WMI_BONUS = 0.8
KNOWN_WMI_BONUS = 0.4

# Named WMIs. verified=True only where I confirmed it against the NHTSA vPIC registry in
# this session (MAJ, MD2, ME3); the rest are recalled and must not be treated as fact.
KNOWN_WMI: dict[str, tuple[str, bool]] = {
    "MAJ": ("Ford India", True), "MD2": ("Bajaj Auto", True), "ME3": ("Royal Enfield", True),
    "MA1": ("Mahindra", False), "MA3": ("Maruti Suzuki", False), "MAK": ("Honda Cars India", False),
    "MAL": ("Hyundai India", False), "MAT": ("Tata Motors", False), "MBH": ("Maruti Suzuki", False),
    "MBJ": ("Toyota Kirloskar", False), "MBL": ("Hero MotoCorp", False),
    "ME4": ("Honda Motorcycle & Scooter India", False), "MD6": ("TVS Motor", False),
}

_YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"  # 30 codes, 1980/2010 = A
_VALUES = {
    **{str(d): d for d in range(10)},
    **dict(zip("ABCDEFGH", range(1, 9))),
    **dict(zip("JKLMN", range(1, 6))),
    "P": 7, "R": 9,
    **dict(zip("STUVWXYZ", range(2, 10))),
}
_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def vin_check_digit_ok(vin: str) -> bool:
    """ISO 3779 / FMVSS 115 check digit. False for anything that is not a clean 17-char VIN."""
    if len(vin) != 17 or any(c not in _VALUES for c in vin):
        return False
    total = sum(_VALUES[c] * w for c, w in zip(vin, _WEIGHTS))
    r = total % 11
    return vin[8] == ("X" if r == 10 else str(r))


def vin_year_candidates(vin: str) -> list[int]:
    """Model years consistent with position 10 (30-year cycle), oldest first."""
    if len(vin) < 10 or vin[9] not in _YEAR_CODES:
        return []
    c = vin[9]
    if c.isdigit():  # '1'..'9' -> 2001..2009 (the 1971-79 reading predates the standard)
        years = [2000 + int(c)]
    else:  # A=1980/2010 ... Y=2000/2030
        i = _YEAR_CODES.index(c)
        years = [1980 + i, 2010 + i]
    now = date.today().year + 1  # model years run a little ahead of the calendar
    return [y for y in years if 1981 <= y <= now]


def _strict_score(p: dict[str, str]) -> float:
    v = p["vin"]
    s = 0.0
    # An unreadable ('?') character carries no evidence either way: skip the checks that
    # would need it rather than counting it for or against the reading.
    if v[9] != UNK:
        if v[9] not in _YEAR_CODES or not vin_year_candidates(v):
            s -= 2.5  # not a year code, or a model year in the future / before 1981
    if v[0] == "M" and v[1] in "ABCDE":
        s += INDIA_WMI_BONUS
    if v[:3] in KNOWN_WMI:
        s += KNOWN_WMI_BONUS
    if all(c.isdigit() or c == UNK for c in v[11:]):
        s += 0.4
    if UNK not in v and vin_check_digit_ok(v):
        s += CHECK_DIGIT_BONUS
    return s


def _build() -> Grammar:
    return Grammar(
        "vin",
        [
            Template(
                "vin17",
                (Choice("label", LABELS), Cls("vin", VIN_CHARS, 17, 17)),
                log_prior=0.0,
                score=_strict_score,
            ),
            # Legacy / non-conformant chassis numbers (and a 17-char read that still
            # contains I/O/Q): allowed, but a strict ISO reading wins whenever it is
            # remotely competitive.
            Template(
                "chassis_legacy",
                (Choice("label", LABELS), Cls("chassis", ALNUM, 6, 17)),
                log_prior=-6.0,
            ),
        ],
    )


VIN_GRAMMAR = _build()


def parse_vin(text: str) -> dict | None:
    best = VIN_GRAMMAR.best_parse(text)
    if best is None:
        return None
    tpl, p, prior = best
    out: dict = {"layout": tpl.name, "log_prior": round(prior, 3), "label": p["label"] or None}
    if tpl.name == "vin17":
        v = p["vin"]
        out.update(
            vin=v, wmi=v[:3], vds=v[3:9], vis=v[9:],
            india_wmi=(v[0] == "M" and v[1] in "ABCDE"),
            known_maker=KNOWN_WMI.get(v[:3], (None, False))[0],
            check_digit_ok=(vin_check_digit_ok(v) if UNK not in v else None),
            year_candidates=vin_year_candidates(v),
        )
    else:
        out.update(vin=p["chassis"], short_chassis=len(p["chassis"]) < 17)
    return out
