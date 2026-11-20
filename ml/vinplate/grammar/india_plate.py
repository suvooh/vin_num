"""Indian registration-mark grammar.

Layouts covered (all are *text* layouts; colour/size/HSRP-ness is the vision model's job):

  standard   SS RR [A..] NNNN     MH 12 AB 1234, KA 01 A 1234, DL 1C AB 1234 (Delhi's class
                                  letter is the first series letter), legacy 1-3 digit
                                  numbers and series-less marks such as MH 02 1234
  bharat     YY BH NNNN XX        22 BH 1234 AA   (Bharat-series, valid in every state)
  diplomatic NNN CD|CC|UN N..     199 CD 1
  dealer     DLR NN NNNN          DLR 01 2345 (illustrative, low prior)
  military   NN L NNNNNN L        03E 123456X     (the up-arrow glyph is not in the alphabet)

State/UT codes include recently retired ones that are still on the road (OR, DN, UA, CT,
TS) because an old plate is still a real plate. Soft priors (``score``) prefer the modern
4-digit/2-letter layout so a clean read of a legacy plate still wins when it is the only
consistent explanation. Nothing here is a *hard* reject beyond the layout itself.

NOT encoded, deliberately: which RTO numbers exist in which state. That table is large,
changes, and I will not recall it from memory; ``RTO_ALLOWED`` is the hook to load one
(state -> set of ints) once it is sourced and verified.
"""

from __future__ import annotations

from vinplate.grammar.base import Choice, Cls, Grammar, Template

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

STATE_NAMES: dict[str, str] = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman & Diu / Dadra & Nagar Haveli", "DL": "Delhi", "GA": "Goa",
    "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand",
    "JK": "Jammu & Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
    "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
    "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim",
    "TG": "Telangana", "TN": "Tamil Nadu", "TR": "Tripura", "UK": "Uttarakhand",
    "UP": "Uttar Pradesh", "WB": "West Bengal",
    # retired / legacy codes still seen on older plates
    "CT": "Chhattisgarh (old)", "DN": "Dadra & Nagar Haveli (old)", "OR": "Odisha (old)",
    "TS": "Telangana (old)", "UA": "Uttarakhand (old)",
}
LEGACY_STATE_CODES = frozenset({"CT", "DN", "OR", "TS", "UA"})
STATE_CODES = tuple(sorted(STATE_NAMES))

# Optional verified table: state code -> allowed RTO numbers. None = no constraint.
RTO_ALLOWED: dict[str, set[int]] | None = None

# Series letters skip I and O (and, in many states, Q) to avoid 1/0 look-alikes.
_SERIES_ODD = set("IOQ")


def _int_or_none(x: str) -> int | None:
    return int(x) if x.isdigit() else None  # None when the digits are unreadable ('?')


def _standard_score(p: dict[str, str]) -> float:
    s = 0.0
    n = len(p["number"])
    # HSRP-era numbers are always zero-padded to 4 digits (0007, not 7). Short numbers exist
    # only on pre-HSRP plates, and a washed-out digit is indistinguishable from an absent one
    # in CTC output, so shortness must be expensive: wrongly accepting a shortened plate is
    # worse than flagging a genuine legacy plate for review.
    s += {4: 0.0, 3: -4.0, 2: -5.0, 1: -6.0}[n]
    s += {2: 0.0, 1: -0.3}[len(p["rto"])]
    s += {0: -1.5, 1: -0.2, 2: 0.0}.get(len(p["series"]), 0.0)
    if len(p["series"]) == 3:
        # Three series letters are legitimate in Delhi, where the vehicle-class letter is
        # printed against the RTO (DL 1C AB 1234). Elsewhere it is almost always an
        # inserted/misread character, so make it expensive but not impossible.
        s += -0.4 if (p["state"] == "DL" and len(p["rto"]) == 1) else -2.5
    rto = _int_or_none(p["rto"])
    if rto == 0:
        s -= 6.0
    elif (rto is not None and RTO_ALLOWED is not None and p["state"] in RTO_ALLOWED
          and rto not in RTO_ALLOWED[p["state"]]):
        s -= 3.0
    s -= 2.0 * sum(c in _SERIES_ODD for c in p["series"])
    if p["state"] in LEGACY_STATE_CODES:
        s -= 0.7
    return s


def _bharat_score(p: dict[str, str]) -> float:
    s = 0.0
    year = _int_or_none(p["year"])
    if year is not None and year < 21:  # BH series started in 2021
        s -= 4.0
    s -= 2.0 * sum(c in _SERIES_ODD for c in p["suffix"])
    return s


def _build() -> Grammar:
    return Grammar(
        "india_plate",
        [
            Template(
                "standard",
                (
                    Choice("state", STATE_CODES),
                    Cls("rto", DIGITS, 1, 2),
                    Cls("series", LETTERS, 0, 3),
                    Cls("number", DIGITS, 1, 4),
                ),
                log_prior=0.0,
                score=_standard_score,
            ),
            Template(
                "bharat",
                (
                    Cls("year", DIGITS, 2, 2),
                    Choice("bh", ("BH",)),
                    Cls("number", DIGITS, 4, 4),
                    Cls("suffix", LETTERS, 1, 2),
                ),
                log_prior=-0.5,
                score=_bharat_score,
            ),
            Template(
                "diplomatic",
                (
                    Cls("code", DIGITS, 1, 3),
                    Choice("mission", ("CD", "CC", "UN")),
                    Cls("number", DIGITS, 1, 4),
                ),
                log_prior=-3.0,
            ),
            Template(
                "military",
                (
                    Cls("year", DIGITS, 2, 2),
                    Cls("base", LETTERS, 1, 1),
                    Cls("serial", DIGITS, 6, 6),
                    Cls("klass", LETTERS, 1, 1),
                ),
                log_prior=-4.0,
            ),
            # Dealer / trade-certificate mark as shown on a user-supplied infographic: DLR 01 2345. The format is
            # ILLUSTRATIVE (not checked against the official notification), so it is a low-prior template and
            # nothing else about trade plates is invented. Digit rule: 1-2 digits then 3-4 digits (DLR 01 2345,
            # DLR 1 234); 'DLR' is required, a bare digit string never matches. Segmentation tries the longest
            # first digit group first, so DLR12345 parses as rto '12' + number '345'.
            Template(
                "dealer",
                (
                    Choice("series", ("DLR",)),
                    Cls("rto", DIGITS, 1, 2),
                    Cls("number", DIGITS, 3, 4),
                ),
                log_prior=-2.0,
            ),
        ],
    )


PLATE_GRAMMAR = _build()


def parse_plate(text: str) -> dict | None:
    """Structured read of a complete plate string, or None if it is not a legal layout."""
    best = PLATE_GRAMMAR.best_parse(text)
    if best is None:
        return None
    tpl, p, prior = best
    out: dict = {"layout": tpl.name, "plate": text, "log_prior": round(prior, 3)}
    if tpl.name == "standard":
        st = p["state"]
        out.update(
            state_code=st,
            state_name=STATE_NAMES.get(st),          # None while the state code still holds a '?'
            rto=p["rto"],
            series=p["series"],
            number=p["number"],
            legacy_state_code=st in LEGACY_STATE_CODES,
        )
        if st == "DL" and p["series"] and len(p["rto"]) == 1:
            # Delhi prints the class letter against the RTO: DL 1C AB 1234
            head = f"{st} {p['rto']}{p['series'][0]}"
            tail = [p["series"][1:], p["number"]]
        else:
            head = f"{st} {p['rto']}"
            tail = [p["series"], p["number"]]
        out["display"] = " ".join([head, *[t for t in tail if t]])
    elif tpl.name == "bharat":
        out.update(year=p["year"], number=p["number"], suffix=p["suffix"],
                   display=f"{p['year']} BH {p['number']} {p['suffix']}")
    elif tpl.name == "diplomatic":
        out.update(code=p["code"], mission=p["mission"], number=p["number"],
                   display=f"{p['code']} {p['mission']} {p['number']}")
    elif tpl.name == "dealer":
        out.update(series=p["series"], rto=p["rto"], number=p["number"],
                   display=f"{p['series']} {p['rto']} {p['number']}")
    else:
        out.update(display=f"{p['year']}{p['base']} {p['serial']}{p['klass']}")
    return out
