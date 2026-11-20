"""Sample plate / VIN strings.

Sampling is deliberately close to uniform inside each slot. The recogniser must read
*pixels*; if it learned the statistics of a particular year's plates it would hallucinate
exactly where evidence is weakest. Structure comes from the layout (letters in letter slots,
digits in digit slots), priors come from the grammar at decode time.
"""

from __future__ import annotations

import numpy as np

from vinplate.grammar.india_plate import LEGACY_STATE_CODES, STATE_CODES
from vinplate.grammar.vin import KNOWN_WMI, VIN_CHARS, _YEAR_CODES

_LET = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ"))          # no I, O  (Q kept: rare)
_LET_ALL = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
_DIG = np.array(list("0123456789"))

# Rough registration-volume weights (large states heavier); only shapes the glyph mix.
_HEAVY = {"UP": 9, "MH": 9, "KA": 7, "TN": 7, "DL": 6, "GJ": 6, "RJ": 6, "HR": 5, "MP": 5,
          "WB": 5, "AP": 5, "TG": 5, "KL": 5, "BR": 4, "PB": 4, "OD": 3, "JH": 3, "AS": 3,
          "CG": 3, "UK": 2, "HP": 2, "GA": 2, "JK": 2, "CH": 2}


def _state_weights() -> np.ndarray:
    w = np.array([_HEAVY.get(s, 0.6 if s in LEGACY_STATE_CODES else 1.0) for s in STATE_CODES])
    return w / w.sum()


_SW = _state_weights()


def _pick(rng, arr, n):
    return "".join(rng.choice(arr, size=n))


def sample_plate(rng: np.random.Generator) -> dict:
    """-> {layout, groups, text, kind}. ``groups`` are the visual groups a plate prints with
    gaps between them (DL puts the class letter against the RTO: DL 1C AB 1234)."""
    r = rng.random()
    if r < 0.04:  # Bharat series
        year = f"{int(rng.integers(21, 27)):02d}"
        suffix = _pick(rng, _LET if rng.random() < 0.97 else _LET_ALL, int(rng.integers(1, 3)))
        number = f"{int(rng.integers(0, 10000)):04d}"
        groups = [year, "BH", number, suffix]
        return dict(layout="bharat", groups=groups, text="".join(groups), kind="bh")
    if r < 0.05:  # diplomatic
        groups = [str(int(rng.integers(1, 400))), str(rng.choice(["CD", "CC", "UN"])), str(int(rng.integers(1, 100)))]
        return dict(layout="diplomatic", groups=groups, text="".join(groups), kind="diplomatic")
    state = str(rng.choice(STATE_CODES, p=_SW))
    delhi = state == "DL" and rng.random() < 0.8
    rto = f"{int(rng.integers(1, 15 if delhi else 100)):d}" if delhi else f"{int(rng.integers(1, 100)):02d}"
    if rng.random() < 0.04 and not delhi:
        rto = rto.lstrip("0") or "1"
    nlet = 2 if rng.random() < 0.8 else 1
    if rng.random() < 0.015:
        nlet = 0                                           # legacy series-less mark
    letters = _LET if rng.random() < 0.97 else _LET_ALL
    series = _pick(rng, letters, nlet)
    if delhi:
        series = _pick(rng, letters, 1) + _pick(rng, letters, 2)
    n = int(rng.integers(0, 10000))
    number = f"{n:04d}" if rng.random() < 0.97 else str(n)  # legacy unpadded numbers are rare
    if delhi:
        groups = [state, rto + series[0], series[1:], number]
    else:
        groups = [state, rto, series, number]
    groups = [g for g in groups if g]
    return dict(layout="standard", groups=groups, text="".join(groups), kind="standard")


# ----------------------------------------------------------------------------- VIN
_INDIA_WMI = [w for w in KNOWN_WMI]
_VIN_ARR = np.array(list(VIN_CHARS))
_YEAR_ARR = np.array(list(_YEAR_CODES))
_ALNUM = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))


def sample_vin(rng: np.random.Generator) -> dict:
    """-> {text, legacy}. Mostly 17-char ISO 3779 VINs, India-biased WMI; ~8% short legacy
    chassis numbers (6-16 chars, I/O/Q allowed because old stamps used them)."""
    if rng.random() < 0.08:
        n = int(rng.integers(6, 17))
        return dict(text=_pick(rng, _ALNUM, n), legacy=True)
    r = rng.random()
    if r < 0.55:
        wmi = str(rng.choice(_INDIA_WMI))
    elif r < 0.85:
        wmi = "M" + str(rng.choice(list("ABCDE"))) + str(rng.choice(_VIN_ARR))
    else:
        wmi = _pick(rng, _VIN_ARR, 3)
    vds = _pick(rng, _VIN_ARR, 5)
    chk = str(rng.choice(np.array(list("0123456789X")))) if rng.random() < 0.6 else str(rng.choice(_VIN_ARR))
    year = str(rng.choice(_YEAR_ARR[:17])) if rng.random() < 0.8 else str(rng.choice(_YEAR_ARR))  # 2010-2026 (A..T) mostly
    plant = str(rng.choice(_VIN_ARR))
    serial = _pick(rng, _DIG, 6) if rng.random() < 0.85 else _pick(rng, _VIN_ARR, 6)
    return dict(text=wmi + vds + chk + year + plant + serial, legacy=False)
