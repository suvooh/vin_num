"""What KIND of vehicle a plate belongs to: from the plate COLOURS in the photo and the TEXT PATTERN.

    classify_plate(crop_rgb, parsed) -> {"key", "label", "confidence", "bg", "ink", "reasons",
                                          "alternatives", "ev", "bh"}

Everything here is deterministic, numpy-only and written so a Dart port reproduces it exactly (the
Dart port in ``app/lib/reader/category.dart`` is checked against golden outputs of this file). That is why
every formula is spelled out and no library colour conversion is used.

PIPELINE
  1. sample    step = max(1, int(sqrt(H*W/4096))); px = crop[::step, ::step] as float64 (N x 3).
  2. 2-means   bg0 = per-channel median of px (even N: mean of the two middle values); ink0 = the pixel
               farthest from bg0 (first maximum). 8 iterations of: assign every pixel to the nearer of the two
               centres by SQUARED Euclidean distance (a pixel goes to the ink centre only if strictly nearer;
               ties -> bg), then recompute both centres as cluster means (an empty cluster keeps its centre).
               After the loop the cluster with MORE members (counted from the last assignment) is the
               background (a tie keeps the initial bg cluster), the other is the ink. If the two final centres
               are < 40 apart the crop is one colour: ink = None and bg = the overall mean of px.
  3. name      ``name_colour`` maps a float RGB centre to a colour name + confidence (below).
  4. rules     the RULE TABLE below turns (background, ink, text layout) into a category.
  5. confidence = min(bg_conf, ink_conf if ink else 0.7) * min(1, dist_between_centres / 120), then the layout /
               colour adjustments in ``classify_plate``.

``name_colour(rgb)``: r, g, b in 0..255 -> /255; v = max, s = (max-min)/max (0 if max == 0), hue h in degrees
(0 if max == min; max == r: 60*((g-b)/d); max == g: 60*((b-r)/d + 2); else 60*((r-g)/d + 4); add 360 if
negative, 360 -> 0; ties for the maximum resolve in the order r, g, b).
    v < 0.28                                   -> black
    s < 0.22                                   -> white if v >= 0.62; grey if v >= 0.40; else black
    s >= 0.22 and 14 <= h < 70, s < 0.30, v >= 0.62 -> white (a pale cream), else by hue:
        red if h < 14 or h >= 340; yellow 14 <= h < 70; green 70 <= h < 195; blue 195 <= h < 270; other.
confidence = 0.5 + 0.5 * clip(min(margins), 0, 1) where margins are |value - boundary| / width for every
boundary that separates the chosen name from its neighbours: v 0.28 (width 0.15; every non-black name), s 0.22
(0.15; every non-black name), v 0.62 / v 0.40 for white / grey / dark-grey-black as listed in ``name_colour``,
and for chromatic names the circular hue distance to each end of the hue band / 20 degrees (red: 340 and 14;
yellow: 14 and 70; green: 70 and 195; blue: 195 and 270; other: 270 and 340). A yellow also gets the distance to
the pale-cream region (see the code) and a cream 'white' gets |s - 0.30| and |v - 0.62| over 0.15.

ABOUT THE RULES: they come from two infographics supplied by the user (colour -> category) and are RECALLED, NOT
VERIFIED against the official notifications (MoRTH / CMVR rules and state practice vary, e.g. some states issue
different colours). They are data: change the tables below and nothing else.
"""

from __future__ import annotations

import math

import numpy as np

# ===================================================================================== RULE TABLE
# One place to correct. Status of each row: R = read off the user's infographics, U = not verified against the
# official notifications. Everything below the table is mechanism.

LABELS: dict[str, str] = {
    "private": "Private vehicle",
    "commercial": "Commercial / transport vehicle",
    "rental": "Rental / self-drive vehicle",
    "ev_private": "Electric vehicle (private)",
    "ev_commercial": "Electric vehicle (commercial)",
    "temporary": "Temporary registration",
    "trade": "Trade certificate / dealer vehicle",
    "diplomatic": "Diplomatic / consular vehicle",
    "military": "Military / defence vehicle",
    "unknown": "Type not determined",
}

# (background colour, ink colour or None = any, category key, confidence factor). FIRST match wins.
COLOUR_RULES: list[tuple[str, str | None, str, float]] = [
    ("blue", None, "diplomatic", 0.7),       # R U  blue plate: diplomatic/consular. x0.7: a blue plate with ordinary text is unusual
    ("green", "yellow", "ev_commercial", 1.0),   # R U  green + yellow text: commercial EV
    ("green", None, "ev_private", 1.0),      # R U  green + white text: private EV
    ("yellow", "red", "temporary", 1.0),     # R U  yellow + RED text: temporary registration
    ("yellow", None, "commercial", 1.0),     # R U  yellow + black text: commercial / transport
    ("black", "yellow", "rental", 1.0),      # R U  black + yellow text: rent-a-cab / self-drive
    ("grey", "yellow", "rental", 1.0),       # R U  a grey-looking dark plate (lighting) read as black
    ("red", None, "trade", 1.0),             # R U  red + white text: trade certificate / dealer
    ("white", None, "private", 1.0),         # R U  white + black text: private / non-transport (BH series too)
]

# A text layout that decides the category by itself, with the backgrounds it is normally printed on.
LAYOUT_RULES: dict[str, tuple[str, frozenset]] = {
    "diplomatic": ("diplomatic", frozenset({"blue"})),            # NNN CD|CC|UN NN
    "military": ("military", frozenset({"black", "grey"})),       # NNL NNNNNN L, dark plate, white text, up-arrow
    "dealer": ("trade", frozenset({"red"})),                      # DLR NN NNNN (illustrative format)
}
BHARAT_BG = "white"                                               # NN BH NNNN AA: white plate, black text

# Where "green" ends and "blue" begins on the hue wheel. MEASURED, not assumed: the real Indian EV plate is a TEAL, not a
# leaf green - 174-175 deg on a clean photo of a green-plate Camry, 184-190 deg (and washed out, saturation 0.24-0.31) in a
# phone photo of that picture on a monitor - while a real diplomatic blue is about 210-225 deg. The first version put the
# boundary at 170 deg and named that EV plate "blue" -> "diplomatic". 195 keeps both real plates on the right side with
# margin (teal 175-190, blue 210+). A plate between 175 and 215 is simply low-confidence (the margin term).
GREEN_BLUE_HUE = 195.0

# TEXT-PATTERN GUARD. A diplomatic plate always carries a CD / CC / UN code (NNN CD NN). So a plate whose text is an ordinary
# registration (state code + RTO + series + number), a BH number, a defence number or a dealer mark can NOT be diplomatic,
# whatever colour it measured: the colour was misread (a teal EV plate, a cool-tinted white plate, a screen's colour cast).
# Such a plate is reported as "type not determined" with the likelier answers as alternatives, never as diplomatic.
NOT_DIPLOMATIC_LAYOUTS = frozenset({"standard", "bharat", "military", "dealer"})

# Categories one colour step away, tried when the confidence is below ALT_BELOW (at most 2, in this order).
ALTERNATIVES: dict[str, list[str]] = {
    "private": ["commercial"],                          # white <-> yellow
    "commercial": ["private", "temporary"],             # yellow <-> white, yellow <-> red text
    "temporary": ["commercial", "trade"],               # red text <-> black text, yellow <-> red
    "rental": ["diplomatic"],                           # black <-> blue
    "diplomatic": ["rental", "ev_private"],             # blue <-> black, blue <-> green
    "ev_private": ["ev_commercial", "diplomatic"],      # white text <-> yellow text, green <-> blue
    "ev_commercial": ["ev_private", "diplomatic"],
    "trade": ["commercial", "temporary"],               # red <-> yellow
    "military": ["rental"],
}
UNKNOWN_ALTERNATIVES = {"black": ["military", "rental"], "grey": ["military", "rental"]}   # else private, commercial
ALT_BELOW = 0.75

# Exact reason strings (the Dart port and the goldens use these verbatim).
R_BG = "plate background looks {}"
R_INK = "text looks {}"
R_ONE_COLOUR = "text colour could not be separated from the background"
R_BH = "format NN BH NNNN AA = Bharat series"
R_BH_COLOUR = "Bharat-series plates are white with black text"
R_LAYOUT = {
    "diplomatic": "format NNN CD NN = diplomatic mission code",
    "military": "format NNL NNNNNN L = military",
    "dealer": "format DLR NN NNNN = dealer",
}
R_AGREE = "colour agrees with the plate format"
R_DISAGREE = "plate colour ({}) does not match the {} format"
R_BLUE = "blue plates with ordinary text are unusual"
R_DIP_FORMAT = "diplomatic plates carry a CD / CC / UN code; this text is an ordinary registration, so the blue colour is probably a colour cast"
R_NO_RULE = "no rule matches this colour pair"

# ======================================================================================= mechanism
_MIN_SEPARATION = 40.0
_SEP_FULL = 120.0


def _hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    v = mx
    s = 0.0 if mx == 0 else d / mx
    if d == 0:
        h = 0.0
    elif mx == r:
        h = 60.0 * ((g - b) / d)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    if h < 0:
        h += 360.0
    if h >= 360.0:
        h -= 360.0
    return h, s, v


def _m(value: float, boundary: float, width: float) -> float:
    return abs(value - boundary) / width


def _hd(h: float, b: float) -> float:
    d = abs(h - b)
    return min(d, 360.0 - d)


def _conf(margins: list[float]) -> float:
    return 0.5 + 0.5 * min(1.0, max(0.0, min(margins)))


def name_colour(rgb) -> tuple[str, float]:
    """A float RGB (0..255) -> (colour name, confidence). See the module docstring for the exact rules."""
    h, s, v = _hsv(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    if v < 0.28:
        return "black", _conf([_m(v, 0.28, 0.15)])
    base = [_m(s, 0.22, 0.15), _m(v, 0.28, 0.15)]
    if s < 0.22:
        if v >= 0.62:
            return "white", _conf(base + [_m(v, 0.62, 0.15)])
        if v >= 0.40:
            return "grey", _conf(base + [_m(v, 0.62, 0.15), _m(v, 0.40, 0.15)])
        return "black", _conf(base + [_m(v, 0.40, 0.15)])
    if 14 <= h < 70 and s < 0.30 and v >= 0.62:
        return "white", _conf(base + [_m(s, 0.30, 0.15), _m(v, 0.62, 0.15)])
    if h < 14 or h >= 340:
        return "red", _conf(base + [_hd(h, 14) / 20.0, _hd(h, 340) / 20.0])
    if h < 70:
        if s < 0.30:
            cream = _m(v, 0.62, 0.15)                     # v < 0.62 here
        elif v >= 0.62:
            cream = _m(s, 0.30, 0.15)
        else:
            cream = max(_m(s, 0.30, 0.15), _m(v, 0.62, 0.15))
        return "yellow", _conf(base + [_hd(h, 14) / 20.0, _hd(h, 70) / 20.0, cream])
    if h < GREEN_BLUE_HUE:
        return "green", _conf(base + [_hd(h, 70) / 20.0, _hd(h, GREEN_BLUE_HUE) / 20.0])
    if h < 270:
        return "blue", _conf(base + [_hd(h, GREEN_BLUE_HUE) / 20.0, _hd(h, 270) / 20.0])
    return "other", _conf(base + [_hd(h, 270) / 20.0, _hd(h, 340) / 20.0])


def two_means(crop: np.ndarray):
    """-> (bg_mean (3,), ink_mean (3,) or None, distance between the final centres). Deterministic."""
    h, w = crop.shape[:2]
    step = max(1, int(math.sqrt(h * w / 4096)))
    px = crop[::step, ::step].reshape(-1, 3).astype(np.float64)
    n = px.shape[0]
    c_bg = np.median(px, axis=0)
    d0 = ((px - c_bg) ** 2).sum(axis=1)
    c_ink = px[int(np.argmax(d0))].copy()
    lab = np.zeros(n, bool)
    for _ in range(8):
        d_bg = ((px - c_bg) ** 2).sum(axis=1)
        d_ink = ((px - c_ink) ** 2).sum(axis=1)
        lab = d_ink < d_bg                       # strictly nearer to the ink centre; ties -> bg
        if (~lab).any():
            c_bg = px[~lab].mean(axis=0)
        if lab.any():
            c_ink = px[lab].mean(axis=0)
    n_ink = int(lab.sum())
    if n_ink > n - n_ink:
        c_bg, c_ink = c_ink, c_bg
    dist = float(np.sqrt(((c_bg - c_ink) ** 2).sum()))
    if dist < _MIN_SEPARATION:
        return px.mean(axis=0), None, dist
    return c_bg, c_ink, dist


def _rgb_ints(c) -> list[int]:
    return [int(min(255, max(0, math.floor(float(v) + 0.5)))) for v in c]


def _unknown(reasons: list[str], bg=None, ink=None, conf: float = 0.0) -> dict:
    return {"key": "unknown", "label": LABELS["unknown"], "confidence": round(conf, 3), "bg": bg, "ink": ink,
            "reasons": reasons, "alternatives": [], "ev": False, "bh": False}


def _alts(keys: list[str]) -> list[dict]:
    return [{"key": k, "label": LABELS[k]} for k in keys[:2]]


def _colour_rule(bg: str, ink: str | None) -> tuple[str, float] | None:
    for rbg, rink, key, factor in COLOUR_RULES:
        if rbg == bg and (rink is None or rink == ink):
            return key, factor
    return None


def classify_plate(crop_rgb: np.ndarray, parsed: dict | None) -> dict:
    """Category of a plate from its crop (HxWx3 uint8) and the parsed text layout (``parse_plate`` result or None)."""
    if crop_rgb is None or crop_rgb.ndim != 3 or crop_rgb.shape[0] < 1 or crop_rgb.shape[1] < 1:
        return _unknown([R_NO_RULE])
    bg_c, ink_c, dist = two_means(crop_rgb)
    bg_name, bg_conf = name_colour(bg_c)
    ink_name, ink_conf = name_colour(ink_c) if ink_c is not None else (None, None)
    bg = {"name": bg_name, "rgb": _rgb_ints(bg_c)}
    ink = {"name": ink_name, "rgb": _rgb_ints(ink_c)} if ink_c is not None else None
    layout = (parsed or {}).get("layout")
    base = min(bg_conf, ink_conf if ink_c is not None else 0.7) * min(1.0, dist / _SEP_FULL)
    reasons = [R_BG.format(bg_name)]
    reasons.append(R_INK.format(ink_name) if ink_c is not None else R_ONE_COLOUR)

    bh = layout == "bharat"
    colour = _colour_rule(bg_name, ink_name)
    if layout in LAYOUT_RULES:
        key, expected_bg = LAYOUT_RULES[layout]
        reasons.append(R_LAYOUT[layout])
        if bg_name in expected_bg:
            conf = min(1.0, base + 0.1)
            reasons.append(R_AGREE)
        else:
            conf = base * 0.6
            reasons.append(R_DISAGREE.format(bg_name, layout))
        alt_keys = [colour[0]] if colour and colour[0] != key else []
        alt_keys += [k for k in ALTERNATIVES.get(key, []) if k not in alt_keys and k != key]
        alts = _alts(alt_keys) if conf < ALT_BELOW else []
        return {"key": key, "label": LABELS[key], "confidence": round(conf, 3), "bg": bg, "ink": ink, "reasons": reasons,
                "alternatives": alts, "ev": False, "bh": False}

    if colour is None:
        keys = UNKNOWN_ALTERNATIVES.get(bg_name, ["private", "commercial"])
        out = _unknown(reasons + [R_NO_RULE], bg, ink, base * 0.6)
        out["alternatives"] = _alts(keys)
        out["bh"] = bh
        if bh:
            out["reasons"].insert(2, R_BH)
        return out

    key, factor = colour
    if key == "diplomatic" and layout in NOT_DIPLOMATIC_LAYOUTS:
        # text-pattern guard (see NOT_DIPLOMATIC_LAYOUTS): the likelier readings of a blue-looking plate with a normal number
        alt = ["ev_commercial", "ev_private"] if ink_name == "yellow" else ["ev_private", "private"]
        out = _unknown(reasons + [R_DIP_FORMAT], bg, ink, base * 0.5)
        out["alternatives"] = _alts(alt)
        out["bh"] = bh
        if bh:
            out["reasons"].insert(2, R_BH)
        return out
    conf = base * factor
    if bg_name == "blue":
        reasons.append(R_BLUE)
    if bh:
        reasons.append(R_BH)
        if bg_name == BHARAT_BG:
            conf = min(1.0, conf + 0.1)
            reasons.append(R_AGREE)
        else:
            conf = conf * 0.6
            reasons.append(R_BH_COLOUR)
    # an ordinary registration rules out a diplomatic plate (NOT_DIPLOMATIC_LAYOUTS), so it is never offered as an alternative
    near = [k for k in ALTERNATIVES.get(key, []) if not (k == "diplomatic" and layout in NOT_DIPLOMATIC_LAYOUTS)]
    alts = _alts(near) if conf < ALT_BELOW else []
    return {"key": key, "label": LABELS[key], "confidence": round(conf, 3), "bg": bg, "ink": ink, "reasons": reasons,
            "alternatives": alts, "ev": key.startswith("ev_"), "bh": bh}
