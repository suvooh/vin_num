"""From a full photo to a plate / VIN *reading*: detect -> deskew -> recognise -> grammar decode.

This replaces "OCR the picture, then regex the text". The recogniser's posteriors are kept, so:

  * a damaged / glared / scratched character comes out as ``?`` (an explicit "unreadable"), not a
    confident wrong letter - PP-OCR emits CJK / punctuation mass there, which is mapped to ``?``;
  * the Indian plate grammar (state codes, series, 1- and 2-row layouts, Bharat series) and the VIN
    grammar (ISO 3779 structure, year code, check digit where it is mandatory) resolve *ambiguity*
    (0/O, 5/S, ...) and may pin a SINGLE unreadable character when a mandatory check digit makes it
    unique - always flagged as inferred, never presented as read;
  * several photometric views of the same crop (``enhance.variants``) and several photos are fused
    on the posteriors, because glare, shadows and blur hide different characters in different views;
  * clutter around the text ("IND", a "TC/20" temporary-registration mark, a "VIN:" label, a second
    sticker line) is handled by decoding sub-windows of the read when the whole line does not fit;
  * 2-row plates (two-wheelers, some cars) and plates the detector split into pieces are rejoined.

What it deliberately does NOT do is invent characters: information that is not in the pixels stays
unknown. ``FieldReading.status`` (accept / review / reject) is the one field to branch on, and
``reading.masked`` - not ``reading.text`` - is what to display whenever status is not ``accept``.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from vinplate.category import classify_plate
from vinplate.decode import Reading, entropy_fuse, fuse_frames, greedy, read_line
from vinplate.decode.ctc import alternatives, ctc_align
from vinplate.decode.lines import concat_lines
from vinplate.enhance import clip_stats, variants
from vinplate.grammar import PLATE_GRAMMAR, VIN_GRAMMAR, parse_plate, parse_vin
from vinplate.geometry.rows import looks_two_row, split_rows
from vinplate.grammar import vin_check_digit_ok
from vinplate.grammar.india_plate import STATE_CODES
from vinplate.ocr.ppocr import PpOcrEngine, TextBox, min_area_rect, warp_quad
from vinplate.quality import QualityReport, assess_crop
from vinplate.vocab import ALPHABET, CHARS, UNK, encode_text

GRAMMARS = {"plate": PLATE_GRAMMAR, "vin": VIN_GRAMMAR}
MIN_EVIDENCE = {"plate": 6, "vin": 9}        # characters actually supported by pixels
LOOKALIKES = ("O0QD", "I1L", "5S", "8B", "2Z", "6G")   # glyphs a damaged / blurred / glared character turns into each other
LOOKALIKE_P = 0.1                            # alternative look-alike evidence that makes a character ambiguous. Measured on 113 look-alike
                                             # positions of synthetic reads: all 3 WRONG characters had >= 0.2; 0.02 also flagged 8 correct
                                             # ones, 0.1 flags 3. Only 3 wrong characters - re-fit on real data.
MIN_CONFIDENCE = 0.15                        # below this a "reading" is noise, not a finding (provisional)
STATUS_RANK = {"accept": 2, "review": 1, "reject": 0}

# --- recogniser time-step budget -------------------------------------------------------------------------
# The recogniser sees a crop resized to 48 px HIGH, and emits one CTC step per ~8 px of width. A condensed HSRP
# plate row (tall, narrow glyphs) cropped by the detector can be only ~2.6:1, i.e. ~16 steps for 9 characters:
# CTC then cannot even separate repeated characters (`88`, `8888`) and drops some. Measured on a real photo of
# HR88B8888: 16 steps -> `HR88??88`; stretched x1.5..x2.0 (23..31 steps) -> `HR88B8888`, 97-98% confidence.
# So every line crop is stretched horizontally up to a minimum aspect ratio before it is recognised. Text that is
# already wide enough is never touched.
MIN_ASPECT = 4.0
MAX_STRETCH = 2.5


def normalise_aspect(crop: np.ndarray) -> np.ndarray:
    """Stretch ``crop`` horizontally so that width / height >= MIN_ASPECT (at most MAX_STRETCH times).

    new_w = int(w * s + 0.5) (round half UP) with s = min(MAX_STRETCH, MIN_ASPECT / (w / h)); PIL BICUBIC."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    aspect = w / h
    if aspect >= MIN_ASPECT:
        return crop
    s = min(MAX_STRETCH, MIN_ASPECT / aspect)
    new_w = int(w * s + 0.5)
    if new_w == w:
        return crop
    return np.asarray(Image.fromarray(crop).resize((new_w, h), Image.BICUBIC))


# --- screen context -------------------------------------------------------------------------------------
# A photo OF a screen shows the screen's interface: a web address, tab titles, "Related searches". The text
# detector has already read those lines, so they are free evidence. A vehicle scene has a few lines of text
# (the plate, a badge); a browser window has dozens, and a URL. HEURISTIC, easy for an attacker who crops to the
# plate alone to avoid - it catches the careless replay, nothing more.
UI_WORDS = ("SEARCH", "RELATED", "SETTINGS", "DOWNLOAD", "SUBSCRIBE", "COPILOT", "GOOGLE", "CHROME", "YOUTUBE",
            "FACEBOOK", "WHATSAPP", "SCREENSHOT", "DASHBOARD", "NOTIFICATIONS", "EXTENSIONS", "BOOKMARKS", "DETAILS",
            "IMAGES", "WINDOWS", "OUTLOOK", "TEAMS")
_URL_TLD = re.compile(r"\.(COM|ORG|NET|IN|CO)(?![A-Z0-9_])")
SCREEN_URL_W, SCREEN_WORDS_W, SCREEN_LINES_W = 1.0, 1.5, 0.5
SCREEN_MIN_WORDS, SCREEN_MIN_LINES = 3, 14
NO_SCREEN_CONTEXT = {"score": 0.0, "url": False, "ui_words": [], "text_lines": 0, "reasons": []}


def _ascii_alnum(t: str) -> int:
    return sum(1 for ch in t if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9"))


def screen_context(raw_lines: list[str], excluded=()) -> dict:
    """Interface-text evidence that the photo shows a screen. ``raw_lines``: the RapidOCR-style greedy text
    WITH punctuation of every detected line; ``excluded``: indices of lines that belong to the chosen plate / VIN
    (they are not scene context). Counting uses ASCII letters / digits only."""
    skip = set(excluded)
    lines = [(i, t) for i, t in enumerate(raw_lines) if i not in skip]
    text_lines = sum(1 for _, t in lines if _ascii_alnum(t) >= 4)
    url = False
    words: set[str] = set()
    for _, t in lines:
        up = t.upper()
        if "HTTP" in up or "WWW" in up or _URL_TLD.search(up):
            url = True
        az = "".join(ch for ch in up if "A" <= ch <= "Z")
        for w in UI_WORDS:
            if w in az:
                words.add(w)
    ui_words = sorted(words)
    many_words = len(ui_words) >= SCREEN_MIN_WORDS
    many_lines = text_lines >= SCREEN_MIN_LINES
    score = SCREEN_URL_W * url + SCREEN_WORDS_W * many_words + SCREEN_LINES_W * many_lines
    reasons = []
    if url:
        reasons.append("a web address (URL) is visible in the scene")
    if many_words:
        reasons.append("browser / app interface words are visible: " + ", ".join(ui_words))
    if many_lines:
        reasons.append(f"{text_lines} separate lines of text are visible, far more than a vehicle scene normally has")
    return {"score": float(score), "url": bool(url), "ui_words": ui_words, "text_lines": int(text_lines), "reasons": reasons}


# --- unknown slots --------------------------------------------------------------------------------------
SLOT_MIN_P = 0.05                       # an option below this share is not worth showing
SLOT_SUGGEST_P, SLOT_SUGGEST_MARGIN = 0.6, 0.3
SLOT_PEAK_EPS = 1e-9                    # score_c = log(peak_c + eps) + grammar.log_prior(string with c in the slot)


def slot_candidates(text: str, posterior, grammar, inferred=()) -> list[dict]:
    """For every unknown (``?``) or weak (``inferred``) character, the ranked characters the grammar allows in
    that slot, weighed by what the recogniser actually saw there AND by the pattern. The slot of an Indian plate
    after the RTO MUST be a letter (the 5th character of a 9-character plate; the 5th and 6th of a 10-character
    one), so a glare-hit ``B`` shows as ``B 0.93, R 0.03`` rather than a bare ``?``. Every character c with
    ``grammar.is_valid(text with c in the slot)`` is scored ``log(peak_c + 1e-9) + grammar.log_prior(that string)``
    (peak_c: the largest posterior of c over the frames CTC-aligned to the slot) and ``p = softmax(score)``: the
    prior penalises I/O/Q series letters, prefers a valid VIN model-year code at index 9 and an all-digit serial,
    and rewards a passing check digit. ``suggested`` is set only when one option dominates (>= 0.6 and 0.3 ahead
    of the next) - the UI still asks the person to confirm it. Positions index ``text`` (not ``masked``, which has
    a ``?`` inserted at every suspected gap)."""
    if posterior is None or not text:
        return []
    marked = set(inferred)
    positions = [i for i, c in enumerate(text) if c == UNK or i in marked]
    if not positions:
        return []
    try:
        spans = ctc_align(posterior, encode_text(text))
    except KeyError:
        return []
    out = []
    for i in positions:
        a, b, pk = spans[i]
        if pk <= 0.0:                    # no frames were aligned to this character
            continue
        peaks = np.exp(posterior[a : b + 1, 1 : 1 + len(ALPHABET)]).max(axis=0)
        # Pattern-aware ranking: a legal character is scored by what the recogniser saw AND by how plausible the
        # whole string is with it in this slot (series letters skip I/O/Q, a VIN's 10th character is a model-year
        # code, serial digits, a passing check digit ...). p = softmax over the legal characters only.
        scored: list[tuple[int, float]] = []
        for k, ch in enumerate(ALPHABET):
            cand = text[:i] + ch + text[i + 1 :]
            if not grammar.is_valid(cand):
                continue
            prior = grammar.log_prior(cand)
            if prior is None:
                continue
            scored.append((k, math.log(float(peaks[k]) + SLOT_PEAK_EPS) + prior))
        if not scored:
            continue
        top = max(s for _, s in scored)
        weights = [(k, math.exp(s - top)) for k, s in scored]
        total = sum(w for _, w in weights)
        ranked = sorted(((k, w / total) for k, w in weights), key=lambda kp: (-kp[1], kp[0]))
        opts = [(k, p) for k, p in ranked[:3] if p >= SLOT_MIN_P]
        if not opts:
            continue
        p0 = opts[0][1]
        p1 = opts[1][1] if len(opts) > 1 else 0.0
        out.append({"position": i, "options": [{"char": ALPHABET[k], "p": round(p, 3)} for k, p in opts],
                    "suggested": ALPHABET[opts[0][0]] if (p0 >= SLOT_SUGGEST_P and p0 - p1 >= SLOT_SUGGEST_MARGIN) else None})
    return out


_CONFUSABLE = {"0": "O", "O": "0", "1": "I", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8", "2": "Z",
               "Z": "2", "6": "G", "G": "6", "4": "A", "A": "4"}


_I, _O, _Q, _1, _0 = (CHARS.index(c) + 1 for c in "IOQ10")
_UNFOLD = str.maketrans("10", "IO")
_WORD_RUN = re.compile(r"[A-Z]{5,}")      # a run of letters that long is a word, not a chassis number
# Words printed on the same stickers / plates as a VIN. 17 characters of "TYRE PRESSURE 32 PSI" is a
# legal-looking VIN to every grammar, so a read containing one of these is clutter, not a VIN. The
# list is deliberately short and made of long / specific words, so a genuine VIN (23 letters, random)
# essentially never contains one; a VIN whose ISO check digit passes is never filtered by it.
LABEL_WORDS = ("PRESSURE", "TYRE", "TIRE", "FRONT", "GVWR", "GAWR", "KERB", "WEIGHT", "MANUFACT", "ENGINE",
               "CAPACITY", "SEATING", "CERTIF", "WARNING", "CAUTION", "BATTERY", "COLOUR", "MOTORS", "LIMITED",
               "VEHICLE", "MODEL", "MADE", "INDIA", "TOTAL", "LOAD")


def _fold_vin(lp: np.ndarray) -> np.ndarray:
    """ISO 3779 forbids I, O and Q in a VIN, so a recogniser that reports one of them is certain
    to have misread a 1 or a 0: move that probability mass onto the digit (the old regex did the
    same on text). Applied to the strict 17-character path only."""
    p = np.exp(lp)
    p[:, _1] += p[:, _I]
    p[:, _0] += p[:, _O] + p[:, _Q]
    p[:, [_I, _O, _Q]] = 1e-9
    return np.log(p / p.sum(axis=1, keepdims=True))


# ------------------------------------------------------------------------------- data
@dataclass
class LineObs:
    """One detected text line with its crop and (lazily computed) per-view posteriors."""

    box: TextBox
    crop: np.ndarray
    probs0: np.ndarray
    lp0: np.ndarray
    text0: str
    view_lp: dict[str, np.ndarray] = field(default_factory=dict)
    views_ready: bool = False
    raw0: str = ""            # RapidOCR-style greedy text WITH punctuation (screen-context evidence only)

    @property
    def n_alnum(self) -> int:
        return sum(c.isalnum() for c in self.text0)

    @property
    def n_digit(self) -> int:
        return sum(c.isdigit() for c in self.text0)


@dataclass
class FieldReading:
    domain: str                               # "plate" | "vin"
    reading: Reading
    parsed: dict | None
    box: tuple[int, int, int, int]            # union bbox in image pixels
    quads: list[list[list[float]]]            # one deskewed quad per text line used
    image: int = 0                            # index of the photo it came from
    views: int = 1                            # photometric views fused
    rows: int = 1
    evidence: int = 0                         # characters backed by pixels
    quality: QualityReport | None = None
    notes: list[str] = field(default_factory=list)
    lines: list = field(default_factory=list, repr=False)     # the LineObs it was read from
    posterior: np.ndarray | None = field(default=None, repr=False)   # the (fused / sub-window) posterior it was decoded from
    category: dict | None = None              # vehicle category (plates only): ``vinplate.category.classify_plate`` result

    @property
    def status(self) -> str:
        return self.reading.status

    def rank(self) -> tuple:
        return (STATUS_RANK[self.reading.status], min(self.evidence, 17), round(self.reading.confidence, 3))

    def as_dict(self) -> dict:
        r = self.reading
        out = {
            "domain": self.domain, "status": r.status, "value": r.text, "masked": r.masked,
            "confidence": round(float(r.confidence), 3), "char_confidence": r.char_conf,
            "inferred": r.inferred, "suspected_gaps": r.suspected_gaps, "grammar_cost": r.grammar_cost,
            "raw_read": r.greedy_text, "alternatives": [{"text": t, "p": p} for t, p in r.n_best],
            "reasons": r.reasons, "parsed": self.parsed, "box": list(self.box), "quads": self.quads,
            "image": self.image, "views": self.views, "rows": self.rows, "evidence_chars": self.evidence,
            "quality": self.quality.as_dict() if self.quality else None, "notes": self.notes,
            "calibrated": r.calibrated,
            "unknown_candidates": slot_candidates(r.text, self.posterior, GRAMMARS[self.domain], r.inferred),
            "category": self.category,
        }
        return out


@dataclass
class SceneResult:
    plate: FieldReading | None = None
    vin: FieldReading | None = None
    lines: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    per_image: list = field(default_factory=list, repr=False)    # one SceneResult per photo (read_frames only)
    size: tuple[int, int] = (0, 0)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    screen_context: dict = field(default_factory=lambda: dict(NO_SCREEN_CONTEXT))

    def as_dict(self) -> dict:
        return {"plate": self.plate.as_dict() if self.plate else None,
                "vin": self.vin.as_dict() if self.vin else None,
                "lines": self.lines, "conflicts": self.conflicts, "image_size": list(self.size),
                "timings_ms": self.timings, "notes": self.notes, "screen_context": self.screen_context}


# ------------------------------------------------------------------------------ geometry
def _axes(b: TextBox) -> tuple[np.ndarray, np.ndarray]:
    u = (b.quad[1] - b.quad[0]) / max(b.width, 1e-6)
    v = (b.quad[3] - b.quad[0]) / max(b.height, 1e-6)
    return u, v


def _stacked(a: TextBox, b: TextBox) -> bool:
    """b is a text row directly below a (2-row plate / wrapped sticker)."""
    u, v = _axes(a)
    d = b.center - a.center
    du, dv = float(d @ u), float(d @ v)
    h = (a.height + b.height) / 2
    return (dv > 0.5 * h and dv - h < 1.0 * h and abs(du) < 0.6 * max(a.width, b.width)
            and 0.55 < a.height / b.height < 1.8 and 0.3 < a.width / b.width < 3.4
            and abs(a.angle - b.angle) < 15)


def _side_by_side(a: TextBox, b: TextBox) -> bool:
    """b follows a on the same text row (a plate the detector split, a state code, a label)."""
    u, v = _axes(a)
    d = b.center - a.center
    du, dv = float(d @ u), float(d @ v)
    h = (a.height + b.height) / 2
    return (du > 0 and du - (a.width + b.width) / 2 < 1.4 * h and abs(dv) < 0.5 * h
            and 0.6 < a.height / b.height < 1.7 and abs(a.angle - b.angle) < 15)


def _near(a: TextBox, b: TextBox) -> bool:
    """Boxes close enough to belong to one plate: the gap between them is small relative to the text."""
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    gx = max(0, max(ax0, bx0) - min(ax1, bx1))
    gy = max(0, max(ay0, by0) - min(ay1, by1))
    h = max(a.height, b.height)
    return gx < 1.6 * h and gy < 0.9 * h


def _union_bbox(boxes: list[TextBox]) -> tuple[int, int, int, int]:
    bb = np.array([b.bbox for b in boxes])
    return int(bb[:, 0].min()), int(bb[:, 1].min()), int(bb[:, 2].max()), int(bb[:, 3].max())


# ------------------------------------------------------------------------------ reader
class SceneReader:
    # Paired ablation (40 synthetic scenes): orig+stretch gave the same gain as all five views (exact 85% -> 90%), so the
    # default is orig + stretch + deglare (deglare only runs when a crop has clipped hot-spots and is meant for glare; its
    # accuracy effect is NOT measured - the benchmark has too few glare scenes). sharpen / channel stay available via views=.
    VIEWS = ("orig", "stretch", "deglare")

    def __init__(self, engine: PpOcrEngine | None = None, *, multi_view: bool = True,
                 temperature: float = 1.0, max_lines: int = 6, refine_top: int = 3,
                 views: tuple[str, ...] | None = None):
        if engine is None:
            from vinplate.ocr.ppocr import default_engine

            engine = default_engine()
        self.engine = engine
        self.multi_view = multi_view
        self.temperature = temperature
        self.max_lines = max_lines
        self.refine_top = refine_top
        self.views = tuple(views) if views is not None else self.VIEWS
        self._last_post: np.ndarray | None = None      # posterior of the reading the last _decode returned

    # ----------------------------------------------------------------------- observation
    def observe(self, rgb: np.ndarray) -> tuple[list[LineObs], dict[str, float]]:
        t0 = time.perf_counter()
        boxes = [b for b in self.engine.detect(rgb) if b.height >= 7 and b.width >= 12]
        t1 = time.perf_counter()
        crops = [normalise_aspect(warp_quad(rgb, b.quad)) for b in boxes]
        probs = self.engine.rec_probs_many(crops) if crops else []
        raw = getattr(self.engine, "greedy", None)
        lines = []
        for b, c, p in zip(boxes, crops, probs):
            lp = self.engine.to_logp(p, self.temperature)
            ln = LineObs(b, c, p, lp, greedy(lp)[0])
            ln.raw0 = raw(p)[0] if raw is not None else ln.text0
            lines.append(ln)
        return lines, {"detect": (t1 - t0) * 1000, "recognise": (time.perf_counter() - t1) * 1000}

    def _view_logps(self, lines: list[LineObs]) -> dict[str, list[np.ndarray]]:
        """Posteriors of every photometric view shared by all ``lines`` (computed once per line)."""
        todo_c, todo_k = [], []
        for ln in lines:
            ln.view_lp.setdefault("orig", ln.lp0)
            if not ln.views_ready:
                for name, img in variants(ln.crop, full=True):
                    if name != "orig" and name in self.views:
                        todo_c.append(img)
                        todo_k.append((ln, name))
                ln.views_ready = True
        if todo_c:
            for (ln, name), p in zip(todo_k, self.engine.rec_probs_many(todo_c)):
                ln.view_lp[name] = self.engine.to_logp(p, self.temperature)
        names = [n for n in lines[0].view_lp if all(n in ln.view_lp for ln in lines)]
        return {n: [ln.view_lp[n] for ln in lines] for n in names}

    # --------------------------------------------------------------------------- decode
    @staticmethod
    def _join(lps: list[np.ndarray]) -> np.ndarray:
        return lps[0] if len(lps) == 1 else concat_lines(lps)

    def _decode(self, domain: str, lps_by_view: list[np.ndarray], *, allow_legacy: bool) -> tuple[Reading, str] | None:
        if domain == "vin":
            strict = self._decode_one("vin", [_fold_vin(lp) for lp in lps_by_view])
            if strict is not None and (parse_vin(strict[0].text) or {}).get("layout") == "vin17":
                return None if self._is_clutter(strict[0].text) else strict
            if not allow_legacy:
                return None
            # legacy / short chassis numbers: weaker structure, so never auto-accepted
            legacy = self._decode_one("vin", lps_by_view)
            if legacy is None:
                return None
            r, note = legacy
            digits = sum(c.isdigit() for c in r.text)
            if _WORD_RUN.search(r.text) or digits < 0.3 * len(r.text) or self._is_clutter(r.text):
                return None                  # reads like a word / label, not a chassis number
            if (parse_vin(r.text) or {}).get("layout") == "vin17":
                return r, note
            if r.status == "accept":
                r.status = "review"
            r.reasons.append("short / legacy chassis number: its structure cannot be validated")
            return r, note
        return self._decode_one(domain, lps_by_view)

    def _decode_one(self, domain: str, lps_by_view: list[np.ndarray]) -> tuple[Reading, str] | None:
        g = GRAMMARS[domain]
        reading = (read_line(lps_by_view[0], g) if len(lps_by_view) == 1 else fuse_frames(lps_by_view, g))
        note, used = "", lps_by_view
        if reading.status == "reject" or not reading.valid or self._implausible(domain, reading):
            win = self._best_window(domain, lps_by_view, g)
            if win is not None and (STATUS_RANK[win[0].status], win[0].confidence) > (STATUS_RANK[reading.status], reading.confidence):
                reading, used = win
                note = "decoded a sub-window of the line (surrounding text ignored)"
        fused = used[0] if len(used) == 1 else entropy_fuse(used)
        self._last_post = fused
        return self._harden(domain, reading, fused, g), note

    def _harden(self, domain: str, r: Reading, fused: np.ndarray, g) -> Reading:
        """Decode-time safety policy. A reading may only stay ``accept`` if

          * no character is ambiguous between look-alikes that the grammar would ALSO allow there
            (O/Q/0/D, I/1/L, 5/S, 8/B, 2/Z, 6/G) with real evidence for the alternative - such a
            character is marked unknown rather than shown as read;
          * (plates) it has series letters and a 4-digit number: a modern plate without them is
            either a very old plate or one with a DROPPED character, and CTC cannot tell which.
        """
        if r.status == "reject" or not r.valid or UNK in r.text:
            return r
        weak: list[int] = []
        try:
            alts = alternatives(fused, r.text, k=4)
        except (KeyError, ValueError):
            alts = []
        for i, (c, alt) in enumerate(zip(r.text, alts)):
            group = next((x for x in LOOKALIKES if c in x), None)
            if group is None:
                continue
            for ch, p in alt:
                if ch in group and ch != c and p >= LOOKALIKE_P and g.is_valid(r.text[:i] + ch + r.text[i + 1 :]):
                    weak.append(i)
                    break
        if weak:
            r.inferred = sorted(set(r.inferred) | set(weak))
            r.reasons.append("ambiguous look-alike character(s) at position(s) %s: shown as unknown" % sorted(weak))
        if domain == "plate":
            p = parse_plate(r.text)
            if p and p.get("layout") == "standard" and (not p["series"] or len(p["number"]) < 4):
                r.reasons.append("no series letters or a short number: a very old plate, or a character was dropped")
                weak.append(-1)
            if p and p.get("layout") == "standard" and any(c in "IOQ" for c in p["series"]):
                # the grammar notes it: series letters skip I and O (and Q in many states); a read that contains
                # one is far more likely a misread D / Q / 0 than a real series
                r.reasons.append("series letter I/O/Q: not issued on HSRP plates, so it is probably a misread look-alike")
                weak.append(-1)
        if weak and r.status == "accept":
            r.status = "review"
        return r

    @staticmethod
    def _is_clutter(text: str) -> bool:
        # the strict path has folded I->1 and O->0 (ENGINE -> ENG1NE), so match words on the unfolded form
        unfolded = text.translate(_UNFOLD)
        return not vin_check_digit_ok(text) and any(w in text or w in unfolded for w in LABEL_WORDS)

    @staticmethod
    def _implausible(domain: str, r: Reading) -> bool:
        return sum(c != UNK for c in r.text) < MIN_EVIDENCE[domain]

    def _best_window(self, domain: str, lps: list[np.ndarray], g) -> tuple[Reading, list[np.ndarray]] | None:
        """Sub-windows of the line. Decoding the whole line fails when it carries extra text; a
        window that starts where a state code (plate) / any 17-character span (VIN) could start and
        ends on a digit is tried instead. Windows are cut on the CTC alignment of the best path."""
        ref = lps[0]
        text, _ = greedy(ref)
        if len(text) < 6 or UNK in text and text.count(UNK) > len(text) // 2:
            return None
        try:
            spans = ctc_align(ref, encode_text(text))
        except KeyError:
            return None
        n = len(text)
        wins: list[tuple[int, int]] = []
        if domain == "plate":
            for i in range(n - 6):
                head = text[i : i + 2]
                bharat = text[i + 2 : i + 4] == "BH"        # 22 BH 1234 AA ends in letters, others in digits
                if not (head in STATE_CODES or UNK in head or bharat or _near_state(head)):
                    continue
                for j in range(i + 7, min(n, i + 11) + 1):
                    if (i, j) != (0, n) and (bharat or text[j - 1].isdigit() or text[j - 1] == UNK):
                        wins.append((i, j))
        else:
            for i in range(0, n - 16):
                if (i, i + 17) != (0, n):
                    wins.append((i, i + 17))
        best: tuple[Reading, list[np.ndarray]] | None = None
        T = ref.shape[0]
        for i, j in wins[:40]:
            a, b = max(0, spans[i][0] - 1), min(T, spans[j - 1][1] + 2)
            sub = [lp[int(a * lp.shape[0] / T): max(int(a * lp.shape[0] / T) + 2, int(b * lp.shape[0] / T))] for lp in lps]
            r = read_line(sub[0], g) if len(sub) == 1 else fuse_frames(sub, g)
            if not r.valid or self._implausible(domain, r):
                continue
            if best is None or (STATUS_RANK[r.status], r.confidence) > (STATUS_RANK[best[0].status], best[0].confidence):
                best = (r, sub)
        return best

    # -------------------------------------------------------------------- one photo
    def read(self, rgb: np.ndarray, *, domains: tuple[str, ...] = ("plate", "vin"),
             allow_legacy_vin: bool = False, image_index: int = 0) -> SceneResult:
        t_all = time.perf_counter()
        res = SceneResult(size=(rgb.shape[1], rgb.shape[0]))
        lines, tm = self.observe(rgb)
        res.timings.update({k: round(v, 1) for k, v in tm.items()})
        res.lines = [{"text": ln.text0, "box": list(ln.box.bbox), "detector_score": round(ln.box.score, 3)}
                     for ln in lines]
        short = sorted((ln for ln in lines if ln.n_digit >= 2 or (len(ln.text0) >= 6 and ln.text0.count(UNK) >= 2)),
                       key=lambda ln: (-ln.n_alnum, ln.box.center[1]))[: self.max_lines]
        short.sort(key=lambda ln: (ln.box.center[1], ln.box.center[0]))
        cands: list[list[LineObs]] = [[ln] for ln in short]
        pairs = [(a, b) for a in short for b in short
                 if a is not b and (_stacked(a.box, b.box) or _side_by_side(a.box, b.box))]
        pairs.sort(key=lambda ab: float(np.linalg.norm(ab[0].box.center - ab[1].box.center)))
        cands += [[a, b] for a, b in pairs[:8]]
        regions = self._regions(lines, short, rgb) if "plate" in domains else []

        t_dec = time.perf_counter()
        scored: list[FieldReading] = []
        for domain in domains:
            for lns in cands:
                est = sum(max(ln.n_alnum, 1) for ln in lns)
                if (domain == "plate" and not 4 <= est <= 16) or (domain == "vin" and est < 6):
                    continue
                dec = self._decode(domain, [self._join([ln.lp0 for ln in lns])], allow_legacy=allow_legacy_vin)
                if dec is None:
                    continue
                reading, note = dec
                fr = self._make_field(domain, lns, reading, rgb, image_index, [note] if note else [], views=1)
                scored.append(fr)
        for rows in regions:
            dec = self._decode("plate", [self._join([ln.lp0 for ln in rows])], allow_legacy=False)
            if dec is not None:
                reading, note = dec
                scored.append(self._make_field("plate", rows, reading, rgb, image_index,
                                               [x for x in (note, "read as one plate region") if x], views=1))
        # refine the most promising candidates with every photometric view
        refined: list[FieldReading] = []
        if self.multi_view:
            for domain in domains:
                top = sorted((f for f in scored if f.domain == domain), key=lambda f: f.rank(), reverse=True)[: self.refine_top]
                for f in top:
                    if f.status == "accept" and f.reading.confidence >= 0.97 and not f.reading.reasons:
                        continue
                    lns = f.lines
                    views = self._view_logps(lns)
                    names = list(views)
                    if len(names) < 2:
                        continue
                    stack = [self._join(views[n]) for n in names]
                    dec = self._decode(domain, stack, allow_legacy=allow_legacy_vin)
                    if dec is None:
                        continue
                    reading, note = dec
                    rf = self._make_field(domain, lns, reading, rgb, image_index,
                                          [x for x in (note, f"fused {len(names)} photometric views") if x], views=len(names))
                    refined.append(rf)
        pool = scored + refined
        res.timings["decode"] = round((time.perf_counter() - t_dec) * 1000, 1)
        res.candidates = [{"domain": f.domain, "value": f.reading.text, "masked": f.reading.masked, "status": f.status,
                           "confidence": round(float(f.reading.confidence), 3), "views": f.views, "evidence": f.evidence}
                          for f in sorted(pool, key=lambda f: f.rank(), reverse=True)[:12]]
        plate, vin = self._assign(pool)
        res.plate, res.vin = plate, vin
        res.screen_context = screen_context([ln.raw0 for ln in lines], self._owned_lines(lines, (plate, vin)))
        res.timings["total"] = round((time.perf_counter() - t_all) * 1000, 1)
        return res

    @staticmethod
    def _owned_lines(lines: list[LineObs], fields: tuple) -> set[int]:
        """Indices of detected lines that lie (>= 40% of their own area) inside a chosen plate / VIN box."""
        owned: set[int] = set()
        for i, ln in enumerate(lines):
            x0, y0, x1, y1 = ln.box.bbox
            area = max(0, x1 - x0) * max(0, y1 - y0)
            for f in fields:
                if f is None:
                    continue
                ix = max(0, min(x1, f.box[2]) - max(x0, f.box[0]))
                iy = max(0, min(y1, f.box[3]) - max(y0, f.box[1]))
                if area > 0 and ix * iy >= 0.4 * area:
                    owned.add(i)
        return owned

    # --------------------------------------------------------------------------- helpers
    def _regions(self, lines: list[LineObs], short: list[LineObs], rgb: np.ndarray,
                 max_regions: int = 3) -> list[list[LineObs]]:
        """Plate-sized regions: connected groups of nearby text boxes, re-read as ONE deskewed
        crop (split into rows when it is plate-shaped). This rescues plates the detector fragmented
        - a row cut by a scratch, a state code split from the number - which no pairing of boxes can."""
        n = len(lines)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if _near(lines[i].box, lines[j].box):
                    parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        seeds = {id(ln) for ln in short}
        out: list[list[LineObs]] = []
        for idx in groups.values():
            members = [lines[i] for i in idx]
            if len(members) < 2 or not any(id(m) in seeds for m in members):
                continue
            quad, long_, short_ = min_area_rect(np.concatenate([m.box.quad for m in members]))
            if short_ < 12 or not 1.1 <= long_ / short_ <= 7.5:
                continue
            crop = warp_quad(rgb, quad, pad=0.05)
            h, w = crop.shape[:2]
            if h < 12:
                continue
            crops = [normalise_aspect(c) for c in (split_rows(crop) if looks_two_row(h, w) else [crop])]
            rows = []
            for k, (c, pr) in enumerate(zip(crops, self.engine.rec_probs_many(crops))):
                lp = self.engine.to_logp(pr, self.temperature)
                a, b = k / len(crops), (k + 1) / len(crops)
                rq = np.array([quad[0] + (quad[3] - quad[0]) * a, quad[1] + (quad[2] - quad[1]) * a,
                               quad[1] + (quad[2] - quad[1]) * b, quad[0] + (quad[3] - quad[0]) * b])
                rows.append(LineObs(TextBox(rq, min(m.box.score for m in members)), c, pr, lp, greedy(lp)[0]))
            out.append(rows)
        out.sort(key=lambda r: -sum(ln.n_digit for ln in r))
        return out[:max_regions]

    def _make_field(self, domain: str, lns: list[LineObs], reading: Reading, rgb: np.ndarray, image: int,
                    notes: list[str], *, views: int) -> FieldReading:
        parsed = (parse_plate if domain == "plate" else parse_vin)(reading.text) if reading.valid and reading.status != "reject" else None
        evidence = sum(1 for i, c in enumerate(reading.text) if c != UNK and i not in reading.inferred)
        biggest = max(lns, key=lambda ln: ln.box.width * ln.box.height)
        return FieldReading(
            domain=domain, reading=reading, parsed=parsed, box=_union_bbox([ln.box for ln in lns]),
            quads=[ln.box.quad.round(1).tolist() for ln in lns], image=image, views=views, rows=len(lns),
            evidence=evidence, quality=assess_crop(biggest.crop) if biggest.crop.shape[0] >= 8 else None, notes=notes,
            lines=list(lns), posterior=self._last_post, category=_category_of(domain, lns, parsed))

    @staticmethod
    def _assign(pool: list[FieldReading]) -> tuple[FieldReading | None, FieldReading | None]:
        """Best plate and best VIN such that no text line is used twice (a VIN sticker must not
        also be reported as a plate, nor a plate as a chassis number)."""
        def ok(f: FieldReading) -> bool:
            # a read this weak is not a finding: keep it out of the result (it stays in `candidates`)
            return f.evidence >= MIN_EVIDENCE[f.domain] and f.status != "reject" and f.reading.confidence >= MIN_CONFIDENCE

        plates = sorted((f for f in pool if f.domain == "plate" and ok(f)), key=lambda f: f.rank(), reverse=True)
        vins = sorted((f for f in pool if f.domain == "vin" and ok(f)), key=lambda f: f.rank(), reverse=True)

        def shares(a: FieldReading, b: FieldReading) -> bool:
            ix = max(0, min(a.box[2], b.box[2]) - max(a.box[0], b.box[0]))
            iy = max(0, min(a.box[3], b.box[3]) - max(a.box[1], b.box[1]))
            small = min((a.box[2] - a.box[0]) * (a.box[3] - a.box[1]), (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))
            return ix * iy > 0.4 * max(small, 1)

        best, best_s = (None, None), -1.0
        for p in plates[:4] or [None]:
            for v in vins[:4] or [None]:
                if p is not None and v is not None and shares(p, v):
                    continue
                s = sum(f.evidence * (1.0 + 0.5 * STATUS_RANK[f.status]) for f in (p, v) if f is not None)
                if s > best_s:
                    best, best_s = (p, v), s
        if best == (None, None) and (plates or vins):
            return (plates[0] if plates else None), (None if plates else vins[0])
        return best

    # ------------------------------------------------------------------- several photos
    def read_frames(self, rgbs: list[np.ndarray], *, domains: tuple[str, ...] = ("plate", "vin"),
                    allow_legacy_vin: bool = False) -> SceneResult:
        """Several photos of the same vehicle: read each, then fuse the readings of the same
        identifier on the posteriors (glare and smudges move between photos). Readings of the same
        domain that cannot be reconciled are reported in ``conflicts`` - never silently averaged."""
        singles = [self.read(im, domains=domains, allow_legacy_vin=allow_legacy_vin, image_index=i) for i, im in enumerate(rgbs)]
        if len(singles) == 1:
            return singles[0]
        out = SceneResult(size=singles[0].size, lines=[l for s in singles for l in s.lines], per_image=singles)
        out.timings = {k: round(sum(s.timings.get(k, 0) for s in singles), 1) for k in ("detect", "recognise", "decode", "total")}
        out.screen_context = max((s.screen_context for s in singles), key=lambda c: c["score"])      # first of equals
        for domain in domains:
            found = [(i, getattr(s, domain)) for i, s in enumerate(singles) if getattr(s, domain) is not None]
            if not found:
                continue
            # group by compatibility, keep the group with the most evidence
            groups: list[list[tuple[int, FieldReading]]] = []
            for item in found:
                for g in groups:
                    if _compatible(g[0][1].reading.masked, item[1].reading.masked):
                        g.append(item)
                        break
                else:
                    groups.append([item])
            groups.sort(key=lambda g: sum(f.evidence * (1 + STATUS_RANK[f.status]) for _, f in g), reverse=True)
            main = groups[0]
            for g in groups[1:]:
                for i, f in g:
                    out.conflicts.append({"domain": domain, "image": i, "masked": f.reading.masked,
                                          "agrees_with": [k for k, _ in main], "main": main[0][1].reading.masked})
            best = max((f for _, f in main), key=lambda f: f.rank())
            if len(main) > 1:
                fused = self._fuse_group(domain, [(i, f) for i, f in main], singles, rgbs, allow_legacy_vin)
                if fused is not None and fused.rank() >= best.rank():
                    best = fused
            setattr(out, domain, best)
        return out

    def _fuse_group(self, domain, group, singles, rgbs, allow_legacy) -> FieldReading | None:
        """Re-decode one identifier from the best-view posteriors of every photo that read it."""
        lps, wts = [], []
        for i, f in group:
            # re-observe is avoided: rebuild the line posteriors from the stored quads
            lns = []
            for q in f.quads:
                crop = normalise_aspect(warp_quad(rgbs[i], np.array(q)))
                p = self.engine.rec_probs(crop)
                lns.append(self.engine.to_logp(p, self.temperature))
            lps.append(self._join(lns))
            wts.append(1.0 + 0.5 * STATUS_RANK[f.status])
        if len(lps) < 2:
            return None
        g = GRAMMARS[domain]
        reading = fuse_frames(lps, g, weights=wts)
        fused = entropy_fuse(lps, wts)
        if reading.status == "reject" or (domain == "vin" and not allow_legacy and (parse_vin(reading.text) or {}).get("layout") != "vin17"):
            return None
        i, f = max(group, key=lambda it: it[1].rank())
        parsed = (parse_plate if domain == "plate" else parse_vin)(reading.text) if reading.status != "reject" else None
        # the category is judged from the best-ranked photo's biggest line crop (the fused posterior has no pixels),
        # against the FUSED reading's text layout
        category = _category_of(domain, f.lines, parsed) if f.lines else f.category
        return FieldReading(domain=domain, reading=reading, parsed=parsed,
                            box=f.box, quads=f.quads, image=i, views=len(lps), rows=f.rows,
                            evidence=sum(1 for k, c in enumerate(reading.text) if c != UNK and k not in reading.inferred),
                            quality=f.quality, notes=[f"fused {len(lps)} photos: " + ", ".join(str(k) for k, _ in group)],
                            posterior=fused, category=category)


def _category_of(domain: str, lns: list, parsed: dict | None) -> dict | None:
    """Vehicle category of a PLATE from the biggest line crop it was read from (None for a VIN or no usable pixels)."""
    if domain != "plate" or not lns:
        return None
    biggest = max(lns, key=lambda ln: ln.box.width * ln.box.height)
    crop = biggest.crop
    if crop is None or crop.ndim != 3 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return classify_plate(crop, parsed)


def _near_state(head: str) -> bool:
    """Two characters that are one common look-alike substitution away from a state code."""
    if len(head) != 2:
        return False
    swaps = ([head[0]] + ([_CONFUSABLE[head[0]]] if head[0] in _CONFUSABLE else []),
             [head[1]] + ([_CONFUSABLE[head[1]]] if head[1] in _CONFUSABLE else []))
    return any(a + b in STATE_CODES for a in swaps[0] for b in swaps[1])


def _compatible(a: str, b: str) -> bool:
    """Two masked readings of the same identifier: equal length and no position where both are
    known and different ('?' matches anything), allowing a single character of slack (a misread)."""
    if len(a) != len(b):
        return False
    bad = sum(1 for x, y in zip(a, b) if x != UNK and y != UNK and x != y)
    return bad <= 1
