"""PP-OCR (PaddleOCR v3 detector + recogniser, ONNX) with access to the recogniser's POSTERIORS.

Why this exists. The generic OCR the backend used (RapidOCR) returns only a string and one
confidence per line, so everything downstream was regex over text and glare, scratches and
damage could not be handled except by guessing. Here the recogniser's per-time-step class
probabilities are exposed and projected onto the plate/VIN alphabet (``vinplate.vocab``), which
is what the grammar-constrained CTC decoder, the ``?`` abstention and multi-view fusion need.

It uses the same two ONNX files the Flutter app bundles (``app/assets/models``) and depends only
on onnxruntime + numpy + scipy + Pillow (no OpenCV: no win-arm64 wheel exists).

Detection (DB net) is a faithful port of RapidOCR's pre/post-processing: min-side upscale to 736,
threshold 0.3, 2x2 dilation, box score >= 0.5, unclip ratio 1.6 - except that boxes are the
minimum-area ROTATED rectangle of each blob, so tilted plates are deskewed by a perspective crop.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

from vinplate.runtime import find_model, make_session
from vinplate.vocab import BLANK, CHARS, UNK, VOCAB_SIZE

# --- detector config (RapidOCR config.yaml) -------------------------------------------------
LIMIT_SIDE = 736
THRESH = 0.3
BOX_THRESH = 0.5
UNCLIP = 1.6
MIN_SIZE = 3
REC_H = 48
TEXT_SCORE = 0.5      # RapidOCR drops lines below this mean confidence (baseline reader only)

_DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------------------- geometry
@dataclass
class TextBox:
    """A detected text line: ``quad`` is TL, TR, BR, BL in image pixels (long side = width)."""

    quad: np.ndarray
    score: float

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.quad[1] - self.quad[0]))

    @property
    def height(self) -> float:
        return float(np.linalg.norm(self.quad[3] - self.quad[0]))

    @property
    def center(self) -> np.ndarray:
        return self.quad.mean(axis=0)

    @property
    def angle(self) -> float:
        d = self.quad[1] - self.quad[0]
        return math.degrees(math.atan2(d[1], d[0]))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.quad.min(axis=0)
        x1, y1 = self.quad.max(axis=0)
        return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def min_area_rect(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Minimum-area bounding rectangle of ``points`` (N, 2) via the convex hull.
    Returns (corners TL,TR,BR,BL with the LONG side along x, long, short)."""
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) >= 3:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except QhullError:  # collinear points
            pass
    edges = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    ang = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 6))
    if len(ang) == 0:
        ang = np.array([0.0])
    best = None
    for a in ang:
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, s], [-s, c]])
        rp = pts @ R.T
        mn, mx = rp.min(axis=0), rp.max(axis=0)
        area = float(np.prod(mx - mn))
        if best is None or area < best[0]:
            best = (area, R, mn, mx)
    _, R, mn, mx = best
    ext = mx - mn
    centre = (mn + mx) / 2
    # axes in image space: rows of R are the rotated x / y axes
    u, v = R[0], R[1]
    long_, short = ext[0], ext[1]
    if short > long_:
        u, v, long_, short = v, u, short, long_
    if u[0] < 0:
        u = -u
    if v[1] < 0:
        v = -v
    cimg = centre @ R
    corners = np.array([cimg - u * long_ / 2 - v * short / 2, cimg + u * long_ / 2 - v * short / 2,
                        cimg + u * long_ / 2 + v * short / 2, cimg - u * long_ / 2 + v * short / 2])
    return corners, float(long_), float(short)


def _homography_coeffs(dst: np.ndarray, src: np.ndarray) -> list[float]:
    """PIL PERSPECTIVE coefficients mapping OUTPUT (dst) points to INPUT (src) points."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    return [float(c) for c in np.linalg.solve(np.array(A, np.float64), np.array(b, np.float64))]


def warp_quad(rgb: np.ndarray, quad: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Upright crop of a quad (TL,TR,BR,BL) by perspective transform. ``pad`` grows the quad
    by that fraction of its height on every side first (a loose box is kinder to the recogniser
    than one that clips a glyph). Vertical text (taller than 1.5x wide) is rotated upright."""
    q = quad.astype(np.float64).copy()
    if pad:
        c = q.mean(axis=0)
        w, h = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
        u = (q[1] - q[0]) / max(w, 1e-6)
        v = (q[3] - q[0]) / max(h, 1e-6)
        L, S = w / 2 + pad * h, h / 2 + pad * h
        q = np.array([c - u * L - v * S, c + u * L - v * S, c + u * L + v * S, c - u * L + v * S])
    H, W = rgb.shape[:2]
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    w = max(2, int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))))
    h = max(2, int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    out = Image.fromarray(rgb).transform((w, h), Image.PERSPECTIVE, _homography_coeffs(dst, q),
                                         Image.BICUBIC)
    out = np.asarray(out)
    if h / w >= 1.5:
        out = np.rot90(out)
    return np.ascontiguousarray(out)


# ----------------------------------------------------------------------------------- engine
def _vocab_matrix(chars: list[str], lower_to_confusables: bool = True) -> np.ndarray:
    """[n_classes, VOCAB_SIZE] 0/1 (or fractional) matrix projecting PP-OCR classes onto our
    alphabet. Letters fold to upper case; blank and the space class go to CTC blank; every other
    symbol (CJK, punctuation, ...) goes to the explicit unreadable class '?'. A glyph that is
    damaged enough to look like a Chinese character is exactly the case to abstain on."""
    c2i = {c: i + 1 for i, c in enumerate(CHARS)}
    m = np.zeros((len(chars), VOCAB_SIZE), np.float32)
    for k, c in enumerate(chars):
        if k == 0 or c == " ":
            m[k, BLANK] = 1.0
        elif lower_to_confusables and c in ("l", "|"):
            # a lowercase l / bar on a plate is overwhelmingly an I, 1 or L
            m[k, c2i["L"]], m[k, c2i["1"]], m[k, c2i["I"]] = 0.34, 0.33, 0.33
        elif len(c) == 1 and c != UNK and c.upper() in c2i and c.isascii():
            m[k, c2i[c.upper()]] = 1.0
        else:
            m[k, c2i[UNK]] = 1.0
    return m


class PpOcrEngine:
    """Detector + recogniser sessions. Thread-safe for inference (ORT sessions are)."""

    def __init__(self, provider: str = "cpu", det_path=None, rec_path=None, charset_path=None,
                 max_det_side: int = 1920):
        self.det = make_session(str(det_path or find_model("ocr_det.onnx")), provider)
        self.rec = make_session(str(rec_path or find_model("ocr_rec.onnx")), provider)
        cs = charset_path or find_model("ocr_charset.txt")
        self.chars = [""] + open(cs, encoding="utf-8").read().splitlines() + [" "]
        self._vmat = _vocab_matrix(self.chars)
        # Nearly all 6.6k classes (CJK, symbols) map to '?'. Gather only the few classes that map to a
        # real character / blank and take '?' as the remainder: a dense [T,6625]x[6625,38] matmul was
        # ~1 s per line (it fights ONNX Runtime's busy thread pool), this is ~1 ms.
        unk = CHARS.index(UNK) + 1
        self._unk = unk
        self._keep = np.nonzero(self._vmat[:, unk] < 1.0)[0]
        self._vsmall = np.ascontiguousarray(self._vmat[self._keep])
        self._det_in = self.det.get_inputs()[0].name
        self._rec_in = self.rec.get_inputs()[0].name
        self.max_det_side = max_det_side
        self.provider = provider
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------- detection
    def detect(self, rgb: np.ndarray, *, min_side: int = LIMIT_SIDE) -> list[TextBox]:
        h, w = rgb.shape[:2]
        ratio = 1.0
        if min(h, w) < min_side:
            ratio = min_side / min(h, w)
        if max(h, w) * ratio > self.max_det_side:
            ratio = self.max_det_side / max(h, w)
        rh = max(32, int(round(int(h * ratio) / 32)) * 32)
        rw = max(32, int(round(int(w * ratio) / 32)) * 32)
        im = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.BILINEAR), np.float32) / 255.0
        x = (im[..., ::-1] - _DET_MEAN) / _DET_STD          # RapidOCR feeds BGR
        prob = self.det.run(None, {self._det_in: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        return self._boxes(prob, rw, rh, w, h)

    @staticmethod
    def _boxes(prob: np.ndarray, rw: int, rh: int, w: int, h: int) -> list[TextBox]:
        seg = prob > THRESH
        dil = seg.copy()                                     # cv2.dilate with a 2x2 kernel (looks up/left)
        dil[1:, :] |= seg[:-1, :]
        dil[:, 1:] |= seg[:, :-1]
        dil[1:, 1:] |= seg[:-1, :-1]
        lab, n = ndi.label(dil, structure=np.ones((3, 3)))
        out: list[TextBox] = []
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            m = lab[sl] == i
            if m.sum() < 8:
                continue
            ys, xs = np.nonzero(m)
            pts = np.stack([xs + sl[1].start, ys + sl[0].start], axis=1).astype(np.float64)
            score = float(prob[sl][m & seg[sl]].mean()) if (m & seg[sl]).any() else 0.0
            if score < BOX_THRESH:
                continue
            corners, long_, short = min_area_rect(pts)
            if short < MIN_SIZE:
                continue
            d = long_ * short * UNCLIP / (2 * (long_ + short))
            c = corners.mean(axis=0)
            u = (corners[1] - corners[0]) / max(long_, 1e-6)
            v = (corners[3] - corners[0]) / max(short, 1e-6)
            L, S = long_ / 2 + d, short / 2 + d
            q = np.array([c - u * L - v * S, c + u * L - v * S, c + u * L + v * S, c - u * L + v * S])
            if min(2 * L, 2 * S) < MIN_SIZE + 2:
                continue
            q[:, 0] = np.clip(q[:, 0] * (w / rw), 0, w - 1)
            q[:, 1] = np.clip(q[:, 1] * (h / rh), 0, h - 1)
            tb = TextBox(q, score)
            if tb.width <= 3 or tb.height <= 3:
                continue
            out.append(tb)
        out.sort(key=lambda b: (b.center[1], b.center[0]))
        return out

    # ------------------------------------------------------------------------ recognition
    def rec_probs(self, crop: np.ndarray) -> np.ndarray:
        """Raw PP-OCR softmax [T, n_classes] for one upright RGB crop (height normalised to 48)."""
        return self.rec_probs_many([crop])[0]

    def rec_probs_many(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """Batched recognition. Crops are grouped by their normalised width so no padding (which
        changes the posteriors) is ever added to a crop."""
        prepared, widths = [], []
        for c in crops:
            h, w = c.shape[:2]
            nw = min(1600, max(8, int(math.ceil(REC_H * w / max(h, 1)))))
            im = np.asarray(Image.fromarray(c).resize((nw, REC_H), Image.BILINEAR), np.float32)
            prepared.append(((im[..., ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1))
            widths.append(nw)
        out: list[np.ndarray | None] = [None] * len(crops)
        groups: dict[int, list[int]] = {}
        for i, nw in enumerate(widths):
            groups.setdefault(nw, []).append(i)
        for nw, idx in groups.items():
            batch = np.stack([prepared[i] for i in idx]).astype(np.float32)
            probs = self.rec.run(None, {self._rec_in: batch})[0]
            for k, i in enumerate(idx):
                out[i] = probs[k]
        return out  # type: ignore[return-value]

    def to_logp(self, probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Project PP-OCR probabilities onto the plate/VIN vocabulary -> log-probs [T, VOCAB_SIZE]."""
        kept = np.ascontiguousarray(probs[:, self._keep], dtype=np.float32)
        p = kept @ self._vsmall
        p[:, self._unk] = np.maximum(0.0, 1.0 - kept.sum(axis=1))
        p = np.clip(p, 1e-9, None)
        lp = np.log(p / p.sum(axis=1, keepdims=True)).astype(np.float64)
        if temperature != 1.0:
            lp = lp / temperature
            lp -= lp.max(axis=1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
        return lp

    def rec_logp(self, crop: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return self.to_logp(self.rec_probs(crop), temperature)

    def greedy(self, probs: np.ndarray) -> tuple[str, float]:
        """RapidOCR-style best-path text and mean confidence (used by the baseline reader)."""
        idx, conf = probs.argmax(-1), probs.max(-1)
        out, cs, prev = [], [], 0
        for t, k in enumerate(idx):
            if k != 0 and k != prev:
                out.append(self.chars[k] if k < len(self.chars) else "")
                cs.append(conf[t])
            prev = k
        return "".join(out), float(sum(cs) / (len(cs) + 1)) if cs else 0.0

    # ------------------------------------------------------------- RapidOCR-compatible API
    def read_lines(self, rgb: np.ndarray) -> list[dict]:
        """What ``rapidocr_onnxruntime.RapidOCR()(img)`` produced for the old backend: text lines
        with confidence and axis-aligned geometry, sorted top-to-bottom. Kept so the old
        pipeline (greedy text + regex) can be benchmarked against the new one on identical models."""
        boxes = self.detect(rgb)
        crops = [warp_quad(rgb, b.quad) for b in boxes]
        lines = []
        for b, probs in zip(boxes, self.rec_probs_many(crops) if crops else []):
            text, conf = self.greedy(probs)
            if conf < TEXT_SCORE:
                continue
            x0, y0, x1, y1 = b.bbox
            lines.append({"text": text, "confidence": round(conf, 3), "cy": (y0 + y1) / 2, "x": float(x0),
                          "x1": float(x1), "y0": float(y0), "h": float(y1 - y0)})
        lines.sort(key=lambda ln: (round(ln["cy"] / max(ln["h"], 1)), ln["x"]))
        return lines


_default: PpOcrEngine | None = None
_default_lock = threading.Lock()


def default_engine() -> PpOcrEngine:
    """Process-wide lazily created CPU engine."""
    global _default
    with _default_lock:
        if _default is None:
            _default = PpOcrEngine()
        return _default
