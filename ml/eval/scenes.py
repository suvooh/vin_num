"""Synthetic full-frame scenes with exact ground truth, for end-to-end reader benchmarks.

    from eval.scenes import make_scenes
    for sc in make_scenes(100, seed=3):
        sc.img, sc.truth, sc.domain, sc.quad, sc.profile, sc.tags

What a scene is
  plate  a real Open Images car photo (CC BY 2.0, ``data/openimages``) whose own, non-Indian,
         labelled plate is painted out and replaced by a *degraded synthetic Indian plate*
         (``synth.plates`` + ``synth.degrade``: glare, dirt, occlusion, perspective, blur,
         lighting, JPEG), feathered in and brightness-matched to the car body it sits on.
  vin    a degraded synthetic VIN strip (label / engraved / dot-matrix / stencil) on a larger
         textured metal-or-paint canvas with neighbouring clutter lines (GVWR, MFD date, an
         engine number that is shaped like a short chassis number, ...), mild global
         rotation / perspective and JPEG.

``truth`` is the VALUE a reader should return (a VIN's "VIN" / "CHASSIS NO" label prefix is
stripped; it is kept in ``meta['label_prefix']``). A ``?`` in ``truth`` marks a character the
synthetic damage hid (>65% of its glyph covered): it is not recoverable from pixels, so an
honest reader should print ``?`` there, and a reader that prints a letter is hallucinating.

Honest limits - read before quoting a number
  * Everything but the background photograph is synthetic and written by the same people who
    wrote the readers: this is a plumbing / RELATIVE-comparison tool (old pipeline vs new,
    slice vs slice), not a field accuracy figure.
  * Backgrounds are ~640 px Flickr renditions, upscaled (soft) so that the pasted plate is
    150-420 px wide. The plate itself is sharper than the photo around it.
  * The Indian plate is pasted over a non-Indian car in non-Indian scenes; real 2-wheelers,
    angled plates on bumpers with shadows, dust, stickers, frames and HSRP hologram glare are
    only approximated.
  * No real VIN photographs exist publicly; VIN scenes follow how the marks look, not data.

Deliberate deviations from ``synth.sampler.sample_plate_whole``: this module calls
``render_plate`` + ``synth.sampler._degrade`` directly (a private helper) instead, because
``sample_plate_whole`` pads the plate with a random *vehicle-body colour* margin (it imitates a
loose detector box) which would show up as a coloured rectangle once pasted onto a real car.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from synth.common import font_bank, load_font, size_for_cap_height
from synth.plates import render_plate
from synth.sampler import _degrade, sample_vin_whole
from synth.vin import _metal, _paint

ML_ROOT = Path(__file__).resolve().parents[1]
OPENIMAGES = ML_ROOT / "data" / "openimages"
PLATE_CLASS = "/m/01jfm_"           # Open Images "Vehicle registration plate"
MAX_SIDE = 1600                     # the backend caps frames at this before OCR
PLATE_W_RANGE = (150.0, 420.0)      # width of the pasted plate in the final frame, px
VIN_CAP_RANGE = (18.0, 60.0)        # target capital-letter height of the VIN text, px
_VIN_NATIVE_CAP = 28.0              # synth.vin draws caps at 24-32 px

Box = tuple[float, float, float, float]


@dataclass
class Scene:
    img: np.ndarray                 # uint8 HxWx3 full frame
    domain: str                     # "plate" | "vin"
    truth: str                      # value to read; '?' = a character synthetic damage hid
    profile: str                    # clean | mild | moderate | severe
    tags: list[str]                 # damage / optics tags from synth
    quad: np.ndarray                # 4x2 float, TL TR BR BL of the plate / VIN region, image px
    meta: dict = field(default_factory=dict)

    @property
    def bbox(self) -> Box:
        return (float(self.quad[:, 0].min()), float(self.quad[:, 1].min()),
                float(self.quad[:, 0].max()), float(self.quad[:, 1].max()))


# --------------------------------------------------------------------------- geometry
def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src for four point pairs."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    h = np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def _pil_coeffs(h_src_to_dst: np.ndarray) -> tuple:
    """PIL's PERSPECTIVE transform wants the OUTPUT -> INPUT mapping."""
    inv = np.linalg.inv(h_src_to_dst)
    inv = inv / inv[2, 2]
    return tuple(float(v) for v in inv.ravel()[:8])


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def _rect(w: float, h: float) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)


def _luma(a: np.ndarray) -> np.ndarray:
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


# ------------------------------------------------------------------------- backgrounds
def _read_plate_boxes(root: Path) -> dict[str, list[Box]]:
    """image id -> normalised plate boxes (x0, y0, x1, y1), ``ocr_locate_eval``'s rules."""
    boxes: dict[str, list[Box]] = {}
    path = root / "val_bbox.csv"
    if not path.is_file():
        return boxes
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["LabelName"] == PLATE_CLASS and r["Confidence"] == "1" and r["IsGroupOf"] == "0":
                boxes.setdefault(r["ImageID"], []).append(
                    tuple(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")))
    return boxes


def _image_size(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:
        return im.size


def load_backgrounds(root: Path | str = OPENIMAGES, min_plate_w: float = 80.0) -> list[tuple[Path, list[Box]]]:
    """Real car photos with at least one labelled plate >= ``min_plate_w`` px wide.
    Returns ``[(image path, [plate boxes in px of the ORIGINAL image])]`` sorted by name.
    Empty list if the Open Images folder is not present."""
    root = Path(root)
    gt = _read_plate_boxes(root)
    out = []
    for p in sorted((root / "images").glob("*.jpg")):
        if p.stem not in gt:
            continue
        w, h = _image_size(p)
        usable = [(x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1 in gt[p.stem]
                  if (x1 - x0) * w >= min_plate_w]
        if usable:
            out.append((p, usable))
    return out


# ----------------------------------------------------------------------- plate scenes
def _fill_box(img: np.ndarray, box: Box, rng: np.random.Generator, pad: float = 0.12) -> None:
    """Paint a box out (in place) with a Coons-patch blend of the colours just outside its four
    sides, plus matching grain, feathered into the photo. Used to remove the real plate."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    x0, y0, x1, y1 = (int(round(v)) for v in (x0 - px, y0 - py, x1 + px, y1 + py))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return
    m = max(3, int(0.3 * bh))

    def strip(a):
        return a.astype(np.float32) if a.size else None

    top = strip(img[max(0, y0 - m):y0, x0:x1].mean(axis=0)) if y0 > 0 else None
    bot = strip(img[y1:min(H, y1 + m), x0:x1].mean(axis=0)) if y1 < H else None
    left = strip(img[y0:y1, max(0, x0 - m):x0].mean(axis=1)) if x0 > 0 else None
    right = strip(img[y0:y1, x1:min(W, x1 + m)].mean(axis=1)) if x1 < W else None
    ring = [s for s in (top, bot, left, right) if s is not None]
    if not ring:
        return
    mean_c = np.mean([s.mean(axis=0) for s in ring], axis=0)
    top = top if top is not None else np.tile(mean_c, (bw, 1))
    bot = bot if bot is not None else np.tile(mean_c, (bw, 1))
    left = left if left is not None else np.tile(mean_c, (bh, 1))
    right = right if right is not None else np.tile(mean_c, (bh, 1))
    u = np.linspace(0, 1, bw, dtype=np.float32)[None, :, None]
    v = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None]
    corners = (top[0] * (1 - v) * (1 - u) + top[-1] * (1 - v) * u + bot[0] * v * (1 - u) + bot[-1] * v * u)
    fill = ((1 - v) * top[None, :, :] + v * bot[None, :, :]
            + (1 - u) * left[:, None, :] + u * right[:, None, :] - corners)
    ring_px = np.concatenate([s.reshape(-1, 3) for s in ring])
    grain = float(np.clip(ring_px.std(axis=0).mean() * 0.15, 1.0, 5.0))
    fill = fill + rng.standard_normal(fill.shape).astype(np.float32) * grain
    a = np.zeros((bh, bw), np.float32)
    ins = max(1, min(bw, bh) // 8)
    a[ins:-ins, ins:-ins] = 1.0
    a = ndi.gaussian_filter(a, max(1.0, ins / 2.0))[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + fill * a


def _ring_luma(img: np.ndarray, box: Box) -> float:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    ex, ey = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    a, b = max(0, int(x0 - ex)), max(0, int(y0 - ey))
    c, d = min(W, int(x1 + ex)), min(H, int(y1 + ey))
    m = np.ones((d - b, c - a), bool)
    m[int(y0) - b: int(y1) - b, int(x0) - a: int(x1) - a] = False
    return float(_luma(img[b:d, a:c])[m].mean()) if m.any() else float(_luma(img).mean())


def _scene_post(img: np.ndarray, rng: np.random.Generator, meta: dict) -> np.ndarray:
    """Optional frame-level softness + JPEG, as a phone / messenger would add."""
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if rng.random() < 0.25:
        sig = float(rng.uniform(0.4, 0.9))
        im = im.filter(ImageFilter.GaussianBlur(sig))
        meta["scene_blur"] = round(sig, 2)
    if rng.random() < 0.6:
        q = int(rng.integers(55, 93))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        meta["scene_jpeg_q"] = q
    return np.asarray(im, np.uint8)


def make_plate_scene(rng: np.random.Generator, bg_rgb: np.ndarray, bbox: Box,
                     other_boxes: list[Box] | None = None, *, spec: dict | None = None,
                     squeeze: float = 1.0) -> Scene:
    """Paste a degraded synthetic Indian plate over the real plate at ``bbox`` (x0, y0, x1, y1 in
    ``bg_rgb`` px). ``other_boxes``: further plates in the photo (px of ``bg_rgb``) that are
    painted out so the pasted plate is the only one - pass ``None`` to keep them as distractors.
    The background is resized so the pasted plate is 150-420 px wide and the longest side <= 1600.

    ``spec``: a ``synth.text.sample_plate``-style dict to render a given number instead of a random one.
    ``squeeze`` < 1 pastes the plate horizontally CONDENSED (its width / height ratio multiplied by
    ``squeeze``): a narrow-font HSRP, or one seen from an angle. Real HSRP crops of 2.6 : 1 (``squeeze``
    ~0.62) starve the recogniser's CTC of time steps (see ``vinplate.scene.normalise_aspect``). The defaults
    (``None`` / 1.0) leave every random draw, and so every benchmark scene, unchanged."""
    bh0, bw0 = bg_rgb.shape[:2]
    bw_box = max(1.0, bbox[2] - bbox[0])
    target_w = float(np.exp(rng.uniform(np.log(PLATE_W_RANGE[0]), np.log(PLATE_W_RANGE[1]))))
    s = min(target_w / bw_box, MAX_SIDE / max(bh0, bw0))
    new = (max(16, int(round(bw0 * s))), max(16, int(round(bh0 * s))))
    resample = Image.LANCZOS if s >= 1 else Image.BICUBIC
    bg = np.asarray(Image.fromarray(bg_rgb).resize(new, resample), np.float32)
    sx, sy = new[0] / bw0, new[1] / bh0
    box = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
    bw, bh = box[2] - box[0], box[3] - box[1]
    luma_ring = _ring_luma(bg, box)

    for ob in other_boxes or []:
        o = (ob[0] * sx, ob[1] * sy, ob[2] * sx, ob[3] * sy)
        if o[2] - o[0] >= 18:
            _fill_box(bg, o, rng)
    _fill_box(bg, box, rng)

    # --- the synthetic plate, degraded exactly like the training / eval lines but WITHOUT the
    # vehicle-body margin that sample_plate_whole adds (see module docstring).
    r = render_plate(rng, spec)
    img8, texts, profile, severity, tags = _degrade(r, rng)
    truth = "".join(texts)
    lh, lw = img8.shape[:2]
    aspect = lw / lh * squeeze
    two_row = len(texts) == 2
    if two_row:
        ph = bh * float(rng.uniform(1.4, 2.0))
        pw = ph * aspect
        if pw > 1.5 * bw:
            pw = 1.5 * bw
            ph = pw / aspect
    else:
        pw = bw * float(rng.uniform(0.9, 1.2))
        ph = pw / aspect

    layer = img8.astype(np.float32)
    # brightness: pull the plate part-way towards the light level of the car it sits on
    own = float(_luma(layer).mean())
    gain = float(np.clip((luma_ring / max(own, 1.0)) ** 0.4, 0.75, 1.3))
    layer = np.clip(layer * gain, 0, 255)

    inset = max(1, int(0.02 * lh))
    alpha = np.zeros((lh, lw), np.float32)
    alpha[inset:-inset, inset:-inset] = 1.0
    alpha = ndi.gaussian_filter(alpha, max(0.8, 0.012 * lh))

    pad = 0.15 * max(pw, ph)
    Wc, Hc = int(math.ceil(pw + 2 * pad)), int(math.ceil(ph + 2 * pad))
    theta = float(np.clip(rng.normal(0, math.radians(2.0)), -math.radians(5), math.radians(5)))
    c, sn = math.cos(theta), math.sin(theta)
    corners = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    corners = corners @ np.array([[c, sn], [-sn, c]])
    corners = corners + rng.uniform(-1, 1, (4, 2)) * np.array([0.02 * pw, 0.04 * ph])
    dst = corners + np.array([Wc / 2, Hc / 2])
    h_fwd = _homography(_rect(lw, lh), dst)
    co = _pil_coeffs(h_fwd)
    rgb_w = np.asarray(Image.fromarray(np.clip(layer, 0, 255).astype(np.uint8)).transform(
        (Wc, Hc), Image.PERSPECTIVE, co, Image.BICUBIC), np.float32)
    a_w = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).transform(
        (Wc, Hc), Image.PERSPECTIVE, co, Image.BICUBIC), np.float32) / 255.0

    H, W = bg.shape[:2]
    cx = (box[0] + box[2]) / 2 + rng.uniform(-0.05, 0.05) * bw
    cy = (box[1] + box[3]) / 2 + rng.uniform(-0.08, 0.08) * bh
    ox = int(round(min(max(cx - Wc / 2, 0), max(0, W - Wc))))
    oy = int(round(min(max(cy - Hc / 2, 0), max(0, H - Hc))))
    x_a, y_a = max(0, ox), max(0, oy)
    x_b, y_b = min(W, ox + Wc), min(H, oy + Hc)
    sl_bg = (slice(y_a, y_b), slice(x_a, x_b))
    sl_ly = (slice(y_a - oy, y_b - oy), slice(x_a - ox, x_b - ox))
    am = a_w[sl_ly][..., None]
    bg[sl_bg] = bg[sl_bg] * (1 - am) + rgb_w[sl_ly] * am

    meta: dict = {"bg_scale": round(float(s), 3), "rows": len(texts), "severity": round(severity, 2),
                  "plate_w_px": round(float(pw), 1), "plate_layout": r.meta.get("layout"),
                  "plate_style": r.meta.get("style"), "painted": r.meta.get("painted")}
    if squeeze != 1.0:
        meta["squeeze"] = squeeze
    out = _scene_post(bg, rng, meta)
    quad = dst + np.array([ox, oy])
    return Scene(out, "plate", truth, profile, list(tags), quad, meta)


# ------------------------------------------------------------------------- VIN scenes
_CLUTTER = [
    "GVWR 1950 KG", "GAWR FRONT 1020 KG", "MFD 06/2023", "MFG DATE 11/2022", "MADE IN INDIA",
    "BS VI", "PAINT CODE NH-731P", "TYRE 185/65 R15", "TYRE PRESSURE 32 PSI", "SEATING CAP 5",
    "COLOUR SILVER", "MODEL ZXI PLUS", "ENGINE CAPACITY 1197 CC", "AIR CONDITIONED",
]
_ALNUM = list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")


def _clutter_text(rng: np.random.Generator) -> tuple[str, bool]:
    r = rng.random()
    if r < 0.3:  # an engine number: shaped like a short chassis number - a deliberate decoy
        n = int(rng.integers(8, 12))
        return "ENG NO " + "".join(rng.choice(_ALNUM, n)), True
    if r < 0.4:
        return "MODEL " + "".join(rng.choice(_ALNUM, int(rng.integers(5, 9)))), False
    return str(rng.choice(_CLUTTER)), False


def make_vin_scene(rng: np.random.Generator) -> Scene:
    """A degraded synthetic VIN strip on a textured canvas with clutter lines around it."""
    v = sample_vin_whole(rng)
    prefix = str(v.meta.get("prefix") or "")
    truth = v.text[len(prefix.replace(" ", "")):] if prefix else v.text
    vh0, vw0 = v.img.shape[:2]

    cap = float(np.exp(rng.uniform(np.log(VIN_CAP_RANGE[0]), np.log(VIN_CAP_RANGE[1]))))
    f = min(cap / _VIN_NATIVE_CAP, 1500.0 / vw0)
    vw, vh = max(32, int(round(vw0 * f))), max(8, int(round(vh0 * f)))
    strip = Image.fromarray(v.img).resize((vw, vh), Image.BICUBIC if f >= 1 else Image.LANCZOS)
    cap_px = _VIN_NATIVE_CAP * f

    W = int(np.clip(vw * rng.uniform(1.15, 1.5), max(vw + 80, 700), 1600))
    H = int(np.clip(max(vh * rng.uniform(3.0, 5.0), 6.0 * cap_px, 480), 480, 1200))
    if rng.random() < 0.5:
        canvas = _metal(rng, W, H)
        tex = "metal"
    else:
        canvas, tex = _paint(rng, W, H)
    lum = float(_luma(canvas).mean())
    canvas = np.clip(canvas + np.linspace(-1, 1, H, dtype=np.float32)[:, None, None]
                     * rng.uniform(4, 20), 0, 255)
    im = Image.fromarray(canvas.astype(np.uint8))
    d = ImageDraw.Draw(im)

    sx = int(rng.integers(max(1, int(0.03 * W)), max(2, W - vw - int(0.03 * W) + 1)))
    sy = int(rng.uniform(0.3, 0.6) * (H - vh))
    gap = 1.3 * cap_px
    decoy = False
    n_clutter = int(rng.integers(1, 5))
    y_above, y_below = sy - gap, sy + vh + gap
    for k in range(n_clutter):
        side = "above" if (rng.random() < 0.5) else "below"
        text, is_decoy = _clutter_text(rng)
        c_cap = cap_px * float(rng.uniform(0.7, 1.1))
        role = str(rng.choice(["print", "print", "stamp"]))
        fpath = str(font_bank().pick(role, rng))
        weight = 500 if "[" in fpath else 0
        font = load_font(fpath, size_for_cap_height(fpath, c_cap, weight), weight)
        while font.getlength(text) > 0.95 * W and c_cap > 8:
            c_cap *= 0.85
            font = load_font(fpath, size_for_cap_height(fpath, c_cap, weight), weight)
        if side == "above":
            y = y_above
            if y - c_cap < 0.02 * H:
                side, y = "below", y_below
        else:
            y = y_below + c_cap
            if y > 0.98 * H:
                side, y = "above", y_above
                if y - c_cap < 0.02 * H:
                    continue
        x = int(np.clip(sx + rng.uniform(-0.1, 0.35) * vw, 0.02 * W, max(0.02 * W, W - font.getlength(text) - 0.02 * W)))
        if lum < 90 or rng.random() < 0.2:     # light ink on dark metal / paint
            ink = int(np.clip(lum + rng.uniform(70, 110), 0, 255))
        else:
            ink = int(np.clip(lum - rng.uniform(55, 95), 0, 255))
        d.text((x, y), text, font=font, fill=(ink, ink, ink), anchor="ls")
        step = c_cap * float(rng.uniform(1.8, 2.6))
        if side == "above":
            y_above -= step
        else:
            y_below = y + step
        decoy = decoy or is_decoy
    im.paste(strip, (sx, sy))
    sheet = np.asarray(im, np.float32)

    # global rotation / mild perspective about the centre, then crop away the fill wedges
    th = float(np.clip(rng.normal(0, math.radians(1.5)), -math.radians(4), math.radians(4)))
    c, sn = math.cos(th), math.sin(th)
    corners = _rect(W, H) - np.array([W / 2, H / 2])
    dst = corners @ np.array([[c, sn], [-sn, c]]) + np.array([W / 2, H / 2])
    dst = dst + rng.uniform(-1, 1, (4, 2)) * np.array([0.012 * W, 0.012 * H])
    h_fwd = _homography(_rect(W, H), dst)
    warped = np.asarray(Image.fromarray(np.clip(sheet, 0, 255).astype(np.uint8)).transform(
        (W, H), Image.PERSPECTIVE, _pil_coeffs(h_fwd), Image.BICUBIC,
        fillcolor=tuple(int(x) for x in sheet.reshape(-1, 3).mean(axis=0))), np.float32)
    m = int(math.ceil(np.abs(dst - _rect(W, H)).max())) + 2
    m = min(m, W // 4, H // 4)
    cropped = warped[m:H - m, m:W - m]
    quad = _apply(h_fwd, _rect(vw, vh) + np.array([sx, sy])) - m
    meta: dict = {"label_prefix": prefix, "texture": tex, "decoy_engine_no": decoy, "cap_px": round(cap_px, 1),
                  "severity": v.meta.get("severity"), "vin_style": v.meta.get("style"),
                  "legacy": bool(v.meta.get("legacy"))}
    out = _scene_post(cropped, rng, meta)
    return Scene(out, "vin", truth, v.profile, list(v.tags), quad, meta)


# ------------------------------------------------------------------------------ driver
_BOX_CACHE: dict[Path, dict[str, list[Box]]] = {}


def make_scenes(n: int, seed: int = 0, plate_frac: float = 0.6, root: Path | str = OPENIMAGES,
                backgrounds: list[tuple[Path, list[Box]]] | None = None,
                keep_other_plates: bool = False) -> Iterator[Scene]:
    """``n`` scenes, deterministic for a given ``seed`` (each scene draws from its own
    ``default_rng((seed, i))`` so a skipped / failing scene cannot shift the others).
    Raises FileNotFoundError if plate scenes are requested but no backgrounds exist."""
    root = Path(root)
    bgs = backgrounds if backgrounds is not None else load_backgrounds(root)
    if plate_frac > 0 and not bgs:
        raise FileNotFoundError(f"no Open Images backgrounds with labelled plates under {root}")
    all_boxes = _BOX_CACHE.setdefault(root, _read_plate_boxes(root))
    master = np.random.default_rng(seed)
    produced, i = 0, 0
    while produced < n and i < 20 * n + 20:
        rng = np.random.default_rng((seed, i))
        i += 1
        want_plate = master.random() < plate_frac
        try:
            if want_plate:
                path, boxes = bgs[int(rng.integers(len(bgs)))]
                box = boxes[int(rng.integers(len(boxes)))]
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                h, w = rgb.shape[:2]
                others = None
                if not keep_other_plates:
                    others = [(x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1 in all_boxes.get(path.stem, [])
                              if abs((x0 * w + x1 * w) / 2 - (box[0] + box[2]) / 2) > 2
                              or abs((y0 * h + y1 * h) / 2 - (box[1] + box[3]) / 2) > 2]
                sc = make_plate_scene(rng, rgb, box, others)
                sc.meta["bg"] = path.name
            else:
                sc = make_vin_scene(rng)
        except Exception as exc:  # an unusable background / degenerate draw: skip, never abort the run
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            continue
        sc.meta["index"] = produced
        produced += 1
        yield sc
