"""Render Indian registration plates (canonical, un-degraded) with per-character geometry.

Covered: white / yellow / green (EV) / black (rental) / blue (diplomatic) plates, HSRP-style
(crisp, embossed, blue IND strip, border, laser code) and old painted/stylised plates
(irregular spacing, rotated glyphs, uneven stroke), 1-row (cars, trucks) and 2-row
(two-wheelers, three-wheelers, some cars). Dimensions follow the CMVR proportions I recall
(500x120 mm 1-row, 200x100 / 340x200 mm 2-row); real plates vary and the training
distribution jitters them. The exact HSRP typeface is not freely licensed, so DIN-like OFL
fonts stand in - a known domain gap, closed with real data in fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from synth.common import font_bank, jitter, load_font, size_for_cap_height
from synth.text import sample_plate

_STYLES = [
    # name, background, foreground, weight
    ("private", (248, 248, 246), (18, 18, 18), 0.62),
    ("commercial", (255, 206, 22), (18, 18, 18), 0.17),
    ("ev_private", (22, 150, 72), (246, 246, 246), 0.05),
    ("ev_commercial", (22, 150, 72), (255, 214, 40), 0.015),
    ("rental", (22, 22, 22), (255, 206, 22), 0.02),
]
_BLUE = (24, 54, 150)


@dataclass
class Rendered:
    img: np.ndarray                                   # float32 HxWx3, 0..255
    rows: list[str]                                   # text of each printed row
    boxes: list[list[tuple[int, int, int, int]]]      # per row, per char: x0,y0,x1,y1
    row_y: list[tuple[int, int]]                      # vertical extent of each row band
    textmask: np.ndarray                              # HxW float 0/1: glyph pixels
    meta: dict = field(default_factory=dict)


def _pick_style(rng, layout: str):
    if layout == "diplomatic":
        return "diplomatic", (0, 72, 172), (246, 246, 246)
    w = np.array([s[3] for s in _STYLES])
    i = int(rng.choice(len(_STYLES), p=w / w.sum()))
    return _STYLES[i][0], _STYLES[i][1], _STYLES[i][2]


def _draw_glyph(canvas: Image.Image, mask: Image.Image, ch: str, font, x: float, base: float,
                fg, rng, painted: bool, shade=None):
    """Draw one character with its left edge at x and baseline at ``base``. Returns its box."""
    l, t, r, b = font.getbbox(ch, anchor="ls")
    pad = 6
    gw, gh = r - l + 2 * pad, b - t + 2 * pad
    glyph = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(glyph).text((pad - l, pad - t), ch, font=font, fill=255, anchor="ls")
    if painted:
        ang = rng.normal(0, 2.5)
        glyph = glyph.rotate(ang, resample=Image.BICUBIC, expand=True)
        g = np.asarray(glyph, np.float32) / 255
        k = rng.choice([-1, 0, 0, 1, 1])  # uneven stroke weight
        if k > 0:
            g = ndi.grey_dilation(g, size=(2, 2))
        elif k < 0:
            g = ndi.grey_erosion(g, size=(2, 2))
        g = g * rng.uniform(0.78, 1.0)
        if rng.random() < 0.3:  # worn paint
            g = g * np.clip(1 - 0.9 * (ndi.gaussian_filter(rng.random(g.shape), 1.2) > 0.62), 0, 1)
        glyph = Image.fromarray((g * 255).astype(np.uint8))
        gw, gh = glyph.size
    ox, oy = int(round(x + l - pad)), int(round(base + t - pad))
    if shade is not None:  # emboss: darker offset copy under the glyph = pressed-metal edge
        canvas.paste(Image.new("RGB", (gw, gh), tuple(shade)), (ox + 1, oy + 1), glyph.point(lambda v: v // 2))
    canvas.paste(Image.new("RGB", (gw, gh), tuple(fg)), (ox, oy), glyph)
    mask.paste(255, (ox, oy), glyph)
    return (int(x + l), int(base + t), int(x + r), int(base + b))


def render_plate(rng: np.random.Generator, spec: dict | None = None) -> Rendered:
    spec = spec or sample_plate(rng)
    groups = list(spec["groups"])
    layout = spec["layout"]
    style_name, bg, fg = _pick_style(rng, layout)
    bg, fg = jitter(rng, bg, 6), jitter(rng, fg, 6)

    two_row = layout != "diplomatic" and len(groups) >= 3 and rng.random() < 0.4
    if two_row:
        rows_g = [groups[:2], groups[2:]] if layout != "standard" or len(groups) > 3 else [groups[:2], groups[2:]]
        ratio = rng.uniform(1.55, 2.2)
        W = 300
    else:
        rows_g = [groups]
        ratio = rng.uniform(3.6, 4.7)
        W = 480
    H = int(round(W / ratio))
    painted = rng.random() < 0.18
    hsrp = not painted

    surround = jitter(rng, (120, 120, 125), 40)
    canvas = Image.new("RGB", (W, H), surround)
    d = ImageDraw.Draw(canvas)
    r = int(0.09 * H)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=bg)
    mask = Image.new("L", (W, H), 0)

    # border
    bm = int(0.06 * H)
    if (hsrp and rng.random() < 0.8) or (painted and rng.random() < 0.35):
        bw = max(1, int(rng.uniform(0.025, 0.05) * H))
        d.rounded_rectangle([bm, bm, W - 1 - bm, H - 1 - bm], radius=max(2, r // 2), outline=fg, width=bw)
    # IND strip (blue hologram block with chakra + IND)
    x_left = bm + int(0.04 * H)
    if hsrp and rng.random() < 0.85 and layout != "diplomatic":
        sw = int((0.075 if not two_row else 0.12) * W)
        sh = int((0.58 if not two_row else 0.42) * H)
        sy = int((H - sh) * rng.uniform(0.3, 0.6))
        d.rectangle([x_left, sy, x_left + sw, sy + sh], fill=jitter(rng, _BLUE, 10))
        cx, cy, cr = x_left + sw // 2, sy + int(0.28 * sh), int(0.17 * sw)
        d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=(235, 235, 235), width=1)
        f_ind = load_font(str(font_bank().pick("hsrp", rng)), max(6, int(0.22 * sw)))
        d.text((x_left + sw // 2, sy + int(0.78 * sh)), "IND", font=f_ind, fill=(240, 240, 240), anchor="mm")
        x_left += sw + int(0.03 * W)
    # laser code + screws (clutter that is *not* part of the registration number)
    if hsrp and rng.random() < 0.6:
        f_s = load_font(str(font_bank().pick("print", rng)), max(6, int(0.07 * H)))
        code = "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"), 10))
        d.text((bm + int(0.02 * W), H - bm - int(0.02 * H)), code, font=f_s, fill=fg, anchor="ls")
    for _ in range(int(rng.integers(0, 3))):
        sx, sy2, sr = rng.uniform(0.06, 0.94) * W, rng.choice([0.14, 0.86]) * H, max(2, int(0.035 * H))
        d.ellipse([sx - sr, sy2 - sr, sx + sr, sy2 + sr], fill=jitter(rng, (130, 130, 135), 20))

    # registration text
    role = "painted" if painted else "hsrp"
    fpath = str(font_bank().pick(role, rng))
    weight = int(rng.choice([500, 600, 700, 800])) if "[" in fpath else 0
    x_right = W - bm - int(0.04 * H)
    n_rows = len(rows_g)
    band = (H - 2 * bm) / n_rows
    boxes, row_y, row_text = [], [], []
    for ri, grp in enumerate(rows_g):
        text = "".join(grp)
        cap = band * rng.uniform(0.60, 0.80) if n_rows == 2 else (H - 2 * bm) * rng.uniform(0.55, 0.78)
        gap_in = rng.uniform(0.10, 0.22)
        gap_grp = rng.uniform(0.35, 0.8)
        while True:
            size = size_for_cap_height(fpath, cap, weight)
            font = load_font(fpath, size, weight)
            adv = []
            for gi, g in enumerate(grp):
                for ci, ch in enumerate(g):
                    w_ch = font.getlength(ch)
                    sp = (gap_in if ci < len(g) - 1 else (gap_grp if gi < len(grp) - 1 else 0)) * cap
                    adv.append((ch, w_ch, sp))
            total = sum(a[1] + a[2] for a in adv)
            avail = x_right - x_left
            if total <= avail or cap < 8:
                break
            cap *= max(0.8, avail / total * 0.98)
        x = x_left + (avail - total) * rng.uniform(0.3, 0.7) if total < avail else x_left
        y_mid = bm + band * (ri + 0.5)
        base0 = y_mid + cap / 2
        rb = []
        for ch, w_ch, sp in adv:
            jy = rng.normal(0, 0.03 * cap) if painted else 0
            jx = rng.normal(0, 0.04 * cap) if painted else 0
            shade = tuple(int(v * 0.72) for v in bg) if hsrp else None
            rb.append(_draw_glyph(canvas, mask, ch, font, x + jx, base0 + jy, fg, rng, painted, shade))
            x += w_ch + sp
        boxes.append(rb)
        row_text.append(text)
        row_y.append((int(bm + band * ri), int(bm + band * (ri + 1))))

    img = np.asarray(canvas, np.float32)
    # retroreflective sheeting grain + very slight bend
    img = img + rng.standard_normal(img.shape[:2], dtype=np.float32)[..., None] * 1.5
    return Rendered(
        img=np.clip(img, 0, 255), rows=row_text, boxes=boxes, row_y=row_y,
        textmask=(np.asarray(mask) > 40).astype(np.float32),
        meta=dict(kind=spec["kind"], layout=layout, style=style_name, rows=n_rows,
                  painted=painted, text=spec["text"]),
    )
