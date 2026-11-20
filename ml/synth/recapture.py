"""Simulated presentation attacks: a photograph of a screen, a photograph of a print.

Physics modelled (and its limits - this is a *simulation*; it is not real devices):

Screen   the display is a lattice of sub-pixels with dark gaps; the camera samples it on its own
         grid, so when the display pitch is within a few camera pixels the two grids alias into
         moire and chroma stripes. Modelled: RGB / BGR stripe and OLED-PenTile layouts, black
         matrix, sub-pixel supersampling (2x2: deliberately a weak anti-alias filter, like a
         real sensor), panel black level and gamma, white-point shift, backlight gradient, viewing
         tilt, glass reflections, optional bezel, camera blur/sharpening/noise/JPEG, and
         rolling-shutter banding.
Print    clustered-dot CMY halftone with screen angles (rosettes) on paper with fibre texture and a
         compressed gamut; or a smooth glossy photo print (no dots, lower contrast/saturation,
         specular glare). Then the same camera chain.

Not modelled: specific phone/panel models, PWM dimming, HDR/ISP tone mapping, Moire-suppression
software in newer cameras, and screens filmed so that the lattice is not resolved at all.
Detection quality measured on this simulator says nothing about those cases: real attack
captures from real devices are required (see docs/CAPTURE_PROTOCOL.md).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from synth import degrade as dg


def _smooth_box(x, centre, half, soft):
    return np.clip(0.5 - (np.abs(x - centre) - half) / soft, 0.0, 1.0)


def camera_chain(img: np.ndarray, rng, s: float, tags: list[str] | None = None) -> np.ndarray:
    """The part of the pipeline every genuine capture goes through: lens blur, sensor noise, JPEG.
    Applied to BONA FIDE images too, so a classifier cannot win by spotting 'has noise/JPEG'."""
    tags = tags if tags is not None else []
    out = img.astype(np.float32)
    if rng.random() < 0.6:
        out = dg.blur(out, rng, s * 0.5)
        tags.append("blur")
    out = dg.noise(out, rng, s * 0.6)
    out = dg.jpeg(out, rng, s)
    return np.clip(out, 0, 255)


def _banding(img: np.ndarray, rng) -> np.ndarray:
    h = img.shape[0]
    period = rng.uniform(18, 140)
    a = rng.uniform(0.02, 0.12)
    g = 1 + a * np.sin(2 * np.pi * np.arange(h) / period + rng.uniform(0, 6.28))
    return np.clip(img * g.astype(np.float32)[:, None, None], 0, 255)


def _reflection(h, w, rng):
    """Glass reflection: soft bright blobs / a window-shaped patch, low alpha."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(0.1, 0.5) * w, rng.uniform(0.1, 0.5) * h
        m = np.maximum(m, np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)))
    if rng.random() < 0.4:  # window pane edge
        edge = rng.uniform(0.2, 0.8) * w
        m = np.maximum(m, np.clip((xx - edge) / (0.05 * w), 0, 1) * rng.uniform(0.3, 1.0))
    return m


def screen_recapture(src: np.ndarray, rng: np.random.Generator, pitch: tuple[float, float] = (0.25, 3.2)):
    """Photograph of a display showing ``src`` (HxWx3 uint8). Returns (uint8 image, meta)."""
    h, w = src.shape[:2]
    d = float(np.exp(rng.uniform(np.log(pitch[0]), np.log(pitch[1]))))        # camera px per display px
    th = np.deg2rad(rng.uniform(-8, 8))
    tilt = rng.uniform(-0.05, 0.05, 2)
    layout = str(rng.choice(["rgb", "bgr", "pentile"]))
    fill_u, fill_v = rng.uniform(0.62, 0.92), rng.uniform(0.80, 0.97)
    soft = rng.uniform(0.06, 0.14)
    # Below ~0.8 camera px per display px one camera pixel spans several display pixels (a whole
    # laptop screen squeezed into a 720p frame is ~0.35): supersample more so the lattice is
    # averaged out like a real lens would, leaving the smooth, blurred, glare-and-tint screen look.
    S = 2 if d >= 0.8 else 4
    dh, dw = int(np.ceil(h / d)) + 6, int(np.ceil(w / d)) + 6
    content = np.asarray(Image.fromarray(src).resize((dw, dh), Image.BILINEAR), np.float32) / 255.0
    lin = content ** 2.2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ox, oy = rng.uniform(0, 1, 2)
    c_, s_ = np.cos(th), np.sin(th)
    acc = np.zeros((h, w, 3), np.float32)
    for sy in range(S):
        for sx in range(S):
            x = xx + (sx + 0.5) / S - 0.5 - w / 2
            y = yy + (sy + 0.5) / S - 0.5 - h / 2
            xr, yr = x * c_ - y * s_, x * s_ + y * c_
            pf = 1 + tilt[0] * xr / w + tilt[1] * yr / h
            u, v = xr / (d * pf) + dw / 2 + ox, yr / (d * pf) + dh / 2 + oy
            i, j = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
            fu, fv = u - i, v - j
            pix = lin[np.clip(j, 0, dh - 1), np.clip(i, 0, dw - 1)]
            mv = _smooth_box(fv, 0.5, fill_v / 2, soft)
            if layout == "pentile":
                even = (i % 2 == 0)
                mr = _smooth_box(fu, 0.25, 0.22, soft) * even
                mb = _smooth_box(fu, 0.25, 0.22, soft) * (~even)
                mg = _smooth_box(fu, 0.75, 0.17, soft)
                masks = (mr, mg, mb)
            else:
                order = (0, 1, 2) if layout == "rgb" else (2, 1, 0)
                masks = tuple(_smooth_box(fu, (2 * order[c] + 1) / 6.0, fill_u / 6.0, soft) for c in range(3))
            for c in range(3):
                acc[..., c] += pix[..., c] * masks[c] * mv
    L = acc / (S * S)
    L = L / (max(L.mean(), 1e-4)) * np.float32(np.mean(lin))            # keep average brightness of the content

    black = rng.uniform(0.004, 0.04)
    L = black + (1 - black) * np.clip(L, 0, 4)
    L = L * np.float32(rng.uniform(0.7, 1.35))
    L = L * np.array([rng.uniform(0.92, 1.0), rng.uniform(0.96, 1.0), rng.uniform(1.0, 1.1)], np.float32)   # cool white point
    grad = 1 + rng.uniform(-0.08, 0.08) * ((xx / w - 0.5) * np.cos(th) + (yy / h - 0.5) * np.sin(th)) * 2
    L = L * grad[..., None]
    tags = ["screen", layout]
    if rng.random() < 0.6:
        L = L + _reflection(h, w, rng)[..., None] * np.float32(rng.uniform(0.02, 0.25))
        tags.append("reflection")
    out = 255 * np.clip(L, 0, 1.5) ** (1 / 2.2)
    out = np.clip(out, 0, 255)
    out = ndi.gaussian_filter(out, (rng.uniform(0.3, 1.1),) * 2 + (0,), truncate=2.5)
    if rng.random() < 0.3:  # phone ISP sharpening
        out = np.clip(out + rng.uniform(0.3, 1.0) * (out - ndi.gaussian_filter(out, (1.2, 1.2, 0), truncate=2.0)), 0, 255)
        tags.append("sharpen")
    if rng.random() < 0.25:
        k = rng.uniform(0.04, 0.12)
        bz = np.zeros((h, w), bool)
        bz[: int(k * h)] = bz[-int(k * h) :] = True
        bz[:, : int(k * w)] = bz[:, -int(k * w) :] = True
        out[bz] = rng.uniform(5, 45)
        tags.append("bezel")
    if rng.random() < 0.2:
        out = _banding(out, rng)
        tags.append("banding")
    out = camera_chain(out, rng, rng.uniform(0.1, 0.5))
    return out.astype(np.uint8), {"pitch": round(d, 2), "layout": layout, "tags": tags}


def print_recapture(src: np.ndarray, rng: np.random.Generator):
    """Photograph of a printed copy: halftone (office laser/inkjet) or smooth glossy photo."""
    h, w = src.shape[:2]
    mode = str(rng.choice(["halftone", "halftone", "glossy"]))
    rgb = ndi.gaussian_filter(src.astype(np.float32) / 255.0, (rng.uniform(0.4, 1.4),) * 2 + (0,), truncate=2.5)
    paper = np.array([245, 242, 232], np.float32) + rng.normal(0, 4, 3).astype(np.float32)
    tags = ["print", mode]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if mode == "halftone":
        p = rng.uniform(2.4, 6.5)
        ink = []
        for ch, ang in zip(1 - rgb.transpose(2, 0, 1), (15, 75, 0)):
            a = np.deg2rad(ang + rng.uniform(-6, 6))
            xr, yr = xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)
            screen = 0.5 + 0.25 * (np.cos(2 * np.pi * xr / p) + np.cos(2 * np.pi * yr / p))
            ink.append(ndi.gaussian_filter(1 / (1 + np.exp(-(ch - screen) * 14)), rng.uniform(0.3, 0.8), truncate=2.5))
        dens = rng.uniform(0.85, 0.97)
        out = paper * (1 - dens * np.stack(ink, -1))
    else:
        g = rgb.mean(-1, keepdims=True)
        sat = rng.uniform(0.7, 0.95)
        out = (g + (rgb - g) * sat)
        out = rng.uniform(0.03, 0.08) + out * rng.uniform(0.82, 0.95)
        out = out * paper
    fibre = ndi.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8) * rng.uniform(1.5, 5)
    out = out + fibre[..., None]
    if mode == "glossy" and rng.random() < 0.5:
        out, _ = dg.glare(out, np.zeros((h, w), np.float32), rng, rng.uniform(0.2, 0.7))
        tags.append("glare")
    grad = 1 + rng.uniform(-0.12, 0.12) * ((xx / w - 0.5) + (yy / h - 0.5))
    out = np.clip(out * grad[..., None], 0, 255)
    out = camera_chain(out, rng, rng.uniform(0.1, 0.5))
    return out.astype(np.uint8), {"mode": mode, "tags": tags}


def bona_fide(src: np.ndarray, rng: np.random.Generator):
    """A genuine capture: the source photo through the same camera chain the attacks end with."""
    out = camera_chain(src, rng, rng.uniform(0.1, 0.5))
    return out.astype(np.uint8), {"tags": ["real"]}
