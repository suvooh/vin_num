"""Photographic degradations for synthetic plates / VIN labels.

Two kinds of operation, applied in this order:

1. DAMAGE (glare, dirt, occlusion) - things that can hide a character. Each returns the
   pixels plus an *alpha map* of how completely it covers each pixel. The caller turns the
   alpha map into per-character visibility and labels a character ``?`` if it is hidden, so
   the network is taught to abstain rather than guess (see vinplate.vocab).
2. OPTICS / SENSOR (perspective, blur, noise, JPEG, low-res, exposure) - these degrade but
   do not delete, and are applied after the labels are fixed.

Everything takes and returns float32 HxWx3 in 0..255 and an ``np.random.Generator``; no
OpenCV (not available as a native Windows-ARM64 wheel) - PIL / numpy / scipy only.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def _grid(h, w):
    return np.mgrid[0:h, 0:w].astype(np.float32)


def _blend(img, color, m):
    """img*(1-m) + color*m, m in 0..1 with shape HxW."""
    m3 = m[..., None]
    return img * (1 - m3) + np.asarray(color, np.float32) * m3


# ------------------------------------------------------------------------------ damage
def glare(img, alpha, rng, s):
    """Flash / sun glare: elongated saturated blobs with bloom. s in 0..1 sets size/strength."""
    h, w = img.shape[:2]
    yy, xx = _grid(h, w)
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx = rng.uniform(0.03, 0.07 + 0.16 * s) * w
        sy = rng.uniform(0.35, 1.1) * h
        th = rng.uniform(0, np.pi)
        xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        mask = np.maximum(mask, np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)))
    m = np.clip(mask * rng.uniform(1.2, 2.4), 0, 1) ** rng.uniform(0.7, 1.4)
    tint = (255, 255, 255) if rng.random() < 0.7 else (255, 248, 225)
    out = _blend(img, tint, m)
    out = out + 45 * ndi.gaussian_filter(m, rng.uniform(1.5, 5), truncate=2.5)[..., None]   # bloom
    return np.clip(out, 0, 255), np.maximum(alpha, np.clip(m * 1.1, 0, 1))


def dirt(img, alpha, rng, s):
    """Mud splatter, film of dust, rain streaks, scratches."""
    h, w = img.shape[:2]
    out = img
    # low-frequency dust film (partial cover only)
    if rng.random() < 0.7:
        n = ndi.gaussian_filter(rng.random((h, w), dtype=np.float32), h * rng.uniform(0.1, 0.4), truncate=2.5)
        n = (n - n.min()) / (np.ptp(n) + 1e-6)
        film = np.clip((n - 0.4) * 2.0, 0, 1) * rng.uniform(0.15, 0.5 + 0.3 * s)
        out = _blend(out, rng.choice([[120, 105, 85], [150, 150, 150], [90, 80, 70]]), film)
        alpha = np.maximum(alpha, film * 0.8)
    # mud blobs
    blobs = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(blobs)
    for _ in range(int(rng.integers(0, 2 + int(6 * s)))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx = rng.uniform(0.01, 0.035 + 0.07 * s) * w
        ry = rng.uniform(0.05, 0.22 + 0.28 * s) * h
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.uniform(150, 255)))
    b = np.asarray(blobs, np.float32) / 255
    if b.max() > 0:
        rough = ndi.gaussian_filter(rng.random((h, w), dtype=np.float32), 1.5, truncate=2.5)
        b = ndi.gaussian_filter(b, rng.uniform(0.8, 2.5)) * (0.7 + 0.6 * rough)
        b = np.clip(b * 1.6, 0, 1)
        col = rng.choice([[85, 65, 45], [60, 50, 40], [110, 95, 70], [40, 40, 40]])
        out = _blend(out, col, b * rng.uniform(0.75, 1.0))
        alpha = np.maximum(alpha, b)
    # vertical rain / oil streaks
    for _ in range(int(rng.integers(0, 3))):
        x = int(rng.uniform(0, w))
        streak = np.zeros((h, w), np.float32)
        streak[:, max(0, x - 1) : x + 2] = 1
        streak = ndi.gaussian_filter(streak, (h * 0.05, rng.uniform(0.8, 2.5)))
        out = _blend(out, (70, 60, 50), np.clip(streak * rng.uniform(0.2, 0.5), 0, 1))
    # fine light scratches
    if rng.random() < 0.4:
        sc = Image.new("L", (w, h), 0)
        sd = ImageDraw.Draw(sc)
        for _ in range(int(rng.integers(1, 6))):
            x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
            sd.line([x0, y0, x0 + rng.uniform(-0.3, 0.3) * w, y0 + rng.uniform(-0.5, 0.5) * h],
                    fill=int(rng.uniform(80, 200)), width=1)
        out = _blend(out, (235, 235, 235), np.asarray(sc, np.float32) / 255 * 0.5)
    return np.clip(out, 0, 255), alpha


def occlusion(img, alpha, rng, s):
    """Things in front of the plate: bumper edges, screws, grill bars, stickers, frames."""
    h, w = img.shape[:2]
    layer = Image.new("L", (w, h), 0)
    cols = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    dc = ImageDraw.Draw(cols)
    for _ in range(int(rng.integers(1, 2 + int(3 * s)))):
        kind = rng.choice(["vbar", "hbar", "screw", "sticker", "edge"])
        col = tuple(int(v) for v in rng.choice([[20, 20, 20], [200, 200, 205], [90, 90, 95], [180, 30, 30]]))
        if kind == "vbar":
            x = rng.uniform(0.05, 0.95) * w
            bw = rng.uniform(0.01, 0.04 + 0.03 * s) * w
            box = [x, -2, x + bw, h + 2]
        elif kind == "hbar":
            y = rng.uniform(0, 1) * h
            bh = rng.uniform(0.04, 0.12 + 0.1 * s) * h
            box = [-2, y, w + 2, y + bh]
        elif kind == "screw":
            cx, cy, r = rng.uniform(0.05, 0.95) * w, rng.uniform(0.15, 0.85) * h, rng.uniform(0.03, 0.09) * h
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            dc.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
            continue
        elif kind == "sticker":
            x0, y0 = rng.uniform(0, 0.9) * w, rng.uniform(0, 0.8) * h
            box = [x0, y0, x0 + rng.uniform(0.04, 0.16) * w, y0 + rng.uniform(0.15, 0.5) * h]
        else:  # edge: a bumper lip or frame cutting across a corner
            box = [-2, rng.uniform(0.8, 1.0) * h, w + 2, h + 2] if rng.random() < 0.5 else [-2, -2, w + 2, rng.uniform(0, 0.12) * h]
        d.rectangle(box, fill=255)
        dc.rectangle(box, fill=col)
    m = np.asarray(layer, np.float32) / 255
    if m.max() == 0:
        return img, alpha
    cm = np.asarray(cols, np.float32)
    m = ndi.gaussian_filter(m, rng.uniform(0.3, 1.2), truncate=2.5)
    out = img * (1 - m[..., None]) + cm * m[..., None]
    return np.clip(out, 0, 255), np.maximum(alpha, m)


# ----------------------------------------------------------------------- optics / sensor
def lighting(img, rng, s):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    # cast + gain + gamma + contrast
    out = out * rng.uniform(0.9, 1.1, 3).astype(np.float32) * np.float32(rng.uniform(0.75, 1.15 + 0.1 * s))
    g = float(np.exp(rng.uniform(-0.5, 0.5 + 0.3 * s)))
    out = 255 * (np.clip(out, 0, 255) / 255) ** g
    c = float(rng.uniform(0.7, 1.15))
    m_ = float(out.mean())
    out = (out - m_) * c + m_
    # shadow: soft gradient or hard-edged
    if rng.random() < 0.5:
        yy, xx = _grid(h, w)
        th = rng.uniform(0, 2 * np.pi)
        t = (xx / w - 0.5) * np.cos(th) + (yy / h - 0.5) * np.sin(th)
        edge = rng.uniform(0.02, 0.5)
        sh = 1 - rng.uniform(0.15, 0.6) * s * np.clip((t - rng.uniform(-0.3, 0.3)) / edge + 0.5, 0, 1)
        out = out * sh.astype(np.float32)[..., None]
    return np.clip(out, 0, 255).astype(np.float32, copy=False)


def overexpose(img, rng, s):
    """Retroreflective sheeting under flash: global wash-out, brightest at the centre."""
    h, w = img.shape[:2]
    yy, xx = _grid(h, w)
    r = np.exp(-(((xx / w - rng.uniform(0.3, 0.7)) / 0.5) ** 2 + ((yy / h - 0.5) / 0.9) ** 2))
    k = rng.uniform(0.25, 0.6 + 0.2 * s) * r
    return np.clip(img + (255 - img) * k[..., None], 0, 255)


def perspective(img, rng, s, fill=None):
    """Residual perspective / rotation left after rectification (or full view angle)."""
    h, w = img.shape[:2]
    j = (0.02 + 0.10 * s)
    dx = rng.uniform(-j, j, 4) * w
    dy = rng.uniform(-j, j, 4) * h * 0.8
    src = np.array([[dx[0], dy[0]], [w + dx[1], dy[1]], [w + dx[2], h + dy[2]], [dx[3], h + dy[3]]])
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    A = []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
    coeffs = np.linalg.solve(np.array(A, np.float64), src.reshape(-1).astype(np.float64))
    fill = tuple(int(v) for v in (img[0, 0] if fill is None else fill))
    im = Image.fromarray(img.astype(np.uint8)).transform(
        (w, h), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC, fillcolor=fill)
    return np.asarray(im, np.float32)


def blur(img, rng, s):
    kind = rng.choice(["gauss", "motion", "defocus"])
    if kind == "gauss":
        sg = rng.uniform(0.3, 0.8 + 1.4 * s)
        return ndi.gaussian_filter(img, (sg, sg * rng.uniform(0.8, 1.6), 0), truncate=2.5)
    k = int(2 * int(rng.uniform(2, 3 + 6 * s)) + 1)
    ker = np.zeros((k, k), np.float32)
    if kind == "motion":
        th = rng.uniform(-0.5, 0.5) if rng.random() < 0.7 else rng.uniform(0, np.pi)
        for t in np.linspace(-1, 1, 4 * k):
            x, y = int(round(k // 2 + t * (k // 2) * np.cos(th))), int(round(k // 2 + t * (k // 2) * np.sin(th)))
            ker[y, x] = 1
    else:
        yy, xx = np.mgrid[:k, :k] - k // 2
        ker[(yy ** 2 + xx ** 2) <= (k // 2) ** 2] = 1
    ker /= ker.sum()
    return ndi.convolve(img, ker[..., None], mode="nearest")


def noise(img, rng, s):
    sig = rng.uniform(1.5, 5 + 16 * s)
    n = rng.standard_normal(img.shape, dtype=np.float32)
    luma = rng.standard_normal(img.shape[:2] + (1,), dtype=np.float32)
    return np.clip(img + np.float32(sig) * (0.6 * n + 0.4 * luma) * (0.5 + img / 400), 0, 255)


def lowres(img, rng, s):
    h, w = img.shape[:2]
    f = rng.uniform(min(0.9, max(0.4, 1 - 0.6 * s)), 0.97)
    im = Image.fromarray(img.astype(np.uint8))
    small = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
    return np.asarray(small.resize((w, h), rng.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST])), np.float32)


def jpeg(img, rng, s):
    q = int(np.clip(rng.uniform(95 - 70 * s, 95), 18, 95))
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, "JPEG", quality=q)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if rng.random() < 0.25:  # a second, lower-quality generation (messaging apps)
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=int(np.clip(q - rng.uniform(5, 25), 15, 90)))
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(out, np.float32)


# --------------------------------------------------------------------------- orchestration
PROFILES = {
    # name: (probability, severity range)
    "clean": (0.18, (0.0, 0.1)),
    "mild": (0.34, (0.1, 0.35)),
    "moderate": (0.30, (0.35, 0.65)),
    "severe": (0.18, (0.65, 1.0)),
}


def sample_severity(rng) -> tuple[str, float]:
    names = list(PROFILES)
    p = np.array([PROFILES[n][0] for n in names])
    n = str(rng.choice(names, p=p / p.sum()))
    lo, hi = PROFILES[n][1]
    return n, float(rng.uniform(lo, hi))


def apply_damage(img: np.ndarray, rng, s: float, tags: list[str]):
    """Glare / dirt / occlusion. Returns (img, alpha) - alpha says how hidden each pixel is."""
    alpha = np.zeros(img.shape[:2], np.float32)
    if rng.random() < 0.15 + 0.6 * s:
        img, alpha = glare(img, alpha, rng, s); tags.append("glare")
    if rng.random() < 0.10 + 0.6 * s:
        img, alpha = dirt(img, alpha, rng, s); tags.append("dirt")
    if rng.random() < 0.08 + 0.45 * s:
        img, alpha = occlusion(img, alpha, rng, s); tags.append("occlusion")
    return img, alpha


def apply_optics(img: np.ndarray, rng, s: float, tags: list[str]) -> np.ndarray:
    if rng.random() < 0.35 + 0.5 * s:
        img = perspective(img, rng, s); tags.append("perspective")
    if rng.random() < 0.7:
        img = lighting(img, rng, s); tags.append("lighting")
    if rng.random() < 0.05 + 0.25 * s:
        img = overexpose(img, rng, s); tags.append("overexposed")
    if rng.random() < 0.15 + 0.7 * s:
        img = blur(img, rng, s); tags.append("blur")
    if rng.random() < 0.10 + 0.5 * s:
        img = lowres(img, rng, s); tags.append("lowres")
    if rng.random() < 0.5 + 0.4 * s:
        img = noise(img, rng, s); tags.append("noise")
    if rng.random() < 0.55:
        img = jpeg(img, rng, s); tags.append("jpeg")
    return img
