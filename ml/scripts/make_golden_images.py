"""Golden data for the Dart image-side port (app/lib/reader/{geometry,enhance,quality,rows,pp_engine}.dart).

Runs the Python REFERENCE implementation on fixed inputs and writes the inputs together with the
expected outputs to app/test/fixtures/golden/image_*.json, which the Dart tests compare against.

    cd ml
    .venv\\Scripts\\python -I -X utf8 scripts/make_golden_images.py

Deterministic (fixed seeds). Re-run it after changing the Python reference, then re-run the Dart tests.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import numpy as np
from PIL import Image

from synth.sampler import sample_plate_whole, sample_vin_whole
from vinplate.enhance import (best_channel, clip_mask, clip_stats, deglare, retinex, sharpen, stretch,
                              variants)
from vinplate.geometry.rows import looks_two_row, split_rows
from vinplate.ocr import ppocr
from vinplate.ocr.ppocr import PpOcrEngine, _homography_coeffs, _vocab_matrix, min_area_rect, warp_quad
from vinplate.quality import assess_crop
from vinplate.vocab import CHARS, UNK

OUT = ML.parent / "app" / "test" / "fixtures" / "golden"
CHARSET = ML.parent / "app" / "assets" / "models" / "ocr_charset.txt"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def img_obj(a: np.ndarray) -> dict:
    return {"h": int(a.shape[0]), "w": int(a.shape[1]), "rgb": b64(a.astype(np.uint8))}


def gray_of(a: np.ndarray) -> dict:  # views are replicated greys: keep one channel
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    return {"h": int(a.shape[0]), "w": int(a.shape[1]), "gray": b64(a[..., 0])}


def f(x) -> float:
    return float(x)


# ------------------------------------------------------------------------------------ geometry
def blob_points(cx, cy, w, h, deg) -> list[float]:
    """All integer pixels inside a rotated rectangle, as DB blobs are (x, y interleaved)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    r = int(math.hypot(w, h) / 2) + 2
    pts = []
    for y in range(int(cy) - r, int(cy) + r + 1):
        for x in range(int(cx) - r, int(cx) + r + 1):
            dx, dy = x - cx, y - cy
            u, v = dx * c + dy * s, -dx * s + dy * c
            if abs(u) <= w / 2 and abs(v) <= h / 2:
                pts += [float(x), float(y)]
    return pts


def geometry_cases() -> dict:
    rng = np.random.default_rng(5)
    rects = []
    for (w, h, deg) in [(120, 30, 0), (80, 20, 12), (200, 40, -8), (60, 60, 30), (150, 25, 85), (40, 12, 45), (300, 18, 3), (25, 90, 5)]:
        pts = blob_points(500 + rng.integers(0, 30), 300 + rng.integers(0, 30), w, h, deg)
        c, lo, sh = min_area_rect(np.array(pts).reshape(-1, 2))
        rects.append({"points": pts, "corners": c.ravel().tolist(), "long": f(lo), "short": f(sh)})
    # a triangle and a collinear run (Qhull fails on the latter; the Python keeps the points)
    tri = [10.0, 10, 90, 12, 40, 50, 41, 49, 60, 30]
    c, lo, sh = min_area_rect(np.array(tri).reshape(-1, 2))
    rects.append({"points": tri, "corners": c.ravel().tolist(), "long": f(lo), "short": f(sh)})
    col = [float(v) for i in range(30) for v in (i, 5)]
    c, lo, sh = min_area_rect(np.array(col).reshape(-1, 2))
    rects.append({"points": col, "corners": c.ravel().tolist(), "long": f(lo), "short": f(sh)})

    homs = []
    for src in ([10, 12, 112, 22, 105, 70, 3, 58], [0, 0, 100, 0, 100, 40, 0, 40], [30, 20, 130, 5, 140, 60, 25, 75]):
        dst = [0, 0, 100, 0, 100, 40, 0, 40]
        homs.append({"dst": dst, "src": src, "coeffs": _homography_coeffs(np.array(dst, float).reshape(4, 2), np.array(src, float).reshape(4, 2))})

    # smooth, textured source so a resampling error shows up
    H, W = 90, 140
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([128 + 90 * np.sin(xx / 9.0 + k) * np.cos(yy / 7.0 - k) for k in range(3)], axis=-1)
    src = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    quads = {
        "axis_aligned": ([20, 20, 120, 20, 120, 50, 20, 50], 0.0),
        "rotated": ([30, 25, 118, 12, 124, 38, 36, 51], 0.0),
        "perspective": ([22, 18, 126, 28, 118, 62, 30, 56], 0.0),
        "padded": ([40, 30, 100, 30, 100, 50, 40, 50], 0.25),
        "vertical_text": ([60, 10, 80, 12, 80, 80, 60, 78], 0.0),
        "clamped_at_border": ([-6, 10, 150, 10, 150, 40, -6, 40], 0.0),
    }
    warps = []
    for name, (q, pad) in quads.items():
        out = warp_quad(src, np.array(q, float).reshape(4, 2), pad=pad)
        warps.append({"name": name, "quad": q, "pad": pad, "out": img_obj(out)})
    return {"rects": rects, "homographies": homs, "warp_source": img_obj(src), "warps": warps}


def boxes_case() -> dict:
    rw, rh, w, h = 192, 128, 384, 256
    prob = np.random.default_rng(2).random((rh, rw)).astype(np.float32) * 0.02

    def bar(cx, cy, bw, bh, deg, core=0.92, fringe=0.45):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        for y in range(rh):
            for x in range(rw):
                u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
                if abs(u) <= bw / 2 and abs(v) <= bh / 2:
                    prob[y, x] = core
                elif abs(u) <= bw / 2 + 2 and abs(v) <= bh / 2 + 2:
                    prob[y, x] = max(prob[y, x], fringe)
    bar(60, 40, 70, 12, 0)          # a horizontal text line
    bar(120, 90, 80, 14, -9)        # a tilted one
    bar(30, 100, 20, 8, 0, core=0.38, fringe=0.31)   # low score: dropped
    prob[10:12, 150:152] = 0.9      # 4-pixel speck: dropped
    boxes = PpOcrEngine._boxes(prob, rw, rh, w, h)
    return {"rw": rw, "rh": rh, "w": w, "h": h, "prob": b64(prob),
            "boxes": [{"quad": b.quad.ravel().tolist(), "score": f(b.score)} for b in boxes]}


def preprocess_cases() -> dict:
    det = []
    for (h, w) in [(480, 640), (1200, 1600), (300, 400), (2000, 3000), (100, 1000), (736, 736), (50, 60)]:
        ratio = 1.0
        if min(h, w) < 736:
            ratio = 736 / min(h, w)
        if max(h, w) * ratio > 1920:
            ratio = 1920 / max(h, w)
        rh = max(32, int(round(int(h * ratio) / 32)) * 32)
        rw = max(32, int(round(int(w * ratio) / 32)) * 32)
        det.append({"h": h, "w": w, "rw": rw, "rh": rh})
    # one real det tensor on a small image, checked by statistics + sampled values
    rng = np.random.default_rng(9)
    im = (rng.random((40, 72, 3)) * 255).astype(np.uint8)
    h, w = im.shape[:2]
    ratio = 736 / min(h, w)
    rh = max(32, int(round(int(h * ratio) / 32)) * 32)
    rw = max(32, int(round(int(w * ratio) / 32)) * 32)
    r = np.asarray(Image.fromarray(im).resize((rw, rh), Image.BILINEAR), np.float32) / 255.0
    x = ((r[..., ::-1] - ppocr._DET_MEAN) / ppocr._DET_STD).transpose(2, 0, 1)
    samples = [[int(c), int(yy), int(xx), f(x[c, yy, xx])] for c, yy, xx in [(0, 0, 0), (1, 100, 200), (2, 300, 500), (0, rh - 1, rw - 1), (2, 5, 700)]]
    det_tensor = {"image": img_obj(im), "rw": rw, "rh": rh, "mean": [f(x[c].mean()) for c in range(3)], "samples": samples}

    rec = []
    for (h, w) in [(20, 60), (48, 200), (30, 900), (10, 5000), (100, 40)]:
        rec.append({"h": h, "w": w, "nw": min(1600, max(8, int(math.ceil(48 * w / max(h, 1)))))})
    crop = (rng.random((22, 61, 3)) * 255).astype(np.uint8)
    nw = min(1600, max(8, int(math.ceil(48 * 61 / 22))))
    resized = np.asarray(Image.fromarray(crop).resize((nw, 48), Image.BILINEAR))
    return {"det_sizes": det, "det_tensor": det_tensor, "rec_widths": rec, "rec_crop": {"image": img_obj(crop), "nw": nw, "resized": img_obj(resized)}}


def vocab_case() -> dict:
    chars = [""] + CHARSET.read_text(encoding="utf-8").splitlines() + [" "]
    eng = PpOcrEngine.__new__(PpOcrEngine)
    eng.chars = chars
    eng._vmat = _vocab_matrix(chars)
    unk = CHARS.index(UNK) + 1
    eng._unk = unk
    eng._keep = np.nonzero(eng._vmat[:, unk] < 1.0)[0]
    eng._vsmall = np.ascontiguousarray(eng._vmat[eng._keep])
    rng = np.random.default_rng(3)
    ix = {c: i for i, c in enumerate(chars)}
    peaks = [ix["M"], ix["H"], 0, ix["1"], ix["l"], ix["|"], ix["中"], ix["-"], ix["b"], ix["Q"], ix[" "], ix["0"]]
    T = 12
    probs = np.zeros((T, len(chars)), np.float64)
    for t in range(T):
        probs[t] = rng.dirichlet(np.full(len(chars), 0.02)) * 0.15
        probs[t, peaks[t]] += 0.85 - rng.random() * 0.3
        probs[t] /= probs[t].sum()
    probs = probs.astype(np.float32)
    return {"classes": len(chars), "t": T, "n_lines": len(chars) - 2, "n_kept": int(len(eng._keep)),
            "probs": b64(probs), "logp_t1": eng.to_logp(probs, 1.0).ravel().tolist(), "logp_t2": eng.to_logp(probs, 2.0).ravel().tolist(),
            "sample_classes": {c: ix[c] for c in ["M", "b", "l", "|", "中", "-", "Q", "0", " "]}}


# ------------------------------------------------------------------------------ enhance / quality / rows
def to_height(a: np.ndarray, h: int, max_w: int = 150) -> np.ndarray:
    im = Image.fromarray(a.astype(np.uint8))
    w = max(8, min(max_w, int(round(im.width * h / im.height))))
    return np.asarray(im.resize((w, h), Image.BILINEAR))


def hand_crops() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(11)
    out = []

    def strokes(base, ink, h=36, w=110):
        a = np.empty((h, w, 3), np.float32)
        a[:] = base
        for x in range(8, w - 8, 14):
            a[6:h - 6, x:x + 4] = ink
        return a

    a = strokes((150, 150, 150), (20, 20, 20))
    a[8:28, 40:70] = 255                                   # specular blob over two strokes
    out.append(("blob_on_grey_plate", a))
    a = strokes((255, 206, 22), (0, 0, 0))                # yellow commercial plate
    out.append(("yellow_plate", a))
    out.append(("flat", np.full((30, 90, 3), 130, np.float32)))
    out.append(("overexposed", np.full((30, 90, 3), 255, np.float32)))
    out.append(("veiled_low_contrast", strokes((170, 170, 170), (130, 130, 130))))
    a = strokes((225, 225, 225), (20, 20, 20), h=26, w=100)  # small + sharp
    out.append(("small_sharp", a))
    a = strokes((225, 225, 225), (20, 20, 20), h=44, w=100)
    from scipy import ndimage as ndi
    out.append(("blurred", ndi.gaussian_filter(a, (3.0, 3.0, 0))))
    a = strokes((60, 60, 70), (10, 10, 10))
    out.append(("dark", a + rng.normal(0, 3, a.shape)))
    return [(n, np.clip(v, 0, 255).astype(np.uint8)) for n, v in out]


def enhance_cases() -> dict:
    crops = hand_crops()
    rng = np.random.default_rng(21)
    for i in range(5):
        ln = sample_plate_whole(rng)
        crops.append((f"synth_plate_{i}_{ln.profile}_{'-'.join(ln.tags)[:30]}", to_height(ln.img, 40)))
    for i in range(2):
        ln = sample_vin_whole(rng)
        crops.append((f"synth_vin_{i}_{ln.profile}", to_height(ln.img, 30, 170)))
    cases = []
    for name, a in crops:
        dg = deglare(a)
        qa = assess_crop(a)
        cases.append({
            "name": name, "crop": img_obj(a),
            "stretch": gray_of(stretch(a)), "channel": gray_of(best_channel(a)),
            "sharpen": gray_of(sharpen(a)), "retinex": gray_of(retinex(a)),
            "deglare": None if dg is None else gray_of(dg),
            "clip_mask": b64(clip_mask(a).astype(np.uint8)), "clip_stats": clip_stats(a),
            "variants_default": [n for n, _ in variants(a, full=False)] + (["deglare"] if dg is not None else []),
            "quality": {"ok": qa.ok, "metrics": qa.metrics, "issues": qa.issues, "guidance": qa.guidance},
        })
    return {"cases": cases}


def rows_cases() -> dict:
    rng = np.random.default_rng(31)
    cases = []
    two = one = 0
    while two < 4 or one < 2:
        ln = sample_plate_whole(rng)
        rows = ln.meta.get("rows", 1)
        if rows == 2 and two < 4:
            two += 1
        elif rows == 1 and one < 2:
            one += 1
        else:
            continue
        a = to_height(ln.img, 64 if rows == 2 else 40, 200)
        pieces = split_rows(a)
        cases.append({"rows": rows, "crop": img_obj(a), "looks_two_row": bool(looks_two_row(*a.shape[:2])),
                      "top_h": int(pieces[0].shape[0]), "bottom_h": int(pieces[1].shape[0])})
    # a hard-edged synthetic two-row image: the cut must land in the gap
    a = np.full((60, 90, 3), 230, np.uint8)
    a[8:24, 5:85] = 20
    a[36:52, 5:85] = 20
    pieces = split_rows(a)
    cases.append({"rows": 2, "crop": img_obj(a), "looks_two_row": bool(looks_two_row(60, 90)),
                  "top_h": int(pieces[0].shape[0]), "bottom_h": int(pieces[1].shape[0])})
    aspect = [{"h": h, "w": w, "two_row": bool(looks_two_row(h, w))} for h, w in [(40, 100), (40, 104), (40, 105), (60, 90), (100, 100), (20, 84)]]
    return {"cases": cases, "aspect": aspect}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "image_geometry.json": {**geometry_cases(), "boxes": boxes_case(), "preprocess": preprocess_cases()},
        "image_vocab.json": vocab_case(),
        "image_enhance.json": enhance_cases(),
        "image_rows.json": rows_cases(),
    }
    for name, data in files.items():
        p = OUT / name
        p.write_text(json.dumps(data), encoding="utf-8")
        print(f"{name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
