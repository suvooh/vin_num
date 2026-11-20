"""Does the OCR text detector find the plate in REAL photos? (the 'locate by OCR' idea)

Python re-implementation of the PP-OCR (PaddleOCR v3) detector+recogniser used by the dealer
platform and the Flutter app, run on the real Open Images plate photos that have ground-truth
plate boxes. Not byte-identical to the Dart port (axis-aligned boxes, simplified resize), so treat
the numbers as indicative. These are NON-Indian plates: they say whether the approach can locate a
plate at all, not how well it does on Indian HSRP/VIN.

    python -I -X utf8 eval/ocr_locate_eval.py --images data/openimages/images --bbox data/openimages/val_bbox.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage as ndi

MODELS = Path(__file__).resolve().parents[2] / "app" / "assets" / "models"
PLATE_CLASS = "/m/01jfm_"


class PpOcr:
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.det = ort.InferenceSession(str(MODELS / "ocr_det.onnx"), so, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(MODELS / "ocr_rec.onnx"), so, providers=["CPUExecutionProvider"])
        self.chars = [""] + (MODELS / "ocr_charset.txt").read_text(encoding="utf-8").splitlines() + [" "]

    @staticmethod
    def _norm_bgr(a):  # RapidOCR feeds BGR
        a = a[..., ::-1].astype(np.float32) / 255.0
        return a

    def detect(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        ratio = 1.0
        if min(h, w) < 736:
            ratio = 736 / h if h < w else 736 / w
        rh, rw = max(32, round(int(h * ratio) / 32) * 32), max(32, round(int(w * ratio) / 32) * 32)
        im = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.BILINEAR))
        x = (self._norm_bgr(im) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        prob = self.det.run(None, {"x": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        lab, n = ndi.label(prob > 0.3)
        boxes = []
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            ys, xs = sl
            mask = lab[sl] == i
            if mask.sum() < 12 or min(ys.stop - ys.start, xs.stop - xs.start) < 3:
                continue
            if prob[sl][mask].mean() < 0.5:
                continue
            bw, bh = xs.stop - xs.start, ys.stop - ys.start
            d = bw * bh * 1.6 / (2 * (bw + bh))                       # DB unclip of a rectangle
            x0, x1 = (xs.start - d) / rw * w, (xs.stop + d) / rw * w
            y0, y1 = (ys.start - d) / rh * h, (ys.stop + d) / rh * h
            boxes.append((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
        return boxes

    def read(self, rgb: np.ndarray, box):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return "", 0.0
        nw = max(8, int(np.ceil(48 * crop.shape[1] / crop.shape[0])))
        im = np.asarray(Image.fromarray(crop).resize((min(nw, 960), 48), Image.BILINEAR))
        x = (self._norm_bgr(im) - 0.5) / 0.5
        p = self.rec.run(None, {"x": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        idx, conf, out, prev = p.argmax(-1), p.max(-1), [], 0
        cs = []
        for t, k in enumerate(idx):
            if k != 0 and k != prev:
                out.append(self.chars[k] if k < len(self.chars) else "")
                cs.append(conf[t])
            prev = k
        return "".join(out), float(np.mean(cs)) if cs else 0.0


def plate_like(t: str) -> bool:
    s = re.sub(r"[^A-Z0-9]", "", t.upper())
    return 4 <= len(s) <= 11 and any(c.isdigit() for c in s) and any(c.isalpha() for c in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--bbox", type=Path, required=True)
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    gt: dict[str, list[tuple[float, float, float, float]]] = {}
    with open(a.bbox, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["LabelName"] == PLATE_CLASS and r["Confidence"] == "1" and r["IsGroupOf"] == "0":
                gt.setdefault(r["ImageID"], []).append(tuple(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")))
    ocr = PpOcr()
    found_any = det_hit = sel_hit = n = 0
    sizes = []
    for p in sorted(a.images.glob("*.jpg")):
        if p.stem not in gt or n >= a.n:
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        h, w = rgb.shape[:2]
        gboxes = [(x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1 in gt[p.stem]]
        # keep the plates a user would actually be aiming at: at least ~40 px wide in this image
        gboxes = [g for g in gboxes if g[2] - g[0] >= 40]
        if not gboxes:
            continue
        n += 1
        boxes = ocr.detect(rgb)

        def inside(b, g, frac=0.6):
            ix = max(0, min(b[2], g[2]) - max(b[0], g[0]))
            iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
            return ix * iy >= frac * (b[2] - b[0]) * (b[3] - b[1])

        hit = [b for b in boxes if any(inside(b, g) for g in gboxes)]
        det_hit += bool(hit)
        reads = [(b, *ocr.read(rgb, b)) for b in boxes]
        cands = [(b, t, c) for b, t, c in reads if plate_like(t) and c >= 0.5]
        if cands:
            found_any += 1
            best = max(cands, key=lambda x: x[2])
            sel_hit += any(inside(best[0], g) for g in gboxes)
        sizes.append(max(g[2] - g[0] for g in gboxes))
    print(f"real photos with a labelled plate >= 40px wide: {n}  (median plate width {np.median(sizes):.0f}px)")
    print(f"  text detector puts a box inside the plate region : {det_hit}/{n} = {det_hit / n:.0%}")
    print(f"  a plate-LIKE reading (>=0.5 conf) exists anywhere : {found_any}/{n} = {found_any / n:.0%}")
    print(f"  ...and the best such reading lies inside the plate: {sel_hit}/{n} = {sel_hit / n:.0%}")


if __name__ == "__main__":
    sys.exit(main())
