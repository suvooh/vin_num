"""Golden data for the Dart scene reader (`app/lib/reader/scene_reader.dart`).

The Dart reader must make the SAME decisions as ``vinplate.scene.SceneReader`` when it is fed the SAME
recogniser output. So this script runs the real Python reader on synthetic scenes with a RECORDING engine
and stores, per run, in call order:

  * every ``detect`` result (rotated quads + scores),
  * every recogniser batch: the crop shapes (h, w) the reader asked about, and the vocab-projected
    log-probabilities it was given (float32, zlib + base64 - Dart reads the same float32 numbers),
  * the final ``as_dict`` of the plate / VIN readings, the candidate ranking and the detected lines.

The Dart test replays the detections / posteriors in the same order, over the same PIXELS (the scene
region is stored as a lossless PNG), so ``warp_quad`` / ``variants`` / ``assess_crop`` / ``split_rows`` /
``normalise_aspect`` run for real on both sides. If the Dart port asks for a different number of crops, or
crops of a different size (a different aspect stretch!), in a different order, the replay detects it and
fails loudly.

Each recogniser output also carries ``raw``: the RapidOCR-style greedy text WITH punctuation
(``PpOcrEngine.greedy``), which the reader needs for its screen-context cue. The expected dicts hold the
``unknown_candidates`` of every field (``FieldReading.as_dict``) and the scene's ``screen_context``.

Besides the random picks the file contains purpose-built cases (all synthetic, deterministic):
``condensed_*`` plates pasted at a width / height ratio ~2.6 (so the stretch of
``vinplate.scene.normalise_aspect`` is exercised and recorded in the crop shapes), ``screen_plate`` (the
same plate inside a drawn browser page: URL, tabs, 'Related searches' -> ``screen_context`` score 3.0) and
frames cases that fuse condensed photos / take the strongest screen context.

Scenes are cut to a region of interest around the plate / VIN (and downscaled to <= 480 px wide) before
the Python reader runs, so the stored PNGs stay small. The REAL PP-OCR engine produced the detections and
posteriors, so the data includes genuine mistakes (look-alikes, dropped characters, fragmented rows).

    cd ml
    .venv\\Scripts\\python.exe -I -X utf8 scripts/make_golden_scene.py            # ~10 minutes
    .venv\\Scripts\\python.exe -I -X utf8 scripts/make_golden_scene.py --quick    # 4 scenes, to test the plumbing
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import zlib
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from eval.scenes import make_scenes  # noqa: E402
from vinplate.ocr import PpOcrEngine  # noqa: E402
from vinplate.scene import SceneReader  # noqa: E402

OUT = ML.parent / "app" / "test" / "fixtures" / "golden" / "scene_golden.json"
MAX_W = 480


class _Tok:
    """What the recording engine hands back for a recogniser output; resolved again in ``to_logp``."""

    def __init__(self, i: int):
        self.i = i


class Recorder:
    """Wraps PpOcrEngine; looks like one to SceneReader and records every call in order."""

    def __init__(self, eng: PpOcrEngine):
        self.eng = eng
        self.events: list[dict] = []
        self.outs: list[np.ndarray] = []      # float32 [T, 38]
        self.raws: list[str] = []             # RapidOCR-style greedy text (with punctuation) of each output

    def detect(self, rgb):
        boxes = self.eng.detect(rgb)
        self.events.append({"k": "detect", "size": [int(rgb.shape[1]), int(rgb.shape[0])],
                            "boxes": [{"quad": [float(v) for v in b.quad.reshape(-1)], "score": float(b.score)} for b in boxes]})
        return boxes

    def rec_probs_many(self, crops):
        probs = self.eng.rec_probs_many(crops)
        toks, ids = [], []
        for p in probs:
            lp = np.asarray(self.eng.to_logp(p, 1.0), np.float64).astype(np.float32)   # the values both sides will use
            ids.append(len(self.outs))
            self.outs.append(lp)
            self.raws.append(self.eng.greedy(p)[0])
            toks.append(_Tok(ids[-1]))
        self.events.append({"k": "rec", "shapes": [[int(c.shape[0]), int(c.shape[1])] for c in crops], "ids": ids})
        return toks

    def rec_probs(self, crop):
        return self.rec_probs_many([crop])[0]

    def to_logp(self, tok, temperature=1.0):
        assert temperature == 1.0, "goldens are recorded at temperature 1.0"
        return self.outs[tok.i].astype(np.float64)

    def greedy(self, tok):
        """What ``SceneReader.observe`` asks for: the raw text of a line (the engine's own ``greedy``)."""
        return self.raws[tok.i], 1.0

    def pack_outs(self):
        return [{"t": int(o.shape[0]), "z": base64.b64encode(zlib.compress(o.tobytes(), 6)).decode(), "raw": self.raws[i]}
                for i, o in enumerate(self.outs)]


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _clean(x):
    return json.loads(json.dumps(x, default=_jsonable))


def region_of_interest(sc) -> np.ndarray:
    """The part of the scene around the plate / VIN, <= MAX_W wide: what a person aiming the phone sees."""
    H, W = sc.img.shape[:2]
    xs, ys = sc.quad[:, 0], sc.quad[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    mx, my = (0.6 * w, 0.3 * w) if sc.domain == "plate" else (0.1 * w, 3.0 * h)
    a, b, c, d = max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my))
    crop = sc.img[b:d, a:c]
    if crop.shape[1] > MAX_W:
        s = MAX_W / crop.shape[1]
        crop = np.asarray(Image.fromarray(crop).resize((MAX_W, max(8, round(crop.shape[0] * s))), Image.LANCZOS))
    return np.ascontiguousarray(crop)


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def field_dicts(res) -> dict:
    return {"plate": res.plate.as_dict() if res.plate else None, "vin": res.vin.as_dict() if res.vin else None}


def run_single(reader_factory, rgb, domains, allow_legacy) -> dict:
    rec = Recorder(reader_factory)
    reader = SceneReader(rec)
    t0 = time.perf_counter()
    res = reader.read(rgb, domains=tuple(domains), allow_legacy_vin=allow_legacy)
    dt = time.perf_counter() - t0
    exp = {**field_dicts(res), "candidates": res.candidates, "lines": res.lines, "screen_context": res.screen_context}
    return {"args": {"domains": list(domains), "allow_legacy": allow_legacy}, "events": rec.events, "outs": rec.pack_outs(),
            "expected": _clean(exp), "python_ms": round(dt * 1000)}


def run_frames(engine, rgbs, domains, allow_legacy) -> dict:
    rec = Recorder(engine)
    reader = SceneReader(rec)
    t0 = time.perf_counter()
    res = reader.read_frames(rgbs, domains=tuple(domains), allow_legacy_vin=allow_legacy)
    dt = time.perf_counter() - t0
    exp = {**field_dicts(res), "conflicts": res.conflicts, "screen_context": res.screen_context,
           "per_image": [{**field_dicts(s), "candidates": s.candidates, "screen_context": s.screen_context} for s in res.per_image]}
    return {"args": {"domains": list(domains), "allow_legacy": allow_legacy}, "events": rec.events, "outs": rec.pack_outs(),
            "expected": _clean(exp), "python_ms": round(dt * 1000)}


HR88B8888 = {"layout": "standard", "groups": ["HR", "88", "B", "8888"], "text": "HR88B8888", "kind": "standard"}


def condensed_scene(seed: int, squeeze: float, spec: dict | None = None):
    """A one-row plate scene whose plate is horizontally CONDENSED by ``squeeze`` (width / height ratio
    ~4.2 * squeeze: ``0.62`` gives the 2.6 : 1 of the real HR88B8888 photo that starved the recogniser).
    Returns the first scene (from ``seed`` on) whose degraded plate is clean / mild and, with ``spec``, still
    carries every character of the requested number."""
    from eval import scenes as ES

    bgs = ES.load_backgrounds()
    all_boxes = ES._read_plate_boxes(ES.OPENIMAGES)
    for k in range(400):
        rng = np.random.default_rng((seed, k))
        path, boxes = bgs[int(rng.integers(len(bgs)))]
        box = boxes[int(rng.integers(len(boxes)))]
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        others = [(x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1 in all_boxes.get(path.stem, [])
                  if abs((x0 * w + x1 * w) / 2 - (box[0] + box[2]) / 2) > 2 or abs((y0 * h + y1 * h) / 2 - (box[1] + box[3]) / 2) > 2]
        try:
            sc = ES.make_plate_scene(rng, rgb, box, others, spec=spec, squeeze=squeeze)
        except Exception:
            continue
        if sc.profile not in ("clean", "mild") or sc.meta["rows"] != 1 or (spec and sc.truth != spec["text"]):
            continue
        sc.meta["bg"] = path.name
        return sc
    raise RuntimeError(f"no clean condensed scene for seed {seed}")


_BROWSER_TABS = ["Car Number Plate - Bing", "E-Challan Status", "Search"]
_BROWSER_LINES = [
    "See more details", "Related searches", "Car Number Plate in Mumbai", "Car Number Plate in Gurgaon",
    "Renewal of RC for 15-year-old cars to", "4444 Number Plate Price and", "Procedure of renewing vehicle registration",
    "Registration Process in India", "How to Check Vehicle Details", "Fancy Number Auction Results",
]


def screen_photo(roi: np.ndarray) -> np.ndarray:
    """The plate scene ``roi`` shown inside a drawn browser page (tabs, address bar, headings, 'Related
    searches', result lines): what the camera sees when it is pointed at a monitor. A FIXTURE generator - drawn
    with the repo's OFL print fonts, no screenshot of a real page is stored."""
    from PIL import ImageDraw

    from synth.common import font_bank, load_font

    font = str(font_bank().by_role["print"][0])
    f_s, f_m = load_font(font, 17), load_font(font, 22)
    W, H = 640, 700
    page = Image.new("RGB", (W, H), (244, 244, 246))
    d = ImageDraw.Draw(page)
    d.rectangle([0, 0, W, 36], fill=(221, 224, 230))
    for i, t in enumerate(_BROWSER_TABS):
        d.text((14 + 205 * i, 9), t, font=f_s, fill=(38, 38, 38))
    d.rectangle([0, 36, W, 76], fill=(255, 255, 255))
    d.text((16, 47), "https://www.bing.com/images/search?view=detailV2&ccid=3SUMWcMG", font=f_s, fill=(60, 60, 60))
    d.text((16, 92), "HR88B8888 Becomes India's Most Expensive Car Registration Number", font=f_s, fill=(20, 20, 120))
    im = Image.fromarray(roi)
    s = 440 / im.width
    im = im.resize((440, max(8, round(im.height * s))), Image.LANCZOS)
    page.paste(im, ((W - im.width) // 2, 128))
    y = 128 + im.height + 20
    for i, t in enumerate(_BROWSER_LINES):
        d.text((24, y + 30 * i), t, font=f_m if i < 2 else f_s, fill=(30, 30, 30))
    return np.asarray(page, np.uint8)


def glare_blob(roi: np.ndarray, cx: float, cy: float, rx: float, ry: float, value: float = 252.0, soft: float = 0.35) -> np.ndarray:
    """A flash-glare hot-spot (soft-edged ellipse of near-white) painted over ``roi`` at (cx, cy): hides the
    character under it the way a specular reflection does, so the recogniser answers '?' there."""
    H, W = roi.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    a = np.clip((1.0 + soft - d) / soft, 0, 1)[..., None]
    return np.clip(roi * (1 - a) + value * a, 0, 255).astype(np.uint8)


def glared(engine, roi: np.ndarray, fx: float, rx: float) -> np.ndarray:
    """``glare_blob`` over the plate (the widest detected text box) at fraction ``fx`` of its width."""
    box = max((b for b in engine.detect(roi) if b.width > 80), key=lambda b: b.width)
    x0, y0, x1, y1 = box.bbox
    return glare_blob(roi, x0 + fx * (x1 - x0), (y0 + y1) / 2, rx * (x1 - x0), 0.75 * (y1 - y0))


def squeeze_x(rgb: np.ndarray, f: float) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(Image.fromarray(rgb).resize((max(8, round(rgb.shape[1] * f)), rgb.shape[0]), Image.LANCZOS)))


def pick(scenes, quick: bool):
    """A diverse set: plates (clean, glare, occlusion, dirt/blur, 2-row, with unreadable characters) and VINs
    (clean, glare/occlusion, severe, labelled)."""
    quotas = {
        "plate_clean": 3, "plate_glare": 2, "plate_occlusion": 2, "plate_damaged": 1, "plate_2row": 2, "plate_unknown": 3,
        "vin_clean": 2, "vin_glare_occ": 2, "vin_severe": 1, "vin_label": 1,
    }
    if quick:
        quotas = {"plate_clean": 1, "plate_2row": 1, "vin_clean": 1, "vin_label": 1}
    chosen, used = [], {k: 0 for k in quotas}

    def bucket(sc):
        t = set(sc.tags)
        if sc.domain == "plate":
            if "?" in sc.truth and sc.meta.get("rows") != 2:
                return "plate_unknown"            # synthetic damage hid a character: the reader must show '?' + candidates
            if sc.meta.get("rows") == 2:
                return "plate_2row"
            if "glare" in t and sc.profile != "clean":
                return "plate_glare"
            if "occlusion" in t:
                return "plate_occlusion"
            if sc.profile in ("moderate", "severe"):
                return "plate_damaged"
            return "plate_clean"
        if sc.meta.get("label_prefix"):
            return "vin_label"
        if sc.profile == "severe":
            return "vin_severe"
        if ({"glare", "occlusion"} & t) and sc.profile != "clean":
            return "vin_glare_occ"
        return "vin_clean"

    for sc in scenes:
        b = bucket(sc)
        if b in quotas and used[b] < quotas[b]:
            used[b] += 1
            chosen.append((b, sc))
        if all(used[k] >= quotas[k] for k in quotas):
            break
    return chosen, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--candidates", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    t0 = time.time()
    scenes = list(make_scenes(24 if a.quick else a.candidates, a.seed))
    chosen, used = pick(scenes, a.quick)
    print("picked", used, f"({len(chosen)} scenes, {time.time() - t0:.0f}s to generate)", flush=True)

    engine = PpOcrEngine()
    cases = []
    for n, (bucket, sc) in enumerate(chosen):
        rgb = region_of_interest(sc)
        runs = []
        if sc.domain == "plate":
            runs.append(run_single(engine, rgb, ["plate"], False))
        else:
            runs.append(run_single(engine, rgb, ["vin"], True))
        if n % 5 == 0:       # some scenes are also read with both domains (an 'auto' photo): exercises line ownership
            runs.append(run_single(engine, rgb, ["plate", "vin"], False))
        cases.append({"name": f"{n:02d}_{bucket}", "kind": "single", "domain": sc.domain, "truth": sc.truth, "profile": sc.profile,
                      "tags": sc.tags, "images": [{"w": int(rgb.shape[1]), "h": int(rgb.shape[0]), "png": png_b64(rgb)}], "runs": runs})
        r0 = runs[0]["expected"]
        got = (r0["plate"] or r0["vin"] or {}).get("masked")
        print(f"  [{n:02d} {bucket:16s}] truth={sc.truth!r:22} read={got!r} runs={len(runs)} python={runs[0]['python_ms']} ms", flush=True)

    # --- several photos of the same vehicle: fused, and a disagreement. Images already stored in a single
    # case are referenced ({"ref": case name}) instead of being stored twice.
    def ref(name):
        return {"ref": name}

    plates = [(n, b, sc) for n, (b, sc) in enumerate(chosen) if sc.domain == "plate"]
    if plates:
        n0, b, full = plates[0]
        rgb_a = region_of_interest(full)
        H, W = full.img.shape[:2]
        xs, ys = full.quad[:, 0], full.quad[:, 1]
        w = xs.max() - xs.min()
        a2, b2 = max(0, int(xs.min() - 1.0 * w)), max(0, int(ys.min() - 0.5 * w))        # a second photo from a bit further away
        c2, d2 = min(W, int(xs.max() + 1.0 * w)), min(H, int(ys.max() + 0.5 * w))
        crop = full.img[b2:d2, a2:c2]
        s2 = min(1.0, MAX_W / crop.shape[1])
        rgb_b = np.ascontiguousarray(np.asarray(Image.fromarray(crop).resize((max(8, round(crop.shape[1] * s2)), max(8, round(crop.shape[0] * s2))), Image.LANCZOS)))
        run = run_frames(engine, [rgb_a, rgb_b], ["plate"], False)
        cases.append({"name": "frames_same_plate", "kind": "frames", "domain": "plate", "truth": full.truth,
                      "images": [ref(f"{n0:02d}_{b}"), {"w": int(rgb_b.shape[1]), "h": int(rgb_b.shape[0]), "png": png_b64(rgb_b)}], "runs": [run]})
        print("  [frames_same_plate]", run["expected"]["plate"]["masked"] if run["expected"]["plate"] else None, "conflicts", len(run["expected"]["conflicts"]), flush=True)
    if len(plates) >= 2:
        rgbs = [region_of_interest(plates[0][2]), region_of_interest(plates[1][2])]
        run = run_frames(engine, rgbs, ["plate"], False)
        cases.append({"name": "frames_different_plates", "kind": "frames", "domain": "plate", "truth": None,
                      "images": [ref(f"{plates[0][0]:02d}_{plates[0][1]}"), ref(f"{plates[1][0]:02d}_{plates[1][1]}")], "runs": [run]})
        print("  [frames_different_plates] conflicts", len(run["expected"]["conflicts"]), flush=True)
    vins = [(n, b, sc) for n, (b, sc) in enumerate(chosen) if sc.domain == "vin"]
    if vins:
        n0, b, sc = vins[0]
        rgb_v = region_of_interest(sc)
        run = run_frames(engine, [rgb_v, rgb_v], ["vin"], True)       # the same photo twice: exercises the fused-group path
        cases.append({"name": "frames_vin", "kind": "frames", "domain": "vin", "truth": sc.truth,
                      "images": [ref(f"{n0:02d}_{b}"), ref(f"{n0:02d}_{b}")], "runs": [run]})
        print("  [frames_vin] ->", (run["expected"]["vin"] or {}).get("masked"), flush=True)

    # --- purpose-built cases for the aspect normalisation and the screen-context cue (not in --quick)
    if not a.quick:
        def single(name, domain, truth, rgb, domains_runs, extra=None):
            runs = [run_single(engine, rgb, d, False) for d in domains_runs]
            cases.append({"name": name, "kind": "single", "domain": domain, "truth": truth, "profile": "synthetic",
                          "tags": (extra or []) + ["synthetic_case"],
                          "images": [{"w": int(rgb.shape[1]), "h": int(rgb.shape[0]), "png": png_b64(rgb)}], "runs": runs})
            r0 = runs[0]["expected"]
            f = r0["plate"] or {}
            print(f"  [{name}] truth={truth!r} read={f.get('masked')!r} status={f.get('status')} cands={len(f.get('unknown_candidates') or [])} "
                  f"context={r0['screen_context']['score']}", flush=True)
            return runs

        cond = {}
        for tag, seed, squeeze in (("s42_a", 5, 0.42), ("s50", 1, 0.5), ("s42_b", 3, 0.42), ("s42_c", 8, 0.42)):
            sc = condensed_scene(seed, squeeze, HR88B8888)
            cond[tag] = (sc, region_of_interest(sc))
            single(f"condensed_{tag}", "plate", sc.truth, cond[tag][1], [["plate"]], ["condensed"])
        sc = condensed_scene(11, 0.45)
        single("condensed_random", "plate", sc.truth, region_of_interest(sc), [["plate"]], ["condensed"])

        # a glare hot-spot hides one character: the reader prints '?' and offers the legal, pattern-ranked candidates
        # (HR88?8888: the 5th character of a 9-character plate is a LETTER; a digit slot with a dominant candidate -> 'suggested')
        glare = {}
        for tag, src, fx, rx in (("glare_s42_a", "s42_a", 0.56, 0.045), ("glare_s42_c", "s42_c", 0.59, 0.07), ("glare_s50", "s50", 0.28, 0.045)):
            glare[tag] = glared(engine, cond[src][1], fx, rx)
            single(tag, "plate", cond[src][0].truth, glare[tag], [["plate"]], ["condensed", "glare"])

        page = screen_photo(cond["s42_a"][1])
        single("screen_plate", "plate", cond["s42_a"][0].truth, page, [["plate"], ["plate", "vin"]], ["screen"])

        def frames(name, truth, imgs):
            run = run_frames(engine, [i for _, i in imgs], ["plate"], False)
            cases.append({"name": name, "kind": "frames", "domain": "plate", "truth": truth,
                          "images": [({"ref": r} if r else {"w": int(i.shape[1]), "h": int(i.shape[0]), "png": png_b64(i)}) for r, i in imgs],
                          "runs": [run]})
            e = run["expected"]
            print(f"  [{name}]", (e["plate"] or {}).get("masked"), "context", e["screen_context"]["score"],
                  [s["screen_context"]["score"] for s in e["per_image"]], flush=True)

        frames("frames_condensed_plate", cond["s42_a"][0].truth,
               [("condensed_s42_a", cond["s42_a"][1]), ("condensed_s42_c", cond["s42_c"][1])])
        frames("frames_screen_plate", cond["s42_a"][0].truth, [("screen_plate", page), ("condensed_s42_c", cond["s42_c"][1])])
        frames("frames_glare_plate", cond["s42_a"][0].truth, [("glare_s42_a", glare["glare_s42_a"]), ("condensed_s42_c", cond["s42_c"][1])])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"seed": a.seed, "max_w": MAX_W, "cases": cases}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
