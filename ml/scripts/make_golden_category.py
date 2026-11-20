"""Golden data for the Dart port of the vehicle-category classifier (`app/lib/reader/category.dart`).

Runs ``vinplate.category`` (the reference) on a fixed set of plate-line crops and writes inputs + expected
outputs to ``app/test/fixtures/golden/category_cases.json``. The Dart test replays the same crops (lossless PNG)
and layouts and compares EXACTLY (confidence is rounded to 3 decimals in both).

  cases        one per crop: the crop, the text layout the grammar gave (``parse_plate(text)["layout"]`` or null),
               the 2-means result (bg / ink centres as float64, distance) and the full ``classify_plate`` dict
  colours      ``name_colour`` for float RGB centres (random + boundary cases): [r, g, b] -> [name, confidence]

Content: every colour scheme (private, commercial, rental, EV private / commercial, temporary, trade, dealer text,
diplomatic, military, BH series) from the repo's own plate renderer under several lighting variants, plus
hand-made edge cases (one colour, two colours < 40 apart, exact 50/50 split, ink as the bigger cluster, 1x1 and tiny
crops, a large crop that needs a sampling stride, layout-vs-colour agreements and disagreements).

    cd ml
    .venv\\Scripts\\python.exe -I -X utf8 scripts/make_golden_category.py

Deterministic: every random draw is seeded.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import platform
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from synth.category_plates import LIGHTING, SCHEMES, degrade, render_scheme  # noqa: E402
from vinplate import category as cat  # noqa: E402
from vinplate.category import LABELS, classify_plate, name_colour, two_means  # noqa: E402
from vinplate.grammar import parse_plate  # noqa: E402

OUT = ML.parent / "app" / "test" / "fixtures" / "golden" / "category_cases.json"


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def strip_plate(bg, ink, h=48, w=240, bar=4, period=16) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    y0, y1 = int(0.2 * h), int(0.8 * h)
    for x in range(0, w, period):
        img[y0:y1, x:x + bar] = ink
    return img


def centres(crop: np.ndarray) -> dict:
    bg, ink, dist = two_means(crop)
    return {"bg": [float(v) for v in bg], "ink": None if ink is None else [float(v) for v in ink], "dist": float(dist)}


def make_case(cid: str, crop: np.ndarray, text: str | None, note: str) -> dict:
    parsed = parse_plate(text) if text else None
    layout = parsed["layout"] if parsed else None
    expected = classify_plate(crop, parsed)
    return {"id": cid, "note": note, "text": text, "layout": layout, "h": int(crop.shape[0]), "w": int(crop.shape[1]),
            "png": png_b64(crop), "kmeans": centres(crop), "expected": expected}


WHITE, BLACK, YELLOW, GREEN, BLUE, RED = (248, 248, 246), (18, 18, 18), (255, 206, 22), (22, 150, 72), (0, 72, 172), (196, 28, 36)


def scheme_cases(rng) -> list[dict]:
    out = []
    schemes = list(SCHEMES)
    degraded = [lt for lt in LIGHTING if lt != "clean"]
    for si, scheme in enumerate(schemes):
        for k in range(2):                                         # two clean plates per scheme
            p = render_scheme(rng, scheme)
            out.append(make_case(f"{scheme}_clean_{k}", p.crop, p.text, f"scheme {scheme}, clean"))
        for j in range(3):                                         # three lighting variants of one plate
            p = render_scheme(rng, scheme)
            light = degraded[(si * 3 + j) % len(degraded)]
            out.append(make_case(f"{scheme}_{light}", degrade(p.crop, light, rng), p.text, f"scheme {scheme}, {light}"))
    return out


def edge_cases(rng) -> list[dict]:
    out = []
    add = lambda cid, crop, text, note: out.append(make_case(cid, crop, text, note))   # noqa: E731
    flat = lambda c, h=30, w=90: np.full((h, w, 3), c, np.uint8)                         # noqa: E731
    std = "MH12AB1234"
    # --- colour pipeline edges
    add("flat_white", flat(250), None, "one colour: no ink, low confidence")
    add("flat_black", flat(10), std, "one dark colour: unknown")
    add("flat_yellow", flat(YELLOW), std, "one yellow colour")
    add("near_colours", strip_plate((200, 200, 200), (215, 205, 200)), std, "two colours 22 apart: below the 40 threshold")
    add("just_separated", strip_plate((200, 200, 200), (200, 200, 242)), std, "centres ~42 apart: ink found")
    left_right = np.empty((30, 100, 3), np.uint8)
    left_right[:, :50], left_right[:, 50:] = WHITE, BLACK
    add("half_half", left_right, std, "exact 50/50 split: a tie keeps the initial background cluster")
    heavy = strip_plate(WHITE, BLACK, bar=11, period=16)
    add("ink_majority", heavy, std, "ink covers 69 %: the bigger cluster is the background (a swap)")
    add("one_pixel", np.full((1, 1, 3), 120, np.uint8), None, "1x1 crop")
    add("tiny_2x3", np.array([[[250, 250, 250], [10, 10, 10], [250, 250, 250]], [[250, 250, 250], [250, 250, 250], [10, 10, 10]]], np.uint8),
        std, "2x3 crop")
    big = strip_plate(YELLOW, BLACK, h=480, w=1600, bar=24, period=96)
    add("large_stride", big, std, "480x1600: sampling stride 13")
    gradient = np.zeros((40, 200, 3), np.uint8)
    gradient[..., 0] = np.linspace(40, 250, 200)[None, :]
    gradient[..., 1] = np.linspace(30, 240, 200)[None, :]
    gradient[..., 2] = np.linspace(20, 230, 200)[None, :]
    add("gradient", gradient, std, "a ramp: not two colours")
    noisy = np.clip(strip_plate(GREEN, WHITE).astype(np.int16) + rng.integers(-25, 26, (48, 240, 3)), 0, 255).astype(np.uint8)
    add("noisy_ev", noisy, "HR26EV7788", "green/white with +-25 per-pixel noise")
    # --- colour names at / near boundaries
    for cid, bg, ink in [("cream_white", (255, 244, 214), (30, 30, 30)), ("pale_yellow", (255, 240, 150), (30, 30, 30)),
                         ("orange_yellow", (230, 100, 40), (10, 10, 10)), ("grey_black_ink", (136, 136, 135), (20, 20, 20)),
                         ("grey_yellow_ink", (110, 110, 112), (230, 190, 20)), ("purple", (140, 60, 160), (250, 250, 250)),
                         ("cyan_blue_band", (30, 190, 210), (250, 250, 250)), ("dark_red", (120, 20, 25), (240, 240, 240)),
                         ("magenta_red_edge", (220, 20, 70), (240, 240, 240)), ("dark_green", (10, 60, 30), (240, 240, 240))]:
        add(cid, strip_plate(bg, ink), std, f"colour naming: {cid}")
    # --- layout vs colour
    dip, mil, dealer, bh = "123CD45", "12A123456B", "DLR012345", "22BH1234AA"
    add("dip_on_blue", strip_plate(BLUE, WHITE), dip, "diplomatic text on blue: agree")
    add("dip_on_white", strip_plate(WHITE, BLACK), dip, "diplomatic text on white: disagree")
    add("plain_on_blue", strip_plate(BLUE, WHITE), std, "ordinary text on blue: unusual")
    add("mil_on_black", strip_plate((26, 26, 30), WHITE), mil, "military on dark: agree")
    add("mil_on_grey", strip_plate((100, 100, 104), WHITE), mil, "military on grey: agree")
    add("mil_on_white", strip_plate(WHITE, BLACK), mil, "military on white: disagree")
    add("dealer_on_red", strip_plate(RED, WHITE), dealer, "dealer text on red: agree")
    add("dealer_on_yellow", strip_plate(YELLOW, BLACK), dealer, "dealer text on yellow: disagree")
    add("bh_on_white", strip_plate(WHITE, BLACK), bh, "BH series on white: flag + agree")
    add("bh_on_yellow", strip_plate(YELLOW, BLACK), bh, "BH series on yellow: flag + colour disagreement")
    add("bh_on_black_white", strip_plate(BLACK, WHITE), bh, "BH series, unmatched colours: unknown with bh flag")
    add("black_white_plain", strip_plate(BLACK, WHITE), std, "dark plate with white ink, ordinary text: unknown, alternatives military/rental")
    add("green_other_ink", strip_plate(GREEN, (240, 120, 20)), std, "green with orange ink: orange is in the yellow hue band -> commercial EV")
    add("yellow_red_dim", strip_plate((200, 150, 20), (150, 20, 20)), std, "dim yellow + dark red: temporary, lower confidence")
    # --- REAL colours measured on the user's photo of a green-plate Toyota Camry (KA01MN4120): the teal EV plate
    ev = "KA01MN4120"
    add("real_ev_teal_clean", strip_plate((24, 181, 166), (246, 246, 246)), ev, "measured EV plate, clean photo (hue 175): electric, not diplomatic")
    add("real_ev_teal_dark", strip_plate((30, 162, 149), (246, 246, 246)), ev, "measured EV plate, darker part (hue 174)")
    add("real_ev_on_screen", strip_plate((132, 189, 193), (250, 250, 250)), ev, "the same plate photographed off a monitor: washed out, hue 184")
    add("real_ev_on_screen2", strip_plate((151, 199, 208), (240, 240, 240)), ev, "off a monitor, torch frame: hue 190, saturation 0.28")
    add("cool_white_plain", strip_plate((170, 200, 235), (20, 20, 20)), ev, "a white plate under cool light named blue + ordinary text: never diplomatic")
    add("blue_yellow_ink_plain", strip_plate(BLUE, (255, 214, 40)), ev, "blue + yellow text + ordinary text: unknown, EV commercial first")
    add("blue_no_text", strip_plate(BLUE, WHITE), None, "blue plate, text unreadable: diplomatic stays possible (low confidence)")
    return out


def colour_cases(rng) -> list[dict]:
    pts = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127.5, 127.5, 127.5),
           (71.4, 71.4, 71.4), (71.5, 71.5, 71.5), (158.1, 158.1, 158.1), (102, 102, 102), (101.9, 101.9, 101.9),
           (255, 244, 214), (255, 244, 213.9), (200, 200, 40), (40, 200, 200), (255, 0, 40), (255, 40, 0), (255, 0, 0.5),
           (220, 20, 70), (221, 20, 70), (230, 30, 200), (24, 54, 150), (0, 72, 172), (22, 150, 72), (255, 206, 22), (196, 28, 36),
           (255, 190, 100), (255, 190, 99), (200, 120, 40), (90, 90, 90), (250, 250, 100), (17, 71, 171),
           (24, 181, 166), (30, 162, 149), (132, 189, 193), (151, 199, 208), (0, 166, 140), (0, 150, 200), (0, 120, 220), (40, 195, 255)]
    pts += [tuple(float(v) for v in rng.uniform(0, 255, 3)) for _ in range(200)]
    pts += [tuple(float(v) for v in np.round(rng.uniform(0, 255, 3), 1)) for _ in range(60)]
    out = []
    for p in pts:
        n, c = name_colour(p)
        out.append({"rgb": [float(v) for v in p], "name": n, "conf": float(c)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rng = np.random.default_rng(2026)

    cases = scheme_cases(rng) + edge_cases(rng)
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids), "duplicate ids"
    # self-check: a decoded PNG reproduces the stored answer (so the file, not just the in-memory crop, is the reference)
    for c in cases:
        crop = np.asarray(Image.open(io.BytesIO(base64.b64decode(c["png"]))).convert("RGB"))
        parsed = parse_plate(c["text"]) if c["text"] else None
        assert classify_plate(crop, parsed) == c["expected"], c["id"]
    colours = colour_cases(np.random.default_rng(7))
    data = {
        "meta": {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "numpy": np.__version__,
                 "python": platform.python_version(), "machine": platform.machine(),
                 "note": "crops are lossless PNGs of plate-line crops; 'layout' is the grammar layout of the plate text "
                         "(null = no parsed layout); confidences are rounded to 3 decimals; kmeans values are float64 "
                         "(JSON round-trips them exactly); rgb ints round half up",
                 "labels": LABELS, "alt_below": cat.ALT_BELOW,
                 "reasons": {"bg": cat.R_BG, "ink": cat.R_INK, "one_colour": cat.R_ONE_COLOUR, "bh": cat.R_BH,
                             "bh_colour": cat.R_BH_COLOUR, "layout": cat.R_LAYOUT, "agree": cat.R_AGREE,
                             "disagree": cat.R_DISAGREE, "blue": cat.R_BLUE, "dip_format": cat.R_DIP_FORMAT, "no_rule": cat.R_NO_RULE},
                 "green_blue_hue": cat.GREEN_BLUE_HUE, "not_diplomatic_layouts": sorted(cat.NOT_DIPLOMATIC_LAYOUTS)},
        "cases": cases,
        "colours": colours,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    keys = {}
    for c in cases:
        keys[c["expected"]["key"]] = keys.get(c["expected"]["key"], 0) + 1
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB): {len(cases)} crops, {len(colours)} colour names; "
          f"categories: {dict(sorted(keys.items()))}")


if __name__ == "__main__":
    main()
