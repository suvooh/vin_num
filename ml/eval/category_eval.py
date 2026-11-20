"""Accuracy of ``vinplate.category.classify_plate`` on synthetic plates of every colour scheme.

    python -m eval.category_eval [--n 40] [--seed 1]

The plates are drawn by ``synth.category_plates`` (the repo's own renderer, scheme colours from the category
rule table) and crops are degraded with named lighting variants.  SYNTHETIC numbers: the colours of the test
plates come from the same table the rules come from, so 'clean' accuracy mostly proves the colour pipeline
(sampling, 2-means, naming, rules) works, NOT that real plates look like this.  Degraded accuracy is the
informative number; real photographs of real plates are the missing evidence.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth.category_plates import LIGHTING, SCHEMES, degrade, render_scheme  # noqa: E402
from vinplate.category import classify_plate  # noqa: E402
from vinplate.grammar import parse_plate  # noqa: E402

CONFIDENT = 0.75          # the category's own "no alternatives needed" threshold


def evaluate(n_per: int = 40, seed: int = 1, lightings: list[str] | None = None) -> dict:
    """Render ``n_per`` plates per scheme (one plate, every lighting variant applied to its crop) and classify them."""
    rng = np.random.default_rng(seed)
    lightings = lightings or LIGHTING
    rows = []
    for scheme in SCHEMES:
        for _ in range(n_per):
            p = render_scheme(rng, scheme)
            parsed = parse_plate(p.text)
            for light in lightings:
                crop = degrade(p.crop, light, rng)
                res = classify_plate(crop, parsed)
                rows.append(dict(scheme=scheme, expected=p.expected, light=light, got=res["key"],
                                 conf=res["confidence"], alts=[a["key"] for a in res["alternatives"]],
                                 bh=res["bh"], text=p.text))
    return {"rows": rows, "n_per": n_per, "seed": seed}


def accuracy(rows: list[dict], **where) -> tuple[int, int]:
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return sum(r["got"] == r["expected"] for r in sel), len(sel)


def summarise(rows: list[dict]) -> dict:
    """Accuracy per lighting, per scheme on clean, wrong-but-confident, wrong-but-alternative-listed counts."""
    out: dict = {"per_light": {}, "clean_per_scheme": {}, "confusion": {}}
    for light in sorted({r["light"] for r in rows}, key=LIGHTING.index):
        sel = [r for r in rows if r["light"] == light]
        wrong = [r for r in sel if r["got"] != r["expected"]]
        out["per_light"][light] = dict(
            n=len(sel), correct=len(sel) - len(wrong),
            unknown=sum(r["got"] == "unknown" for r in wrong),
            confident_wrong=sum(r["conf"] >= CONFIDENT and r["got"] != "unknown" for r in wrong),
            right_in_alternatives=sum(r["expected"] in r["alts"] for r in wrong))
    for scheme in SCHEMES:
        out["clean_per_scheme"][scheme] = accuracy(rows, scheme=scheme, light="clean")
    conf: dict = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["light"] != "clean":
            conf[r["expected"]][r["got"]] += 1
    out["confusion"] = {k: dict(v) for k, v in conf.items()}
    return out


def render_report(res: dict) -> str:
    rows = res["rows"]
    s = summarise(rows)
    lines = [f"vehicle-category accuracy, synthetic plates, {res['n_per']} plates per scheme x {len(SCHEMES)} schemes, "
             f"seed {res['seed']}", "",
             f"{'lighting':<12}{'n':>6}{'correct':>9}{'acc':>8}{'unknown':>9}{'conf.wrong':>12}{'right in alts':>15}"]
    for light, d in s["per_light"].items():
        lines.append(f"{light:<12}{d['n']:>6}{d['correct']:>9}{d['correct'] / d['n']:>8.1%}{d['unknown']:>9}"
                     f"{d['confident_wrong']:>12}{d['right_in_alternatives']:>15}")
    deg = [r for r in rows if r["light"] != "clean"]
    if deg:
        ok = sum(r["got"] == r["expected"] for r in deg)
        lines.append(f"{'ALL degraded':<12}{len(deg):>6}{ok:>9}{ok / len(deg):>8.1%}")
    lines += ["", "clean accuracy per scheme:"]
    for k, (a, b) in s["clean_per_scheme"].items():
        lines.append(f"  {k:<14}{a:>4}/{b:<4}{a / max(b, 1):>8.1%}")
    keys = list(dict.fromkeys(r["expected"] for r in rows))
    got_keys = sorted({g for v in s["confusion"].values() for g in v}, key=lambda g: (g not in keys, g))
    lines += ["", "confusion over all DEGRADED variants (rows = expected, columns = got):",
              f"{'':<15}" + "".join(f"{g[:9]:>10}" for g in got_keys)]
    for k in keys:
        v = s["confusion"].get(k, {})
        lines.append(f"{k:<15}" + "".join(f"{v.get(g, 0):>10}" for g in got_keys))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="plates per scheme")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    print(render_report(evaluate(a.n, a.seed)))


if __name__ == "__main__":
    main()
