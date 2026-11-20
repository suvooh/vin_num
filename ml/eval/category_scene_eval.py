"""End-to-end vehicle-category accuracy through the REAL scene reader on the synthetic scene benchmark.

    python -I -X utf8 eval/category_scene_eval.py --n 60 --seed 3

Synthetic plates (rendered in the colours of ``synth.plates``) pasted on real Open Images car photos with the
benchmark's damage / optics profiles, found by the PP-OCR detector, cropped by the scene reader and classified by
``vinplate.category``. The expected category is the style the renderer drew. SYNTHETIC: the plate colours come from
the same table as the rules, the scene mix is private-heavy (62 % white) and the damage is simulated, so this only
shows how the colour step survives the real detect / crop path and degradation - not field accuracy.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scenes import make_scenes  # noqa: E402

STYLE_TO_KEY = {"private": "private", "commercial": "commercial", "ev_private": "ev_private", "ev_commercial": "ev_commercial",
                "rental": "rental", "diplomatic": "diplomatic"}


def run(n: int, seed: int) -> list[dict]:
    from vinplate.ocr.ppocr import PpOcrEngine
    from vinplate.scene import SceneReader

    reader = SceneReader(PpOcrEngine())
    rows = []
    for i, sc in enumerate(make_scenes(n, seed=seed, plate_frac=1.0)):
        res = reader.read(sc.img, domains=("plate",))
        fr = res.plate
        cat = fr.category if fr is not None else None
        rows.append(dict(index=i, profile=sc.profile, style=sc.meta.get("plate_style"), layout=sc.meta.get("plate_layout"),
                         found=fr is not None, status=fr.status if fr else None,
                         key=cat["key"] if cat else None, conf=cat["confidence"] if cat else None,
                         alts=[a["key"] for a in cat["alternatives"]] if cat else []))
        print(f"  scene {i + 1}", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    return rows


def report(rows: list[dict]) -> str:
    exp = lambda r: STYLE_TO_KEY.get(r["style"], r["style"])      # noqa: E731
    found = [r for r in rows if r["found"] and r["key"] is not None]
    ok = [r for r in found if r["key"] == exp(r)]
    unknown = [r for r in found if r["key"] == "unknown"]
    wrong = [r for r in found if r["key"] != exp(r) and r["key"] != "unknown"]
    conf_wrong = [r for r in wrong if r["conf"] >= 0.75]
    lines = [f"scenes {len(rows)}; plate found {len(found)}; category correct {len(ok)} ({len(ok) / max(len(found), 1):.1%} of found); "
             f"'unknown' {len(unknown)}; wrong {len(wrong)} (of which confidence >= 0.75: {len(conf_wrong)})",
             "", f"{'profile':<10}{'found':>7}{'correct':>9}{'unknown':>9}{'wrong':>7}"]
    for p in ("clean", "mild", "moderate", "severe"):
        f = [r for r in found if r["profile"] == p]
        lines.append(f"{p:<10}{len(f):>7}{sum(r['key'] == exp(r) for r in f):>9}{sum(r['key'] == 'unknown' for r in f):>9}"
                     f"{sum(r['key'] not in (exp(r), 'unknown') for r in f):>7}")
    styles = sorted({exp(r) for r in found} | {r["key"] for r in found})
    lines += ["", "confusion (rows = rendered style, columns = got):", f"{'':<15}" + "".join(f"{s[:9]:>10}" for s in styles)]
    conf = collections.defaultdict(collections.Counter)
    for r in found:
        conf[exp(r)][r["key"]] += 1
    for s in sorted(conf):
        lines.append(f"{s:<15}" + "".join(f"{conf[s].get(c, 0):>10}" for c in styles))
    lines += ["", "wrong answers: " + ", ".join(f"#{r['index']} {exp(r)}->{r['key']} ({r['conf']}, {r['profile']}, alts {r['alts']})" for r in wrong)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    print(report(run(a.n, a.seed)))


if __name__ == "__main__":
    main()
