"""End-to-end reader benchmark on synthetic full-frame scenes (``eval.scenes``).

    cd ml
    python -I -X utf8 eval/e2e_eval.py --n 120 --seed 3 --readers old,new --json out.json
    python -I -X utf8 eval/e2e_eval.py --n 20 --readers fake          # plumbing check, no models

A *reader* is any ``Callable[[np.ndarray, str], ReadOut]`` taking the full RGB frame and the
domain ("plate" | "vin") and returning ``{text, masked, status, box}``. Built-in readers
(constructed lazily, so this module imports without the OCR stack):

  old   the pipeline being replaced: PP-OCR detect + greedy recognise (``PpOcrEngine.read_lines``,
        RapidOCR-compatible lines) then ``eval/legacy_extract.py`` regex extraction (the old backend).
  new   ``vinplate.scene.SceneReader`` (detect -> grammar-constrained decode -> select).
  fake  an oracle that returns the truth (tests the harness itself).

How readings are scored - see ``score_reading``. The headline numbers, in order of importance:

  accepted_wrong  of the scenes whose truth is fully visible, the share where the reader said
                  ``accept`` but the text is wrong. The dangerous error; drive it to ~0.
  faithful        share of scenes where a reading was returned (accept/review) and every
                  character it SHOWS is correct - a hidden character printed as ``?`` counts as
                  faithful, a hallucinated one does not.
  accept / review / none   the coverage trade-off: how much is automated, flagged, or missed.

These scenes are synthetic (a real car photo + a synthetic Indian plate / VIN). They answer
"did the new pipeline improve on the old one, and on which damage slice", not "what is the
field accuracy" - there is no real Indian-plate / VIN test set in this repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, TypedDict

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
if str(ML_ROOT) not in sys.path:          # `python -I eval/e2e_eval.py` does not add it
    sys.path.insert(0, str(ML_ROOT))

FOUND = ("accept", "review")


class ReadOut(TypedDict):
    text: str                                       # the decoder's best string ('' if nothing)
    masked: str                                     # same, with '?' where evidence was weak / missing
    status: str                                     # accept | review | reject | none
    box: tuple[float, float, float, float] | None   # x0, y0, x1, y1 of the reading in the frame, if known


Reader = Callable[[np.ndarray, str], ReadOut]


def none_readout() -> ReadOut:
    return ReadOut(text="", masked="", status="none", box=None)


# ------------------------------------------------------------------------------ scoring
def score_reading(domain: str, truth: str, text: str, masked: str, status: str) -> dict:
    """Score one reading against ground truth. ``truth`` may contain ``?`` = a character the
    synthetic damage hid (unrecoverable). ``domain`` is carried for symmetry (no per-domain rule).

    found                status in ('accept', 'review') - a reading was offered.
    has_hidden           '?' in truth.
    exact                found and text == truth and not has_hidden.
    asserted_wrong_chars among the characters ``masked`` SHOWS (non-'?'), the number that differ
                         from truth at that position. A letter where truth is '?' counts as wrong
                         (a hallucinated hidden character). If len(masked) != len(truth) the
                         alignment is unknown: ``length_mismatch`` is set and EVERY shown
                         character counts as wrong. 0 / False when nothing was found (a reject
                         or 'none' asserts nothing).
    faithful             found, no asserted-wrong characters and no length mismatch: every
                         character is either read correctly or honestly printed as '?'.
    accepted_wrong       status == 'accept' and not has_hidden and text != truth. THE safety metric.
    accepted_with_hidden status == 'accept' and has_hidden. Reported separately and NOT counted
                         wrong: the truth is partly unknowable, so 'accept' is merely suspicious.
    shown_chars / shown_correct   character-level precision of what was shown.
    masked_len_delta     len(masked) - len(truth) (a nonzero value usually means the reader
                         inserted a '?' for a "possibly dropped" character that was not dropped).
    """
    shown_src = masked or text
    found = status in FOUND
    has_hidden = "?" in truth
    wrong = 0
    shown = 0
    correct = 0
    length_mismatch = False
    if found:
        if len(shown_src) == len(truth):
            for a, b in zip(shown_src, truth):
                if a != "?":
                    shown += 1
                    if a == b:
                        correct += 1
                    else:
                        wrong += 1
        else:
            length_mismatch = True
            shown = sum(1 for a in shown_src if a != "?")
            wrong = shown
    return {
        "found": found,
        "has_hidden": has_hidden,
        "exact": bool(found and text == truth and not has_hidden),
        "asserted_wrong_chars": wrong,
        "length_mismatch": length_mismatch,
        "faithful": bool(found and wrong == 0 and not length_mismatch),
        "accepted_wrong": bool(status == "accept" and not has_hidden and text != truth),
        "accepted_with_hidden": bool(status == "accept" and has_hidden),
        "shown_chars": shown,
        "shown_correct": correct,
        "masked_len_delta": len(shown_src) - len(truth) if found else 0,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def located(box, truth_box) -> bool | None:
    """Did the reader point at the right place? IoU >= 0.25 with the scene's plate / VIN box, or
    the reported box lies (>= 80%) inside it. None when the reader gave no box."""
    if box is None:
        return None
    ix = max(0.0, min(box[2], truth_box[2]) - max(box[0], truth_box[0]))
    iy = max(0.0, min(box[3], truth_box[3]) - max(box[1], truth_box[1]))
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return bool(box_iou(box, truth_box) >= 0.25 or ix * iy / area >= 0.8)


# ---------------------------------------------------------------------------- aggregation
def _slice_stats(rs: list[dict]) -> dict:
    sc = [score_reading(r["domain"], r["truth"], r["text"], r["masked"], r["status"]) for r in rs]
    n = len(rs)
    n_read = sum(not s["has_hidden"] for s in sc)
    acc = sum(r["status"] == "accept" for r in rs)
    rev = sum(r["status"] == "review" for r in rs)
    exact = sum(s["exact"] for s in sc)
    aw = sum(s["accepted_wrong"] for s in sc)
    acc_read = sum(r["status"] == "accept" and not s["has_hidden"] for r, s in zip(rs, sc))
    found_n = sum(s["found"] for s in sc)
    wrong_chars = sum(s["asserted_wrong_chars"] for s in sc)
    loc = [r["located"] for r in rs if r.get("located") is not None]
    ms = [r["ms"] for r in rs if r.get("ms") is not None]
    lo, hi = wilson(aw, n_read)
    return {
        "n": n,
        "n_readable": n_read,                       # scenes whose truth has no hidden character
        "accept": acc / n if n else 0.0,
        "review": rev / n if n else 0.0,
        "none": (n - acc - rev) / n if n else 0.0,  # reject + nothing found
        "exact": exact / n_read if n_read else None,
        "faithful": sum(s["faithful"] for s in sc) / n if n else 0.0,
        "accepted_wrong": aw / n_read if n_read else None,
        "accepted_wrong_n": aw,
        "accepted_wrong_ci95": [round(lo, 4), round(hi, 4)],
        "accept_precision": (1 - aw / acc_read) if acc_read else None,
        "accepted_with_hidden": sum(s["accepted_with_hidden"] for s in sc),
        "wrong_chars_per_scene": wrong_chars / n if n else 0.0,
        "wrong_chars_per_found": wrong_chars / found_n if found_n else None,
        "length_mismatch": sum(s["length_mismatch"] for s in sc),
        "located": (sum(loc) / len(loc)) if loc else None,
        "ms_median": float(np.median(ms)) if ms else None,
    }


def summarise(rows: Iterable[dict]) -> dict:
    """``rows``: dicts with reader, domain, truth, profile, tags, text, masked, status (and
    optionally ms, located). Returns ``{reader: {slice_name: stats}}`` with slices ``all``,
    ``domain=plate|vin``, ``profile=...``, ``tag=...`` (a scene counts in every tag it carries).
    Denominators: exact / accepted_wrong / their CI use the scenes with no hidden character
    (``n_readable``); every other rate uses ``n``. Scores are recomputed from the raw fields."""
    by_reader: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_reader[r["reader"]].append(r)
    out: dict[str, dict] = {}
    for reader, rs in by_reader.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            groups["all"].append(r)
            groups[f"domain={r['domain']}"].append(r)
            groups[f"profile={r['profile']}"].append(r)
            for t in sorted(set(r.get("tags") or [])):
                groups[f"tag={t}"].append(r)
        out[reader] = {k: _slice_stats(v) for k, v in groups.items()}
    return out


_ORDER_HEAD = ["all", "domain=plate", "domain=vin", "profile=clean", "profile=mild", "profile=moderate", "profile=severe"]


def _pct(x, w=5):
    return f"{'-':>{w}}" if x is None else f"{100 * x:{w}.1f}"


def format_table(summary: dict) -> str:
    readers = list(summary)
    slices: list[str] = []
    for r in readers:
        for k in summary[r]:
            if k not in slices:
                slices.append(k)
    slices.sort(key=lambda k: (_ORDER_HEAD.index(k) if k in _ORDER_HEAD else len(_ORDER_HEAD), k))
    head = ("| slice | reader | n | accept% | review% | none% | exact% | faithful% | acc-wrong% [95% CI] "
            "| wrong chars/scene | located% | ms |\n|---|---|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|")
    lines = [head]
    for k in slices:
        for rd in readers:
            s = summary[rd].get(k)
            if not s:
                continue
            ci = s["accepted_wrong_ci95"]
            aw = "-" if s["accepted_wrong"] is None else f"{100 * s['accepted_wrong']:.1f} [{100 * ci[0]:.1f}-{100 * ci[1]:.1f}]"
            ms = "-" if s["ms_median"] is None else f"{s['ms_median']:.0f}"
            lines.append(f"| {k} | {rd} | {s['n']} | {_pct(s['accept'])} | {_pct(s['review'])} | {_pct(s['none'])} "
                         f"| {_pct(s['exact'])} | {_pct(s['faithful'])} | {aw} | {s['wrong_chars_per_scene']:.2f} "
                         f"| {_pct(s['located'])} | {ms} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- run
def run(readers: dict[str, Reader], scenes: Iterable, *, progress: bool = False,
        on_row: Callable[[dict, object], None] | None = None) -> list[dict]:
    """Run every reader on every scene (scenes are iterated once). Returns one row per
    (reader, scene): the raw ReadOut, wall time ``ms``, ``located`` and the score fields. A reader
    that raises is recorded as status 'none' with ``error`` set (and counts as a miss)."""
    rows: list[dict] = []
    for i, sc in enumerate(scenes):
        for name, reader in readers.items():
            t0 = time.perf_counter()
            err = None
            try:
                if getattr(reader, "wants_scene", False):
                    out = reader(sc.img, sc.domain, scene=sc)
                else:
                    out = reader(sc.img, sc.domain)
            except Exception as exc:        # a crash on one scene must not abort the benchmark
                out, err = none_readout(), f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t0) * 1000.0
            out = {**none_readout(), **(out or {})}
            row = {
                "reader": name, "index": i, "domain": sc.domain, "truth": sc.truth, "profile": sc.profile,
                "tags": list(sc.tags), "text": out["text"], "masked": out["masked"], "status": out["status"],
                "box": None if out["box"] is None else [round(float(v), 1) for v in out["box"]],
                "ms": round(ms, 1), "located": located(out["box"], sc.bbox), "error": err,
            }
            row.update(score_reading(sc.domain, sc.truth, out["text"], out["masked"], out["status"]))
            rows.append(row)
            if on_row:
                on_row(row, sc)
        if progress:
            print(f"\r  scene {i + 1}", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return rows


# -------------------------------------------------------------------------- reader factories
def _make_fake() -> Reader:
    """Oracle: returns the truth (status 'review' when the truth has a hidden character)."""
    def fake(rgb, domain, scene=None) -> ReadOut:
        t = scene.truth
        return ReadOut(text=t, masked=t, status="review" if "?" in t else "accept", box=tuple(scene.bbox))
    fake.wants_scene = True            # type: ignore[attr-defined]
    return fake


def _load_backend_extract():
    from eval import legacy_extract            # the regex extractor the backend used to ship

    return legacy_extract


def _make_old() -> Reader:
    """PP-OCR detect + greedy recognise -> backend regex extractor (what ``backend/`` did)."""
    from vinplate.ocr.ppocr import PpOcrEngine
    extract_fields = _load_backend_extract().extract_fields
    engine = PpOcrEngine()

    def old(rgb, domain) -> ReadOut:
        found = extract_fields(engine.read_lines(rgb))["plate" if domain == "plate" else "vin"]
        if not found:
            return none_readout()
        return ReadOut(text=found["value"], masked=found["value"], status="accept", box=None)
    return old


_VIN_LABELS = ("CHASSISNO", "CHASISSNO", "CHASISNO", "CHASSIS", "VINNO", "VIN")


def _strip_vin_label(text: str, masked: str) -> tuple[str, str]:
    """The grammar's optional label ("VIN", "CHASSIS NO") is part of its reading but not of the
    VIN; scene truth excludes it, so strip it from both strings (they share leading characters)."""
    for lab in _VIN_LABELS:
        if text.startswith(lab) and len(text) - len(lab) >= 6:
            return text[len(lab):], masked[len(lab):] if len(masked) >= len(lab) else masked
    return text, masked


def _make_new() -> Reader:
    """``vinplate.scene.SceneReader`` - interface: ``.read(rgb) -> SceneResult`` with ``.plate`` /
    ``.vin`` each None or a FieldReading carrying ``.reading`` (text, masked, status) and ``.box``."""
    from vinplate.ocr.ppocr import PpOcrEngine
    from vinplate.scene import SceneReader
    import os
    # ablation switches: VINPLATE_VIEWS=stretch,sharpen (or "none"), VINPLATE_TEMP=2.0, VINPLATE_MULTIVIEW=0
    views = os.environ.get("VINPLATE_VIEWS")
    reader = SceneReader(PpOcrEngine(), temperature=float(os.environ.get("VINPLATE_TEMP", "1.0")),
                         multi_view=os.environ.get("VINPLATE_MULTIVIEW", "1") != "0",
                         views=None if views is None else tuple(v for v in views.split(",") if v and v != "none"))

    def new(rgb, domain) -> ReadOut:
        # the same call the API makes when the photo's role is known ('plate' / 'vin' photos); a VIN
        # photo allows short legacy chassis numbers
        res = reader.read(rgb, domains=(domain,), allow_legacy_vin=(domain == "vin"))
        fr = res.plate if domain == "plate" else res.vin
        if fr is None:
            return none_readout()
        rd = fr.reading
        text, masked = rd.text, rd.masked
        if domain == "vin":
            text, masked = _strip_vin_label(text, masked)
        status = getattr(fr, "status", None) or rd.status
        box = getattr(fr, "box", None)
        return ReadOut(text=text, masked=masked, status=status,
                       box=None if box is None else tuple(float(v) for v in box))
    return new


READER_FACTORIES: dict[str, Callable[[], Reader]] = {"fake": _make_fake, "old": _make_old, "new": _make_new}


def build_readers(names: Iterable[str]) -> dict[str, Reader]:
    readers: dict[str, Reader] = {}
    for n in names:
        if n not in READER_FACTORIES:
            print(f"unknown reader '{n}' (choose from {', '.join(READER_FACTORIES)}); skipped", file=sys.stderr)
            continue
        try:
            readers[n] = READER_FACTORIES[n]()
        except Exception as exc:       # lazy import of the parent's modules / model files
            print(f"reader '{n}' unavailable, skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    return readers


# ----------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    from eval.scenes import make_scenes

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=120, help="number of scenes")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--plate-frac", type=float, default=0.6, help="share of plate scenes (rest are VIN)")
    ap.add_argument("--readers", default="old,new", help="comma list from: " + ",".join(READER_FACTORIES))
    ap.add_argument("--json", type=Path, help="write {args, summary, rows} here")
    ap.add_argument("--dump-dir", type=Path, help="save a PNG of every accepted-wrong scene here")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    readers = build_readers([x.strip() for x in a.readers.split(",") if x.strip()])
    if not readers:
        print("no usable reader; nothing to run", file=sys.stderr)
        return 2
    if a.dump_dir:
        a.dump_dir.mkdir(parents=True, exist_ok=True)

    def dump(row, sc):
        if row["accepted_wrong"] and a.dump_dir:
            from PIL import Image
            Image.fromarray(sc.img).save(a.dump_dir / f"{row['reader']}_{row['index']:04d}_{sc.truth}_got_{row['text']}.png")

    scenes = make_scenes(a.n, seed=a.seed, plate_frac=a.plate_frac)
    rows = run(readers, scenes, progress=not a.quiet, on_row=dump)
    summary = summarise(rows)
    print(f"\nscenes: {len({r['index'] for r in rows})}  seed: {a.seed}  readers: {', '.join(readers)}")
    print("synthetic scenes - plumbing / relative comparison only, NOT field accuracy.\n")
    print(format_table(summary))
    errs = [r for r in rows if r["error"]]
    if errs:
        print(f"\n{len(errs)} reader errors, first: {errs[0]['reader']} scene {errs[0]['index']}: {errs[0]['error']}")
    if a.json:
        a.json.write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "summary": summary,
                                      "rows": rows}, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
