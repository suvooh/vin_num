"""Evaluate the reader on labelled crops and report the numbers that matter for an
identity workflow.

    # REAL data: a folder with labels.csv (image,text,domain[,tags]) of plate / VIN crops
    python -m eval.run_eval --labels data/real_eval --plate-model build/rec_plate_int8.onnx \\
                            --vin-model build/rec_vin_int8.onnx

    # plumbing check on SYNTHETIC data (NOT a real-world accuracy number)
    python -m eval.run_eval --synthetic 300 --plate-model ... --vin-model ...

Headline metrics, in order of importance:
  accepted_wrong_rate   of the readings the system ACCEPTS automatically, the share that is
                        wrong. This is the dangerous error (a wrong plate/VIN passed as
                        verified) and the number to drive down and to put an SLO on.
  coverage              share of inputs accepted without human review (the cost of safety).
  accept_precision      1 - accepted_wrong_rate.
  review_waste          share of reviewed items whose reading was actually right (over-flagging).
  exact / cer           raw strict accuracy and character error rate, status ignored.

Always report these with the size of the set and its provenance; a number from synthetic
data says the pipeline works, nothing about real photographs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from training.train_rec import lev


def matches(pred: str, truth: str) -> bool:
    """Exact match, except a '?' in the TRUTH (synthetic only: a character that was hidden)
    matches anything. Real ground truth never contains '?'."""
    return len(pred) == len(truth) and all(t == "?" or p == t for p, t in zip(pred, truth))


def summarize(rows: list[dict]) -> dict:
    """rows: {truth, pred, status, domain, tags}. Pure function - unit-testable."""
    def block(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        acc = [r for r in rs if r["status"] == "accept"]
        rev = [r for r in rs if r["status"] == "review"]
        rej = [r for r in rs if r["status"] == "reject"]
        ok = lambda r: matches(r["pred"], r["truth"])  # noqa: E731
        chars = sum(len(r["truth"]) for r in rs)
        return {
            "n": n,
            "coverage": len(acc) / n, "review_rate": len(rev) / n, "reject_rate": len(rej) / n,
            "accepted": len(acc), "accepted_wrong": sum(not ok(r) for r in acc),
            "accepted_wrong_rate": (sum(not ok(r) for r in acc) / len(acc)) if acc else None,
            "review_waste": (sum(ok(r) for r in rev) / len(rev)) if rev else None,
            "exact": sum(ok(r) for r in rs) / n,
            "cer": sum(lev(r["pred"], r["truth"].replace("?", "")) if "?" not in r["truth"] else lev(r["pred"], r["truth"]) for r in rs) / max(chars, 1),
        }
    out = {"overall": block(rows), "by_domain": {}, "by_tag": {}}
    for d in {r["domain"] for r in rows}:
        out["by_domain"][d] = block([r for r in rows if r["domain"] == d])
    tags = defaultdict(list)
    for r in rows:
        for t in set(r.get("tags", [])) or {"none"}:
            tags[t].append(r)
    out["by_tag"] = {t: block(rs) for t, rs in sorted(tags.items()) if len(rs) >= 5}
    return out


def _fmt(b: dict) -> str:
    if not b.get("n"):
        return "n=0"
    f = lambda v: "  -  " if v is None else f"{v * 100:5.1f}%"  # noqa: E731
    return (f"n={b['n']:5d} coverage {f(b['coverage'])} review {f(b['review_rate'])} reject {f(b['reject_rate'])} | "
            f"ACCEPTED-WRONG {b['accepted_wrong']}/{b['accepted']} ({f(b['accepted_wrong_rate'])}) | "
            f"exact {f(b['exact'])} cer {b['cer'] * 100:.2f}% review-waste {f(b['review_waste'])}")


def run(items, reader) -> list[dict]:
    rows = []
    for img, truth, domain, tags in items:
        res = reader.read_vin(img) if domain == "vin" else reader.read_plate(img)
        rows.append({"truth": truth, "pred": res.reading.text, "status": res.status, "domain": domain, "tags": tags})
    return rows


def load_labelled(folder: Path):
    with open(folder / "labels.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield (np.asarray(Image.open(folder / r["image"]).convert("RGB")), r["text"].strip().upper(),
                   r.get("domain", "plate"), [t for t in r.get("tags", "").split(";") if t])


def synthetic(n: int, seed: int):
    from synth.sampler import sample_plate_whole, sample_vin_whole

    rng = np.random.default_rng(seed)
    for i in range(n):
        ln = sample_vin_whole(rng) if i % 3 == 2 else sample_plate_whole(rng)
        yield ln.img, ln.text, ln.domain, [ln.profile, *ln.tags]


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--labels", type=Path)
    g.add_argument("--synthetic", type=int)
    ap.add_argument("--plate-model", required=True)
    ap.add_argument("--vin-model", required=True)
    ap.add_argument("--provider", default="auto")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    from vinplate.pipeline import OnnxRecognizer, Reader

    reader = Reader(OnnxRecognizer(a.plate_model, a.vin_model, a.provider))
    items = load_labelled(a.labels) if a.labels else synthetic(a.synthetic, a.seed)
    src = f"REAL labelled set: {a.labels}" if a.labels else f"SYNTHETIC ({a.synthetic}) - plumbing check only, NOT a real-world number"
    rows = run(items, reader)
    rep = summarize(rows)
    print(f"=== {src}\nrecogniser provider: {getattr(reader.rec, 'provider', '?')}")
    print("OVERALL ", _fmt(rep["overall"]))
    for d, b in rep["by_domain"].items():
        print(f"  {d:6s}", _fmt(b))
    for t, b in rep["by_tag"].items():
        print(f"  tag:{t:12s}", _fmt(b))
    if a.json:
        a.json.write_text(json.dumps({"source": src, **rep}, indent=2, default=float))


if __name__ == "__main__":
    main()
