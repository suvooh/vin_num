"""Golden data for the Dart port of the decoder (app/lib/reader).

Runs the CURRENT Python reference (`vinplate.decode`, `vinplate.grammar`) on a fixed set of inputs
and writes inputs + expected outputs to ``app/test/fixtures/golden/decoder_cases.json``. The Dart
tests (`app/test/reader/golden_test.dart`) replay the same inputs and compare.

Inputs are posteriors ROUNDED to 8 decimals, and the expected outputs are computed FROM the rounded
values, so both implementations see bit-identical input. Run (slow part is the real-posterior cases,
which need the PP-OCR models):

    cd ml
    .venv\\Scripts\\python.exe -I -X utf8 scripts/make_golden.py              # everything
    .venv\\Scripts\\python.exe -I -X utf8 scripts/make_golden.py --no-real    # without real posteriors

Deterministic: every random draw is seeded. Nothing here depends on dict/set hash order (cases whose
reference output is order-unspecified, i.e. `_no_legal` with several differing frames, are not generated).
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402

from synth.text import sample_plate, sample_vin  # noqa: E402
from vinplate.decode import fuse_frames, read_line  # noqa: E402
from vinplate.decode.ctc import alternatives, ctc_align, ctc_loglik, greedy, prefix_beam_search, suspected_gaps  # noqa: E402
from vinplate.decode.lines import concat_lines  # noqa: E402
from vinplate.decode.reader import entropy_fuse  # noqa: E402
from vinplate.grammar import PLATE_GRAMMAR, VIN_GRAMMAR, parse_plate, parse_vin, vin_check_digit_ok, vin_year_candidates  # noqa: E402
from vinplate.testing import make_logp  # noqa: E402
from vinplate.vocab import ALPHABET, encode_text  # noqa: E402

GRAMMARS = {"plate": PLATE_GRAMMAR, "vin": VIN_GRAMMAR}
DEFAULT_OUT = ML.parent / "app" / "test" / "fixtures" / "golden" / "decoder_cases.json"


def r8(a) -> np.ndarray:
    return np.round(np.asarray(a, dtype=np.float64), 8)


def flat(a: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel()]


def reading_json(r) -> dict:
    return {
        "text": r.text, "valid": bool(r.valid), "status": r.status, "confidence": float(r.confidence),
        "seq_posterior": float(r.seq_posterior), "char_conf": [float(c) for c in r.char_conf],
        "inferred": [int(i) for i in r.inferred], "suspected_gaps": [int(i) for i in r.suspected_gaps],
        "grammar_cost": float(r.grammar_cost), "greedy_text": r.greedy_text,
        "n_best": [[t, float(p)] for t, p in r.n_best], "reasons": list(r.reasons), "masked": r.masked,
    }


def aux_json(lp: np.ndarray, g, reading) -> dict | None:
    """Intermediate results, so a mismatch can be localised to the sub-function."""
    if not reading.text:
        return None
    gt, gconf = greedy(lp)
    labs = encode_text(reading.text)
    spans = ctc_align(lp, labs)
    return {
        "greedy": [gt, [float(c) for c in gconf]],
        "ctc_loglik_text": float(ctc_loglik(lp, labs)),
        "ctc_loglik_greedy": float(ctc_loglik(lp, encode_text(gt))) if gt else None,
        "align": [[int(a), int(b), float(p)] for a, b, p in spans],
        "alternatives": [[[c, float(p)] for c, p in alt] for alt in alternatives(lp, reading.text, k=3)],
        "gaps": [int(x) for x in suspected_gaps(spans, lp.shape[0])],
        "beam": [[h.text, float(h.ctc_logp), float(h.prior), float(h.score)]
                 for h in prefix_beam_search(lp, g, beam=32)[:4]],
    }


# ------------------------------------------------------------------------------ (a) simulated
def sim_case(rng, i: int, kind: str) -> dict:
    text = sample_plate(rng)["text"] if kind == "plate" else sample_vin(rng)["text"]
    n = len(text)
    mode = str(rng.choice(["clean", "clean", "confuse", "flat", "unreadable", "smeared", "mixed", "mixed"]))
    kw: dict = {"frames_per_char": int(rng.integers(2, 6)), "p_peak": float(np.round(rng.uniform(0.85, 0.985), 3))}

    def pick(k: int) -> tuple[int, ...]:
        return tuple(int(x) for x in rng.choice(n, size=min(k, n), replace=False))

    if mode in ("confuse", "mixed"):
        kw["confuse"] = {}
        for pos in pick(int(rng.integers(1, 3))):
            wrong = str(rng.choice([c for c in ALPHABET if c != text[pos]]))
            kw["confuse"][pos] = (wrong, float(np.round(rng.uniform(0.1, 0.7), 3)))
    if mode in ("flat", "mixed"):
        kw["flat"] = pick(int(rng.integers(1, 3)))
    if mode in ("unreadable", "mixed"):
        kw["unreadable"] = pick(int(rng.integers(1, 3)))
    if mode == "smeared":
        kw["smeared"] = True
    lp = r8(make_logp(text, **kw))
    g = GRAMMARS[kind]
    r = read_line(lp, g)
    return {"id": f"sim{i:02d}_{kind}_{mode}", "kind": "read_line", "grammar": kind, "truth": text,
            "params": repr(kw), "t": int(lp.shape[0]), "logp": flat(lp), "expected": reading_json(r),
            "aux": aux_json(lp, g, r)}


def sim_cases(n: int = 60) -> list[dict]:
    rng = np.random.default_rng(20261111)
    return [sim_case(rng, i, "plate" if i % 3 != 2 else "vin") for i in range(n)]


# ---------------------------------------------------------------------------------- (b) fusion
def fuse_case(rng, i: int, kind: str, *, ragged: bool, weighted: bool) -> dict:
    text = sample_plate(rng)["text"] if kind == "plate" else sample_vin(rng)["text"]
    n = len(text)
    nframes = 3 if i % 2 == 0 else 2
    order = [int(x) for x in rng.permutation(n)]
    chunk = max(1, n // 4)
    lps = []
    for f in range(nframes):
        glared = tuple(order[f * chunk: (f + 1) * chunk][:2])           # a different glared span per frame
        fpc = (3, 4, 3)[f] if ragged else 3
        lps.append(r8(make_logp(text, frames_per_char=fpc, flat=glared, p_peak=0.93)))
    weights = [1.0, 1.5, 0.75][:nframes] if weighted else None
    g = GRAMMARS[kind]
    r = fuse_frames(lps, g, weights=weights)
    fused = entropy_fuse(lps, weights)
    return {"id": f"fuse{i:02d}_{kind}{'_ragged' if ragged else ''}{'_weighted' if weighted else ''}", "kind": "fuse",
            "grammar": kind, "truth": text, "ts": [int(lp.shape[0]) for lp in lps], "logps": [flat(lp) for lp in lps],
            "weights": weights, "expected": reading_json(r),
            "fused": {"t": int(fused.shape[0]), "sum": float(fused.sum()), "row0": flat(fused[0]),
                      "row_mid": flat(fused[fused.shape[0] // 2])}}


def fuse_cases(n: int = 16) -> list[dict]:
    rng = np.random.default_rng(777)
    return [fuse_case(rng, i, "plate" if i % 4 != 3 else "vin", ragged=(i % 3 == 1), weighted=(i % 5 == 4)) for i in range(n)]


def concat_cases() -> list[dict]:
    """Two-row plate: rows read separately, then joined with `concat_lines` (blank trimming)."""
    rng = np.random.default_rng(5)
    out = []
    for i in range(4):
        text = sample_plate(rng)["text"]
        cut = max(2, len(text) // 2 - 1)
        a, b = r8(make_logp(text[:cut], frames_per_char=3)), r8(make_logp(text[cut:], frames_per_char=3))
        joined = r8(concat_lines([a, b]))
        r = read_line(joined, PLATE_GRAMMAR)
        out.append({"id": f"concat{i}", "kind": "concat", "truth": text, "ts": [int(a.shape[0]), int(b.shape[0])],
                    "logps": [flat(a), flat(b)], "joined_t": int(joined.shape[0]), "joined_sum": float(joined.sum()),
                    "expected": reading_json(r)})
    return out


# -------------------------------------------------------------------------------- (c) real ones
def real_cases(max_cases: int, seed: int = 21) -> tuple[list[dict], list[dict]]:
    from eval.scenes import make_scenes
    from vinplate.enhance import stretch
    from vinplate.ocr import PpOcrEngine
    from vinplate.scene import SceneReader

    eng = PpOcrEngine()
    rd = SceneReader(eng, multi_view=False)
    reads: list[dict] = []
    fuses: list[dict] = []
    for sc in make_scenes(80, seed):
        if len(reads) >= max_cases:
            break
        lines, _ = rd.observe(sc.img)
        taken = 0
        for ln in lines:
            if sum(c.isalnum() for c in ln.text0) < 6 or taken >= 2 or len(reads) >= max_cases:
                continue
            taken += 1
            lp = r8(ln.lp0)
            g = GRAMMARS[sc.domain]
            r = read_line(lp, g)
            cid = f"real{len(reads):02d}_{sc.domain}"
            reads.append({"id": cid, "kind": "read_line", "grammar": sc.domain, "truth": sc.truth,
                          "raw": ln.text0, "t": int(lp.shape[0]), "logp": flat(lp), "expected": reading_json(r),
                          "aux": aux_json(lp, g, r)})
            if len(fuses) < 8:        # the same crop through a contrast-stretched view: realistic multi-view fusion
                lp2 = r8(eng.rec_logp(stretch(ln.crop)))
                rf = fuse_frames([lp, lp2], g)
                fuses.append({"id": f"{cid}_fuse", "kind": "fuse", "grammar": sc.domain, "truth": sc.truth,
                              "ts": [int(lp.shape[0]), int(lp2.shape[0])], "logps": [flat(lp), flat(lp2)],
                              "weights": None, "expected": reading_json(rf), "fused": None})
        print(f"  scene {sc.domain:5s} {sc.truth:20s} -> {len(reads)} real cases so far", flush=True)
    return reads, fuses


# ------------------------------------------------------------------------------ (d) grammar
PLATE_STRINGS = [
    "MH12AB1234", "KA01A1234", "DL1CAB1234", "DL1C1234", "DL1CA1234", "DL11CAB1234", "MH021234", "22BH1234AA", "23BH5678A",
    "21BH1234AA", "18BH1234A", "BH", "199CD1", "999UN9999", "1CD1", "5CC12", "03E123456X", "03E12345X", "03E1234567X",
    "UP16C0059", "TN71JE5250", "JK68RF4827", "MH12?B1234", "M?12AB1234", "??12AB1234", "MH??AB1234", "MH12AB????",
    "MH12AB123?", "?H12AB1234", "22BH123?AA", "22?H1234AA", "DL1?AB1234", "AP39?W????", "?????????", "??", "MH12AB1?34",
    "KA01AB12", "KA01AB1", "KA0AB1234", "XX12AB1234", "MH12ABCD1234", "MH1AB1234", "MH00AB1234", "MH0AB1234",
    "OR05AB1234", "TS09EA1234", "UA07C1234", "CT05AB1234", "DN09AB1234", "MH12IO1234", "MH12QQ1234", "MH12AB12345",
    "", "M", "MH", "MH12", "12AB", "AB1234", "MH12AB", "mh12ab1234", "MH 12 AB 1234", "MH12AB1234X", "ABC123",
    # dealer / trade-certificate template (DLR NN NNNN, low prior, illustrative): valid, segmentation, near misses
    "DLR012345", "DLR12345", "DLR1234", "DLR0123", "DLR123", "DLR1", "DLR", "DLR1234567", "DLR?12345", "DLR01?345", "DLR0123?5",
    "DLRO12345", "012345", "01DLR2345", "DL12345", "DLR 01 2345", "DLR01234",
]
VIN_STRINGS = [
    "MAL907KE1E6602791", "1M8GDM9AXKP042788", "MBLHA10EJEHK25347", "VINMAL907KE1E6602791", "VINNOMAL907KE1E6602791",
    "CHASSISMAL907KE1E6602791", "CHASSISNOMAL907KE1E6602791", "CHASISNOMAL907KE1E6602791", "MAL907KE1E660279",
    "MAL907KE1?6602791", "?AL907KE1E6602791", "MAL907KE1E66027??", "MA????KE1E6602791", "?????????????????",
    "TYREPRESSURE32PS1", "ENG1NEFR0NTWE1GHT", "ABC", "ABCDEF", "ABCDEFGH1", "MD2A3A1J9A2834023", "ME4WT6D55FM700822",
    "MAL907KE1E660279I", "MAL907KE1E660279O", "MAL907KE1E660279Q", "81GZ16D60VVY9", "81GZI6D6QVVY9Q", "MCUXK9BK4ABGY1HKR",
    "MAJ1234567890ABCD", "MD2A3A1J9A2834023X", "VIN", "", "MAL907KE1E6602791MAL", "MAT?????????????1", "MAL907KE10660279Z",
]
CHECK_STRINGS = ["1M8GDM9AXKP042788", "1M8GDM9A1KP042788", "11111111111111111", "MAL907KE1E6602791", "", "1M8GDM9AXKP04278",
                 "1M8GDM9AXKP0427888", "1M8GDM9AXKP04278I", "1M8GDM9AXKP04278?", "5YJSA1E26HF000337", "WP0ZZZ99ZTS392124"]


def grammar_cases(rng) -> tuple[list[dict], list[dict], list[dict]]:
    plates = PLATE_STRINGS + [sample_plate(rng)["text"] for _ in range(40)]
    vins = VIN_STRINGS + [sample_vin(rng)["text"] for _ in range(30)]
    cases = []
    for s in dict.fromkeys(plates + vins):
        row = {"s": s}
        for name, g, parse in (("plate", PLATE_GRAMMAR, parse_plate), ("vin", VIN_GRAMMAR, parse_vin)):
            p = g.log_prior(s)
            row[name] = {"valid": bool(g.is_valid(s)), "prior": None if p is None else float(p), "parsed": parse(s)}
        cases.append(row)
    checks = [{"s": s, "ok": bool(vin_check_digit_ok(s))} for s in CHECK_STRINGS + [sample_vin(rng)["text"] for _ in range(20)]]
    years = [{"s": s, "years": [int(y) for y in vin_year_candidates(s)]}
             for s in ["MAL907KE1E6602791", "MAL907KEAE6602791", "MAL907KEFE6602791", "MAL907KETE6602791", "MAL907KEYE6602791",
                       "MAL907KE5E6602791", "MAL907KE9E6602791", "MAL907KE0E6602791", "MAL907KEUE6602791", "MAL907KE?E6602791",
                       "SHORT", "MAL907KE1", "MAL907KEXE6602791", "MAL907KEWE6602791", "MAL907KEVE6602791", "MAL907KEBE6602791"]]
    return cases, checks, years


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-real", action="store_true", help="skip the cases that need the PP-OCR models")
    ap.add_argument("--real-n", type=int, default=25)
    a = ap.parse_args()

    print("simulated ...", flush=True)
    sims = sim_cases(60)
    print("fusion ...", flush=True)
    fuses = fuse_cases(16)
    concats = concat_cases()
    reals: list[dict] = []
    if not a.no_real:
        print("real posteriors (slow) ...", flush=True)
        reals, rfuses = real_cases(a.real_n)
        fuses += rfuses
    print("grammar ...", flush=True)
    g_cases, checks, years = grammar_cases(np.random.default_rng(99))

    data = {
        "meta": {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "numpy": np.__version__,
                 "python": platform.python_version(), "machine": platform.machine(),
                 "note": "logp inputs are rounded to 8 decimals; expected values are computed from the rounded inputs"},
        "year_now": datetime.date.today().year,
        "read_line": sims + reals,
        "fuse": fuses,
        "concat": concats,
        "grammar": g_cases,
        "vin_check": checks,
        "vin_year": years,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {a.out}  ({a.out.stat().st_size / 1e6:.1f} MB): {len(sims)} sim + {len(reals)} real read_line, "
          f"{len(fuses)} fuse, {len(concats)} concat, {len(g_cases)} grammar strings, {len(checks)} check digits, {len(years)} years")


if __name__ == "__main__":
    main()
