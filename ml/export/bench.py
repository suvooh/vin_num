"""Latency of an ONNX model on CPU vs the Qualcomm NPU (Hexagon HTP via QNN execution provider).

    python -m export.bench build/rec_plate_int8.onnx --provider qnn --runs 200

Note on what this measures: session.run() wall time for a single line (batch 1), after warm-up,
including host<->NPU transfer. It does NOT include image decode, rectification, or the
Python-side grammar decode. The first-session-creation time (graph compile for HTP) is
reported separately; production should ship a pre-compiled QNN context binary so users never
pay it.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from vinplate.runtime import make_session  # noqa: F401  (re-exported: lives in vinplate so the package is self-contained)


def bench(path: str, provider: str, runs: int, warmup: int = 10) -> dict:
    t0 = time.perf_counter()
    sess = make_session(path, provider)
    create_s = time.perf_counter() - t0
    inp = sess.get_inputs()[0]
    x = np.random.rand(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"provider": sess.get_providers()[0], "create_s": round(create_s, 2),
            "median_ms": round(statistics.median(ts), 2), "p95_ms": round(ts[int(0.95 * len(ts)) - 1], 2),
            "min_ms": round(ts[0], 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--provider", choices=["cpu", "npu", "qnn", "gpu"], default="cpu")
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()
    try:
        print(a.model, bench(a.model, a.provider, a.runs))
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
