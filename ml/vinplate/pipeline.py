"""End-to-end reading of a plate / VIN *crop*: quality gate -> line recogniser -> grammar decode.

Scope of this module, stated plainly: it starts from a crop (or a rectified quad) of one
plate or one VIN line. Finding that crop in a full photo is the detector's job, which is NOT
included here yet (it needs real/synthetic scene data and a GPU; see README "Status").

The recogniser is injected (``OnnxRecognizer`` for production, any object with a ``logp``
method for tests), so decode / fusion / row-handling logic is testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from vinplate.decode import Reading, fuse_frames, log_softmax, read_line
from vinplate.decode.lines import concat_lines
from vinplate.geometry import PLATE_LINE_SHAPE, VIN_LINE_SHAPE, fit_line, to_gray
from vinplate.geometry.rows import looks_two_row, split_rows
from vinplate.grammar import PLATE_GRAMMAR, VIN_GRAMMAR, parse_plate, parse_vin
from vinplate.quality import QualityReport, assess_crop


class Recognizer(Protocol):
    def logp(self, domain: str, gray_line: np.ndarray) -> np.ndarray:
        """gray_line: [H, W] uint8 canvas for the domain -> log-probabilities [T, V]."""


class OnnxRecognizer:
    """INT8/FP32 ONNX line recogniser; runs on the Qualcomm NPU when present, else CPU."""

    def __init__(self, plate_model: str | Path, vin_model: str | Path, provider: str = "auto"):
        from vinplate.runtime import make_session

        def mk(path):
            if provider in ("qnn", "auto"):
                try:
                    return make_session(str(path), "qnn")
                except BaseException:  # SystemExit when no NPU / provider missing
                    if provider == "qnn":
                        raise
            return make_session(str(path), "cpu")

        self.sess = {"plate": mk(plate_model), "vin": mk(vin_model)}
        self.provider = {k: s.get_providers()[0] for k, s in self.sess.items()}

    def logp(self, domain: str, gray_line: np.ndarray) -> np.ndarray:
        x = gray_line[None, None].astype(np.float32) / 255.0
        logits = self.sess[domain].run(None, {"image": x})[0][0]
        return log_softmax(logits)


def _usable(r: Reading) -> bool:
    """Structured fields (state, RTO, WMI, ...) are only exposed for readings that were not
    rejected: a parsed structure invites trust that a rejected reading has not earned."""
    return r.valid and r.status != "reject"


@dataclass
class ReadResult:
    domain: str
    reading: Reading
    parsed: dict | None
    quality: QualityReport | None
    rows: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """accept / review / reject - the one field to branch on. A failed quality gate
        downgrades an otherwise clean reading to review (it was read from a poor capture)."""
        if self.reading.status == "accept" and self.quality is not None and not self.quality.ok:
            return "review"
        return self.reading.status

    def as_dict(self) -> dict:
        return {"domain": self.domain, "status": self.status, "reading": self.reading.as_dict(),
                "parsed": self.parsed, "quality": self.quality.as_dict() if self.quality else None,
                "rows": self.rows, "notes": self.notes}


class Reader:
    def __init__(self, recognizer: Recognizer, gate: bool = True):
        self.rec = recognizer
        self.gate = gate

    # ------------------------------------------------------------------ plates
    def _plate_logp(self, rgb: np.ndarray, rows: int | None) -> tuple[np.ndarray, int]:
        h, w = rgb.shape[:2]
        n = rows or (2 if looks_two_row(h, w) else 1)
        crops = split_rows(rgb) if n == 2 else [rgb]
        lps = [self.rec.logp("plate", to_gray(fit_line(c, PLATE_LINE_SHAPE))) for c in crops]
        return (concat_lines(lps) if n == 2 else lps[0]), n

    def read_plate(self, rgb: np.ndarray, rows: int | None = None) -> ReadResult:
        q = assess_crop(rgb) if self.gate else None
        lp, n = self._plate_logp(rgb, rows)
        r = read_line(lp, PLATE_GRAMMAR)
        return ReadResult("plate", r, parse_plate(r.text) if _usable(r) else None, q, n)

    def read_plate_frames(self, crops: list[np.ndarray], rows: int | None = None) -> ReadResult:
        """Several crops of the SAME plate (burst / video) - glare and smudges move between
        frames, so the fused reading recovers characters no single frame has."""
        lps, n = [], 1
        for c in crops:
            lp, n = self._plate_logp(c, rows)
            lps.append(lp)
        # frames of one plate are aligned only when they share a length; entropy_fuse resamples
        r = fuse_frames(lps, PLATE_GRAMMAR)
        quals = [assess_crop(c) for c in crops] if self.gate else []
        best_q = max(quals, key=lambda x: (x.ok, -len(x.issues))) if quals else None
        return ReadResult("plate", r, parse_plate(r.text) if _usable(r) else None, best_q, n,
                          [f"fused {len(crops)} frames"])

    # --------------------------------------------------------------------- VIN
    def read_vin(self, rgb: np.ndarray) -> ReadResult:
        q = assess_crop(rgb) if self.gate else None
        lp = self.rec.logp("vin", to_gray(fit_line(rgb, VIN_LINE_SHAPE)))
        r = read_line(lp, VIN_GRAMMAR)
        return ReadResult("vin", r, parse_vin(r.text) if _usable(r) else None, q)

    def read_vin_frames(self, crops: list[np.ndarray]) -> ReadResult:
        lps = [self.rec.logp("vin", to_gray(fit_line(c, VIN_LINE_SHAPE))) for c in crops]
        r = fuse_frames(lps, VIN_GRAMMAR)
        quals = [assess_crop(c) for c in crops] if self.gate else []
        best_q = max(quals, key=lambda x: (x.ok, -len(x.issues))) if quals else None
        return ReadResult("vin", r, parse_vin(r.text) if _usable(r) else None, best_q,
                          notes=[f"fused {len(crops)} frames"])
