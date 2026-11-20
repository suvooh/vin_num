"""One call from a capture bundle to an authenticity decision.

    auth = Authenticator(pad_model="build_pad/pad_int8.onnx")           # model optional
    d = auth.assess(frames, payload=jpeg_bytes, roi=plate_box, flash_on=on_frame,
                    plate_mask=mask, hsrp_expected=True, attestation=server_verified)
    d.status        # reject | review | no_flags
    d.not_run       # every check that lacked inputs - never silently a pass

Runs whatever the inputs allow: provenance/metadata (needs the original file bytes),
classical pixel forensics, the learned recapture model on native-resolution patches (needs the
model), flash retroreflection (needs a flash-ON frame + plate mask), planarity (needs >= 2 frames
with real camera translation). ``frames[0]`` is the primary flash-OFF frame, full resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from vinplate.authenticity.active import planarity, retroreflection
from vinplate.authenticity.forensics import analyze_pixels
from vinplate.authenticity.fusion import AuthDecision, decide
from vinplate.authenticity.provenance import ProvenanceReport, analyze_provenance

PATCH, K = 96, 4


def pick_patches(rgb: np.ndarray, roi: tuple[int, int, int, int] | None = None, k: int = K,
                 size: int = PATCH) -> np.ndarray | None:
    """k native-resolution (NOT resized) patches from inside ``roi`` (x0, y0, x1, y1), preferring
    textured, non-overlapping locations: a patch of flat sky/paint says nothing about moire."""
    x0, y0, x1, y1 = roi or (0, 0, rgb.shape[1], rgb.shape[0])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if x1 - x0 < size or y1 - y0 < size:
        return None
    g = rgb[..., 1].astype(np.float32)
    cands = []
    step = size // 2
    for y in range(y0, y1 - size + 1, step):
        for x in range(x0, x1 - size + 1, step):
            cands.append((float(g[y : y + size, x : x + size].std()), x, y))
    cands.sort(reverse=True)
    chosen: list[tuple[int, int]] = []
    for _, x, y in cands:
        if all(abs(x - cx) >= size or abs(y - cy) >= size for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) == k:
            break
    while len(chosen) < k:                      # small ROI: allow overlap rather than fail
        chosen.append(chosen[len(chosen) % max(1, len(chosen))] if chosen else (x0, y0))
    return np.stack([rgb[y : y + size, x : x + size] for x, y in chosen])


class PadModel:
    """ONNX recapture classifier; runs on the NPU/GPU when asked and available, else CPU."""

    def __init__(self, path: str | Path, provider: str = "cpu"):
        from vinplate.runtime import make_session

        self.sess = make_session(str(path), provider)
        self.provider = self.sess.get_providers()[0]

    def score(self, patches: np.ndarray) -> float:
        x = patches.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        logits = self.sess.run(None, {"patches": x})[0]
        return float(1 / (1 + np.exp(-float(np.mean(logits)))))


class Authenticator:
    def __init__(self, pad_model: PadModel | str | Path | Callable[[np.ndarray], float] | None = None,
                 provider: str = "cpu"):
        if isinstance(pad_model, (str, Path)):
            pad_model = PadModel(pad_model, provider)
        self.pad = pad_model
        self.pad_error: str | None = None    # why no recapture model is loaded, when from_default could not

    @classmethod
    def from_default(cls, provider: str = "cpu") -> "Authenticator":
        """The shipped PadNet (``pad_int8.onnx``) if it can be found and loaded, else an Authenticator
        with no recapture model. A missing or unloadable model is never an error: the check is then
        reported in ``not_run`` (``pad_error`` says why) instead of silently counting as a pass."""
        from vinplate.runtime import find_model

        try:
            path = find_model("pad_int8.onnx")
        except FileNotFoundError as exc:
            auth = cls(None)
            auth.pad_error = f"recapture model not found: {exc}"
            return auth
        try:
            return cls(PadModel(path, provider))
        except Exception as exc:   # noqa: BLE001 - unusable accelerator, or a corrupt file
            if provider != "cpu":
                try:
                    auth = cls(PadModel(path, "cpu"))
                    auth.pad_error = f"provider '{provider}' unavailable ({exc}); using CPU"
                    return auth
                except Exception as exc2:   # noqa: BLE001
                    exc = exc2
            auth = cls(None)
            auth.pad_error = f"recapture model could not be loaded: {exc}"
            return auth

    def recapture_score(self, rgb: np.ndarray, roi: tuple[int, int, int, int] | None = None):
        """(score in 0..1 or None, note or None) from native-resolution patches inside ``roi``.
        ``(None, None)`` when no model is loaded; a note when the region is too small to judge."""
        if self.pad is None:
            return None, None
        note = None
        patches = pick_patches(rgb, roi)
        if patches is None and roi is not None:
            # A plate in a full-car framing is often smaller than one native patch. Screen / print
            # artefacts (moire, halftone, glare) cover the whole recaptured frame, so judge a window
            # around the plate that is at least 3 patches across instead of giving up.
            x0, y0, x1, y1 = roi
            cx, cy, half = (x0 + x1) // 2, (y0 + y1) // 2, int(1.5 * PATCH)
            wide = (max(0, cx - half), max(0, cy - half), min(rgb.shape[1], cx + half), min(rgb.shape[0], cy + half))
            patches = pick_patches(rgb, wide)
            if patches is not None:
                note = "plate smaller than one native patch: recapture model judged the surrounding area"
        if patches is None:
            return None, "recapture model not run: region smaller than one native patch"
        score = self.pad.score(patches) if hasattr(self.pad, "score") else float(self.pad(patches))
        return float(score), note

    def assess(self, frames: list[np.ndarray], *, payload: bytes | None = None,
               roi: tuple[int, int, int, int] | None = None, flash_on: np.ndarray | None = None,
               plate_mask: np.ndarray | None = None, hsrp_expected: bool = False,
               attestation: dict | None = None, require_assurance: str | None = None) -> AuthDecision:
        primary = frames[0]
        prov = analyze_provenance(payload) if payload else ProvenanceReport()
        forensics = analyze_pixels(primary, was_jpeg=bool(payload))
        model_scores: dict[str, float] = {}
        notes: list[str] = []
        score, note = self.recapture_score(primary, roi)
        if score is not None:
            model_scores["recapture"] = score
        if note:
            notes.append(note)
        active: dict = {}
        if flash_on is not None and plate_mask is not None:
            active["retroreflection"] = retroreflection(primary, flash_on, plate_mask, hsrp_expected=hsrp_expected)
        if len(frames) >= 2:
            active["planarity"] = planarity(frames[0], frames[1])
        d = decide(prov, forensics, attestation, model_scores, require_assurance, active)
        d.reasons.extend(notes)
        return d
