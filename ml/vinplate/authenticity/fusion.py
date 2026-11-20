"""Combine authenticity evidence into one decision. Deliberately conservative.

Rules, in order:
  1. A hard provenance flag (declared AI / screenshot / print, failed C2PA validation,
     undecodable file) or a failed attestation check  ->  reject.
  2. Presentation-attack (photo of a screen or print) evidence is summed from independent signals,
     each with a weight. >= REVIEW_AT -> review;  >= REJECT_AT -> reject. A single weak cue can
     never reject: wrongly rejecting a genuine customer is a real cost too.
        retroreflection failed under flash (HSRP expected)   2.0   physics: strongest
        learned recapture score >= 0.9                       2.0
        learned recapture score >= 0.5                       1.0
        moire / chroma-moire pattern                         1.0
        horizontal (rolling-shutter) banding                 0.5   weak
        motion fits a single plane                           0.5   weak (see active.planarity)
        flash lit a small hot-spot, not the whole plate      1.0   glare on glass (see active.gain_uniformity)
        screen / interface text visible in the scene         0.5-3.0  (see scene.screen_context: URL 1.0,
                                                                       >=3 interface words 1.5, >=14 text lines 0.5)
  3. Other soft flags (editor metadata, stale EXIF thumbnail, tamper / AI-generated model
     above threshold, ELA / noise / JPEG-grid flags)  ->  review.
  4. Otherwise ``no_flags``. That is NOT "authentic". ``assurance`` says what actually backs the
     image ("none" / "attested" / "c2pa") and ``positive`` lists affirmative physical evidence
     (e.g. "retroreflection_passed"). Nothing here may auto-approve a high-value action on
     assurance "none" with no positive evidence; that is the caller's policy (``require_assurance``).

Every check that did not run is listed in ``not_run``: a missing model or a missing flash frame
is never silently treated as a pass. All weights/thresholds are PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vinplate.authenticity.forensics import ForensicsReport
from vinplate.authenticity.provenance import ProvenanceReport

ASSURANCE_ORDER = ["none", "attested", "c2pa"]
MODEL_THRESHOLDS = {"tamper": 0.5, "ai_generated": 0.5}          # generic models -> review
REVIEW_AT, REJECT_AT = 1.0, 3.0
PA_FLAGS = {"periodic_pattern": 1.0, "chroma_periodic_pattern": 1.0, "horizontal_banding": 0.5}
ACTIVE_CHECKS = ("retroreflection", "planarity")


@dataclass
class AuthDecision:
    status: str                       # reject | review | no_flags
    assurance: str                    # none | attested | c2pa
    reasons: list[str] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)
    positive: list[str] = field(default_factory=list)
    pa_score: float = 0.0
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _verdict(x) -> str | None:
    if x is None:
        return None
    return x.get("verdict") if isinstance(x, dict) else x.verdict


def decide(
    provenance: ProvenanceReport,
    forensics: ForensicsReport | None = None,
    attestation: dict | None = None,
    model_scores: dict[str, float] | None = None,
    require_assurance: str | None = None,
    active: dict | None = None,
    context: dict | None = None,
) -> AuthDecision:
    """``attestation``: result of the SERVER-side check of the capture, e.g.
    ``{"verified": True, "device_integrity": True, "nonce_ok": True, "hash_matches": True}``.
    ``active``: ``{"retroreflection": RetroResult, "planarity": PlanarityResult}`` from
    ``authenticity.active`` (needs the flash-off/flash-on burst). ``model_scores``: optional
    trained detectors, keys ``recapture`` / ``tamper`` / ``ai_generated`` in 0..1. ``context``: the scene's
    ``screen_context`` (interface text read by the OCR); its ``score`` is ALREADY the weighted presentation-attack
    contribution (0..3)."""
    reasons: list[str] = []
    reject = review = False
    positive: list[str] = []
    model_scores = model_scores or {}

    for f in provenance.hard:
        reject = True
        reasons.append(f"{f.code}: {f.detail}")

    assurance = "none"
    if attestation is not None:
        bad = [k for k in ("verified", "device_integrity", "nonce_ok", "hash_matches") if attestation.get(k) is False]
        if bad:
            reject = True
            reasons.append(f"attestation failed: {', '.join(bad)}")
        elif attestation.get("verified"):
            assurance = "attested"
    if provenance.c2pa.get("verified"):
        assurance = "c2pa"

    for f in provenance.soft:
        review = True
        reasons.append(f"{f.code}: {f.detail}")

    # ---- presentation attack: weighted, independent signals
    pa = 0.0
    parts: list[str] = []
    if forensics is not None:
        for fl in forensics.flags:
            if fl in PA_FLAGS:
                pa += PA_FLAGS[fl]
                parts.append(f"{fl} (+{PA_FLAGS[fl]})")
            else:
                review = True
                reasons.append(f"pixel forensics: {fl} ({forensics.scores})")
    rs = model_scores.get("recapture")
    if rs is not None:
        w = 2.0 if rs >= 0.9 else (1.0 if rs >= 0.5 else 0.0)
        if w:
            pa += w
            parts.append(f"learned recapture score {rs:.2f} (+{w})")
    retro = (active or {}).get("retroreflection")
    rv = _verdict(retro)
    if rv == "not_retroreflective":
        pa += 2.0
        parts.append("plate did not retroreflect under flash (+2.0)")
    elif rv == "specular_hotspot":
        pa += 1.0
        parts.append("plate showed a specular hot-spot under flash, not a retroreflective response (+1.0)")
    elif rv == "retroreflective":
        positive.append("retroreflection_passed")
    if context and float(context.get("score") or 0.0) > 0.0:
        w = float(context["score"])
        pa += w
        parts.append(f"screen / interface context in the scene (+{w:g}): " + "; ".join(context.get("reasons") or []))
    pv = _verdict((active or {}).get("planarity"))
    if pv == "planar":
        pa += 0.5
        parts.append("scene motion fits a single plane (+0.5)")
    if pa >= REJECT_AT:
        reject = True
        reasons.append(f"presentation attack suspected, score {pa:.1f} >= {REJECT_AT}: " + "; ".join(parts))
    elif pa >= REVIEW_AT:
        review = True
        reasons.append(f"possible presentation attack, score {pa:.1f}: " + "; ".join(parts))
    elif parts:
        reasons.append("weak presentation-attack hints only: " + "; ".join(parts))

    # ---- other models (tamper / AI-generated)
    for name, thr in MODEL_THRESHOLDS.items():
        if name in model_scores and model_scores[name] >= thr:
            review = True
            reasons.append(f"model '{name}' score {model_scores[name]:.2f} >= {thr}")

    not_run = [n for n in ("recapture", *MODEL_THRESHOLDS) if n not in model_scores]
    not_run += [n for n in ACTIVE_CHECKS if _verdict((active or {}).get(n)) in (None, "inconclusive")]

    if require_assurance and ASSURANCE_ORDER.index(assurance) < ASSURANCE_ORDER.index(require_assurance):
        review = True
        reasons.append(f"assurance '{assurance}' below required '{require_assurance}'")

    status = "reject" if reject else ("review" if review else "no_flags")
    return AuthDecision(status, assurance, reasons, not_run, positive, round(pa, 2),
                        {"provenance": provenance.as_dict(),
                         "forensics": forensics.as_dict() if forensics else None,
                         "screen_context": context,
                         "active": {k: (v.as_dict() if hasattr(v, "as_dict") else v) for k, v in (active or {}).items()}})
