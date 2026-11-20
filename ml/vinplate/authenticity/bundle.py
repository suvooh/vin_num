"""Several images of ONE vehicle -> one authenticity / liveness decision.

``Authenticator.assess`` judges a primary frame (plus at most one flash frame and one second frame).
A real capture session produces a *bundle*: a flash-off primary, a flash-on frame, and a few frames
from other angles. This module runs every check that bundle allows and answers two different
questions, which must not be confused:

  * ``status``    - were any red flags raised?   reject | review | no_flags
  * ``liveness``  - is there POSITIVE physical evidence of a real scene in front of the camera?
                    confirmed | suspected_spoof | not_established

``no_flags`` is NOT liveness. A well-made screen replay with moire suppressed and no flash frame
raises no flag at all; it ends as ``no_flags`` + ``not_established`` and the caller's policy
(``require_liveness=True``) decides what that is worth. ``confirmed`` needs exactly ONE physical cue and
no presentation-attack evidence against it:

  * retroreflection: an HSRP plate brightened far more than its surround under flash, UNIFORMLY across
    the plate (a bright spot on glass is reported as ``specular_hotspot`` and counts against).

Two things that used to count as proof no longer do, because a REAL monitor photo defeated them:

  * parallax (motion between two angles does not fit a single plane) is only listed as SUPPORTING
    evidence: a screen held in front of a real office background also shows depth;
  * a LOW learned-recapture (PadNet) score is never positive evidence: it scored a photo of a monitor
    0.003 ("genuine"). PadNet can only count AGAINST a bundle.

Screen / interface text in the scene (``screen_context``, from the OCR) adds to the presentation-attack
score and makes the bundle a ``suspected_spoof`` from 1.0 upwards.

Honest limits (carried into the output, not hidden):
  * retroreflection applies only to HSRP plates, needs a plate-metered, exposure-locked flash pair,
    and has only been exercised against a toy physics model - never real phones or real HSRP plates;
  * the basis of every ``confirmed`` verdict is recorded (``evidence['liveness_basis']``);
  * every threshold used here lives in ``fusion.py`` / ``active.py`` and is PROVISIONAL.

Recapture score is the MAX over frames: one re-photographed frame in the bundle is an attack by
itself. Hard provenance flags and failed attestation in ANY frame reject the bundle.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from vinplate.authenticity.active import PlanarityResult, RetroResult, planarity, retroreflection
from vinplate.authenticity.assess import Authenticator
from vinplate.authenticity.forensics import ForensicsReport, analyze_pixels
from vinplate.authenticity.fusion import REVIEW_AT, AuthDecision, decide
from vinplate.authenticity.provenance import HARD, Finding, ProvenanceReport, analyze_provenance

ROLES = ("primary", "flash_on", "angle", "other")
MASK_ERODE = 0.10                    # shrink the plate ROI by this fraction per side for the flash mask
SPOOF_VERDICTS = ("not_retroreflective", "specular_hotspot")
SCREEN_SPOOF_AT = 1.0                # screen_context score from which the bundle is a suspected spoof by itself
_ORDER = {"no_flags": 0, "review": 1, "reject": 2}
# Hard provenance findings that say the file is not a clean capture but do NOT say "this is a
# re-photographed / generated image" (decode failures, broken C2PA). Every other hard finding is a
# declared screenshot / print / AI origin, i.e. evidence the scene was not live.
_HARD_NOT_A_SPOOF_CLAIM = {"undecodable", "c2pa_unreadable", "c2pa_invalid"}
_MIN_BASELINE_PX = float(inspect.signature(planarity).parameters["min_baseline_px"].default)


# ------------------------------------------------------------------------------ data model
@dataclass
class Frame:
    rgb: np.ndarray                                    # uint8 HxWx3
    role: str = "primary"                              # primary | flash_on | angle | other
    payload: bytes | None = None                       # original file bytes (provenance / EXIF / C2PA)
    roi: tuple[int, int, int, int] | None = None       # plate region x0,y0,x1,y1 in THIS frame's pixels
    name: str = ""


@dataclass
class BundleDecision:
    status: str                                        # reject | review | no_flags
    liveness: str                                      # confirmed | suspected_spoof | not_established
    assurance: str = "none"                            # none | attested | c2pa
    reasons: list[str] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)
    positive: list[str] = field(default_factory=list)
    pa_score: float = 0.0
    per_frame: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"status": self.status, "liveness": self.liveness, "assurance": self.assurance,
                "reasons": list(self.reasons), "not_run": list(self.not_run), "positive": list(self.positive),
                "pa_score": self.pa_score, "per_frame": self.per_frame, "evidence": self.evidence}


# ------------------------------------------------------------------------------- helpers
def _clip_roi(roi, shape) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    h, w = shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in roi)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _plate_mask(shape, roi, erode: float = MASK_ERODE) -> np.ndarray:
    x0, y0, x1, y1 = roi
    dx, dy = int(round((x1 - x0) * erode)), int(round((y1 - y0) * erode))
    m = np.zeros(shape[:2], bool)
    m[y0 + dy : y1 - dy, x0 + dx : x1 - dx] = True
    return m


def _match_size(ref: np.ndarray, mov: np.ndarray) -> np.ndarray | None:
    """``mov`` at ``ref``'s size, or None if their aspect ratios differ too much to be the same
    framing (e.g. portrait vs landscape): comparing those would measure nothing."""
    if mov.shape[:2] == ref.shape[:2]:
        return mov
    (h, w), (mh, mw) = ref.shape[:2], mov.shape[:2]
    if abs((w / h) / (mw / mh) - 1.0) > 0.03:
        return None
    return np.asarray(Image.fromarray(mov).resize((w, h), Image.BILINEAR))


def _compatible(a: str, b: str) -> bool:
    """Could ``a`` and ``b`` be two reads of the same identifier? '?' matches any single character
    and may also stand for nothing (the decoder inserts '?' at suspected dropped characters), but a
    known character can never be added, dropped or changed."""
    n, m = len(a), len(b)
    ok = [[False] * (m + 1) for _ in range(n + 1)]
    ok[0][0] = True
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            v = False
            if i and j and (a[i - 1] == b[j - 1] or "?" in (a[i - 1], b[j - 1])):
                v = ok[i - 1][j - 1]
            if not v and i and a[i - 1] == "?":
                v = ok[i - 1][j]
            if not v and j and b[j - 1] == "?":
                v = ok[i][j - 1]
            ok[i][j] = v
    return ok[n][m]


def read_conflicts(readings: list[str | None], labels: list[str] | None = None) -> list[str]:
    """Human-readable conflicts between reads of the same plate/VIN taken from different frames."""
    clean = [(i, "".join(str(r).upper().split())) for i, r in enumerate(readings) if r and str(r).strip()]
    out = []
    for x in range(len(clean)):
        for y in range(x + 1, len(clean)):
            (i, a), (j, b) = clean[x], clean[y]
            if not _compatible(a, b):
                la = labels[i] if labels and i < len(labels) else f"read{i}"
                lb = labels[j] if labels and j < len(labels) else f"read{j}"
                out.append(f"plate text differs between frames: '{a}' ({la}) vs '{b}' ({lb})")
    return out


def _merge_provenance(reports: list[tuple[str, ProvenanceReport]]) -> ProvenanceReport:
    """One report for the decision engine. The same finding in several frames is listed once and
    tagged with every frame it appeared in. C2PA counts as verified only when every frame that
    carries a payload verified."""
    merged = ProvenanceReport()
    seen: dict[tuple[str, str, str], list[str]] = {}
    for name, rep in reports:
        for f in rep.findings:
            seen.setdefault((f.code, f.severity, f.detail), []).append(name)
    for (code, sev, detail), names in seen.items():
        merged.findings.append(Finding(code, sev, f"[{', '.join(names)}] {detail}"))
    if reports:
        merged.c2pa = {"present": any(r.c2pa.get("present") for _, r in reports),
                       "verified": True if all(r.c2pa.get("verified") for _, r in reports) else None}
        merged.sha256 = reports[0][1].sha256
        merged.dhash = reports[0][1].dhash
    return merged


def _merge_forensics(reports: list[tuple[str, ForensicsReport]]) -> ForensicsReport | None:
    if not reports:
        return None
    merged = ForensicsReport()
    for name, rep in reports:
        for k, v in rep.scores.items():
            merged.scores[k] = max(merged.scores.get(k, float("-inf")), v)
        for fl in rep.flags:
            if fl not in merged.flags:       # a flag is one cue, however many frames show it
                merged.flags.append(fl)
        merged.notes += [f"[{name}] {n}" for n in rep.notes]
    merged.calibrated = all(r.calibrated for _, r in reports)
    return merged


def _bump(status: str, to: str) -> str:
    return to if _ORDER[to] > _ORDER[status] else status


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _verdict(x) -> str | None:
    return None if x is None else (x.get("verdict") if isinstance(x, dict) else x.verdict)


# ------------------------------------------------------------------------------ the entry point
def assess_bundle(
    frames: list[Frame],
    *,
    authenticator: Authenticator | None = None,
    hsrp_expected: bool = False,
    attestation: dict | None = None,
    require_liveness: bool = False,
    require_assurance: str | None = None,
    readings: list[str | None] | None = None,
    screen_context: dict | None = None,
) -> BundleDecision:
    """``frames``: the capture bundle. ``readings``: optional OCR reads of the plate (strings that may
    contain '?'), used only to check that the frames show the same text. ``attestation``: the
    SERVER-verified attestation result, exactly as for ``decide``. ``screen_context``: the scene reader's
    ``SceneResult.screen_context`` (interface text visible in the photos), or None."""
    if not frames:
        raise ValueError("assess_bundle needs at least one frame")
    for f in frames:
        if f.role not in ROLES:
            raise ValueError(f"unknown frame role {f.role!r}; expected one of {ROLES}")
    auth = authenticator or Authenticator()
    names = [f.name or f"frame{i}" for i, f in enumerate(frames)]
    p_idx = next((i for i, f in enumerate(frames) if f.role == "primary"), 0)
    primary = frames[p_idx]
    p_roi = _clip_roi(primary.roi, primary.rgb.shape)
    notes: list[str] = []                                  # extra reasons produced here
    not_run_why: dict[str, str] = {}

    # ---- per-frame passive checks ------------------------------------------------------
    prov_reports, for_reports, per_frame = [], [], []
    recapture: dict[str, float] = {}
    for i, f in enumerate(frames):
        roi = _clip_roi(f.roi, f.rgb.shape)
        if roi is None and f.role == "flash_on":
            roi = p_roi                                    # same burst: the plate is where it was
        entry: dict = {"index": i, "name": names[i], "role": f.role,
                       "roi": [int(v) for v in roi] if roi else None, "size": [int(f.rgb.shape[1]), int(f.rgb.shape[0])]}
        if f.payload:
            prov = analyze_provenance(f.payload)
            prov_reports.append((names[i], prov))
            entry["provenance"] = {"hard": [x.code for x in prov.hard], "soft": [x.code for x in prov.soft],
                                   "sha256": prov.sha256, "dhash": prov.dhash}
        for_rep = analyze_pixels(f.rgb, was_jpeg=bool(f.payload))
        for_reports.append((names[i], for_rep))
        entry["forensics"] = {"scores": {k: round(float(v), 3) for k, v in for_rep.scores.items()}, "flags": list(for_rep.flags)}
        score, note = auth.recapture_score(f.rgb, roi)
        if score is not None:
            recapture[names[i]] = score
        entry["recapture"] = None if score is None else round(float(score), 3)
        if note:
            entry["note"] = note
            notes.append(f"[{names[i]}] {note}")
        per_frame.append(entry)
    if auth.pad is None:
        not_run_why["recapture"] = getattr(auth, "pad_error", None) or "no recapture model loaded"
    model_scores = {}
    if recapture:
        worst = max(recapture, key=recapture.get)
        model_scores["recapture"] = recapture[worst]
        if recapture[worst] >= 0.5:
            notes.append(f"highest learned recapture score is in frame '{worst}' ({recapture[worst]:.2f})")

    # ---- flash: retroreflection ----------------------------------------------------------
    active: dict = {}
    flash_frames = [i for i, f in enumerate(frames) if f.role == "flash_on"]
    if flash_frames:
        fi = flash_frames[0]
        if len(flash_frames) > 1:
            notes.append(f"{len(flash_frames)} flash-on frames supplied; only '{names[fi]}' was used")
        on = _match_size(primary.rgb, frames[fi].rgb)
        if p_roi is None:
            not_run_why["retroreflection"] = "no plate region (ROI) for the primary frame, so no plate mask can be built"
        elif on is None:
            not_run_why["retroreflection"] = "flash-on frame has a different aspect ratio than the primary frame"
        else:
            retro = retroreflection(primary.rgb, on, _plate_mask(primary.rgb.shape, p_roi), hsrp_expected=hsrp_expected)
            active["retroreflection"] = retro
            if retro.verdict == "inconclusive":
                not_run_why["retroreflection"] = "; ".join(retro.reasons) or "inconclusive"
    else:
        not_run_why["retroreflection"] = "no flash-on frame in the bundle"

    # ---- angles: parallax / planarity ----------------------------------------------------
    plan_results: list[tuple[str, PlanarityResult]] = []
    angle_frames = [i for i, f in enumerate(frames) if f.role == "angle"]
    skipped: list[str] = []
    for ai in angle_frames:
        mov = _match_size(primary.rgb, frames[ai].rgb)
        if mov is None:
            skipped.append(names[ai])
            continue
        plan_results.append((names[ai], planarity(primary.rgb, mov)))
    conclusive = [(n, r) for n, r in plan_results if r.verdict in ("planar", "non_planar")]
    non_planar = [(n, r) for n, r in conclusive
                  if r.verdict == "non_planar" and (r.baseline_px or 0.0) >= _MIN_BASELINE_PX]
    if non_planar:
        active["planarity"] = non_planar[0][1]
    elif conclusive and all(r.verdict == "planar" for _, r in conclusive):
        active["planarity"] = conclusive[0][1]             # ONE weak cue however many angles agree
    elif conclusive:
        active["planarity"] = PlanarityResult("inconclusive", reasons=["angle frames gave conflicting or ambiguous results"])
    elif plan_results:
        active["planarity"] = plan_results[0][1]
    if "planarity" not in active or active["planarity"].verdict == "inconclusive":
        if not angle_frames:
            not_run_why["planarity"] = "no angle frames in the bundle"
        elif skipped and not plan_results:
            not_run_why["planarity"] = f"angle frames {skipped} differ in aspect ratio from the primary"
        else:
            why = [r for _, r in plan_results if r.reasons]
            not_run_why["planarity"] = "; ".join(why[0].reasons) if why else "inconclusive"
    if skipped and plan_results:
        notes.append(f"angle frames {skipped} skipped: aspect ratio differs from the primary")

    # ---- consistency of the reads across frames -----------------------------------------
    conflicts: list[str] = []
    if readings:
        labels = names if len(readings) == len(frames) else None
        conflicts = read_conflicts(list(readings), labels)

    # ---- fuse -------------------------------------------------------------------------------
    prov = _merge_provenance(prov_reports)
    forensics = _merge_forensics(for_reports)
    d: AuthDecision = decide(prov, forensics, attestation, model_scores, require_assurance, active, screen_context)
    status = d.status
    reasons = list(d.reasons) + notes
    fl_by_frame = {e["name"]: e["forensics"]["flags"] for e in per_frame if e["forensics"]["flags"]}
    if fl_by_frame:
        reasons.append("pixel-forensics flags by frame: " + "; ".join(f"{n}: {', '.join(v)}" for n, v in fl_by_frame.items()))
    if conflicts:
        status = _bump(status, "review")
        reasons += conflicts

    # ---- liveness: positive physical evidence only ----------------------------------------------
    positive = list(d.positive)
    basis: list[str] = []
    retro_v = _verdict(active.get("retroreflection"))
    if retro_v == "retroreflective" and hsrp_expected:
        basis.append("retroreflection_passed")
    if non_planar:
        # supporting evidence ONLY: a screen held in front of a real background also shows depth
        positive.append("parallax_non_planar")
    ctx_score = float((screen_context or {}).get("score") or 0.0)
    medium_hard = [f for f in prov.hard if f.code not in _HARD_NOT_A_SPOOF_CLAIM]
    if d.pa_score >= REVIEW_AT or retro_v in SPOOF_VERDICTS or ctx_score >= SCREEN_SPOOF_AT or medium_hard:
        liveness = "suspected_spoof"
    elif basis and status != "reject":
        liveness = "confirmed"
    else:
        liveness = "not_established"

    if liveness == "not_established":
        missing = []
        if "retroreflection" in not_run_why:
            missing.append(f"retroreflection not run ({not_run_why['retroreflection']})")
        if "planarity" in not_run_why:
            missing.append(f"parallax not run ({not_run_why['planarity']})")
        if status == "no_flags":
            reasons.append("no flags were raised, but there is NO positive evidence of a live scene: that is not liveness. "
                           + ("; ".join(missing) if missing else ""))
    if non_planar and liveness != "suspected_spoof":
        reasons.append("the angle frames show depth, but depth alone does not establish liveness (a screen held in front of a real "
                       "background also shows depth); only the flash retroreflection test can confirm it")
    if recapture and max(recapture.values()) < 0.5:
        reasons.append("the learned screen / print model scored this bundle clean, but a clean score is never used as proof of a live "
                       "scene: it was trained on simulated attacks only and has scored a real monitor photo as genuine")
    if require_liveness and liveness != "confirmed":
        status = _bump(status, "review")
        reasons.append(f"liveness required but {liveness.replace('_', ' ')}")

    evidence = dict(d.evidence)
    evidence.update({
        "liveness_basis": basis,
        "recapture_by_frame": {n: round(float(s), 3) for n, s in recapture.items()},
        "planarity_by_frame": {n: r.as_dict() for n, r in plan_results},
        "read_conflicts": conflicts,
        "not_run_reasons": not_run_why,
        "primary_frame": names[p_idx],
        "hsrp_expected": bool(hsrp_expected),
        "screen_context": screen_context,
        "calibrated": False,
    })
    return BundleDecision(status=status, liveness=liveness, assurance=d.assurance, reasons=_dedupe(reasons),
                          not_run=list(d.not_run), positive=_dedupe(positive), pa_score=d.pa_score,
                          per_frame=per_frame, evidence=evidence)
