"""Golden data for the on-device (Dart) authenticity port: app/test/fixtures/golden/liveness_*.json.

Three files, each produced by the REAL Python reference so the Dart tests compare against it:

  liveness_planarity.json  gray frame pairs (flat replay / layered scene / too little motion) and the
                           Python `planarity()` verdict, tile count, baseline and inlier fraction;
  liveness_retro.json      toy-physics flash pairs (real HSRP / screen / print / hand-shake / no
                           headroom / glare patch / non-uniform gain ...) and the Python `retroreflection()`
                           verdict, gains, gain_spread and hotspot flag;
  liveness_cases.json      ~20 scripted bundles. `assess_bundle` is run for real with the expensive
                           checks (flash test, planarity, pixel forensics, provenance, PadNet) replaced
                           by CANNED results, so what is compared is the DECISION LOGIC only: status,
                           liveness, pa_score, positive, not_run, basis, reasons, screen_context. A case may
                           carry the scene reader's `screen_context` (interface-text evidence, key
                           "screen_context"); it is passed to `assess_bundle(..., screen_context=)`.

Run from ml/:   .venv\\Scripts\\python.exe -I -X utf8 scripts/make_golden_liveness.py
Frames are stored as zlib-compressed gray uint8 (base64). Deterministic: re-running reproduces the files.
"""

from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
OUT = ML.parent / "app" / "test" / "fixtures" / "golden"

from tests.test_active import PLATE, W as TOY_W, _with_glare_patch, scenes, textured, to_u8, warp   # noqa: E402  (toy physics, shared with the Python tests)
from vinplate.authenticity import Authenticator, Frame, assess_bundle        # noqa: E402
from vinplate.authenticity import bundle as B                                # noqa: E402
from vinplate.authenticity.active import PlanarityResult, RetroResult, planarity, retroreflection  # noqa: E402
from vinplate.authenticity.forensics import ForensicsReport                  # noqa: E402
from vinplate.authenticity.provenance import Finding, ProvenanceReport       # noqa: E402


def pack(gray: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(gray, np.uint8).tobytes(), 9)).decode()


def rgb3(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, -1).astype(np.uint8)


def num(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------------------ planarity
def planarity_golden() -> dict:
    big = to_u8(textured(480, 640, 11)).astype(np.float32)[..., 0].astype(np.uint8)
    flat = warp(big, (7, 3), 1.0).astype(np.uint8)
    bg, fg = warp(big, (3, 1), 0.3), warp(big, (11, 4), 0.3)
    m = np.zeros(big.shape, bool)
    m[100:380, 140:520] = True
    layered = np.where(m, fg, bg).astype(np.uint8)
    tiny = warp(big, (0.3, 0.1), 0.0).astype(np.uint8)
    blank = np.full_like(big, 128)
    pairs = {"flat": flat, "layered": layered, "tiny": tiny}
    out = {"h": 480, "w": 640, "a": pack(big), "pairs": {}}
    for name, b in pairs.items():
        r = planarity(rgb3(big), rgb3(b))
        out["pairs"][name] = {"b": pack(b), "expect": {"verdict": r.verdict, "tiles": r.tiles, "baseline_px": num(r.baseline_px),
                                                       "inlier_frac": num(r.inlier_frac), "residual_px": num(r.residual_px)}}
    r = planarity(rgb3(blank), rgb3(blank))
    out["blank"] = {"expect": {"verdict": r.verdict, "tiles": r.tiles}}
    return out


# ---------------------------------------------------------------------------------- retro
def retro_golden() -> dict:
    H, W = 240, 400
    cases = {}

    def add(name, off, on, plate=PLATE, hsrp=True):
        r = retroreflection(rgb3(off) if off.ndim == 2 else off, rgb3(on) if on.ndim == 2 else on, plate, hsrp_expected=hsrp)
        ys, xs = np.nonzero(plate)
        cases[name] = {
            "off": pack(off if off.ndim == 2 else off[..., 0]), "on": pack(on if on.ndim == 2 else on[..., 0]),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "hsrp": hsrp,
            "expect": {"verdict": r.verdict, "ratio": num(r.ratio), "gain_plate": num(r.gain_plate), "gain_surround": num(r.gain_surround),
                       "gain_spread": num(r.gain_spread), "hotspot": bool(r.hotspot), "reasons": list(r.reasons)}}

    for kind in ("real_hsrp", "screen", "print"):
        off, on = scenes(kind)
        add(kind, off[..., 0], on[..., 0])
    off, on = scenes("real_hsrp", 3)
    shaken = np.roll(np.roll(on, 3, axis=0), -4, axis=1)
    add("hsrp_handshake_dim", off[..., 0], (shaken.astype(np.float32) * 0.85).astype(np.uint8)[..., 0])
    off, on = scenes("screen", 4)
    add("screen_handshake", off[..., 0], np.roll(on, 3, axis=1)[..., 0])
    off, on = scenes("screen")
    add("not_hsrp", off[..., 0], on[..., 0], hsrp=False)
    add("flash_did_not_fire", off[..., 0], off[..., 0].copy())
    sat = np.full((H, W), 250, np.uint8)
    add("saturated", sat, sat)
    add("no_headroom", np.full((H, W), 170, np.uint8), np.full((H, W), 200, np.uint8))
    # --- flash uniformity: a screen whose flash reflection is a glare PATCH on part of the plate is a hot-spot
    for frac in (0.15, 0.25, 0.35):
        off, on = _with_glare_patch(frac)
        add(f"screen_glare_patch_{int(frac * 100)}", off[..., 0], on[..., 0])
    off, on = _with_glare_patch(0.25)
    add("glare_patch_not_hsrp", off[..., 0], on[..., 0], hsrp=False)
    # a strong plate-level gain that is NOT uniform (left 30% barely brightens): never a pass
    off, on = scenes("real_hsrp")
    lin = (on[..., 0].astype(np.float32) / 255.0) ** 2.2
    lo = (off[..., 0].astype(np.float32) / 255.0) ** 2.2
    boost = np.where(np.arange(TOY_W)[None, :] < 160, 0.05, 1.0)
    lin2 = np.where(PLATE, lo * 1.25 + (lin - lo * 1.25) * boost, lin)
    add("hsrp_nonuniform_gain", off[..., 0], to_u8(np.clip(lin2, 0, 1))[..., 0])
    return {"h": H, "w": W, "cases": cases}


# ----------------------------------------------------------------------- decision cases
HARD_SCREEN = {"code": "xmp_screencapture", "severity": "hard", "detail": "XMP DigitalSourceType = screencapture"}
SOFT_EDITOR = {"code": "editor_marker", "severity": "soft", "detail": "metadata mentions an image editor (photoshop)"}

RETRO_OK = {"verdict": "retroreflective", "ratio": 4.2, "gain_plate": 5.0, "gain_surround": 1.19, "reasons": [], "gain_spread": 1.3, "hotspot": False}
RETRO_BAD = {"verdict": "not_retroreflective", "ratio": 1.0, "gain_plate": 1.1, "gain_surround": 1.1,
             "reasons": ["plate brightened 1.10x under flash vs 1.10x for its surround (a screen emits its own light; a print reflects like its surroundings)"],
             "gain_spread": None, "hotspot": False}
RETRO_NONE = {"verdict": "inconclusive", "ratio": None, "gain_plate": None, "gain_surround": None, "reasons": ["plate already saturated without flash (strong ambient light): not measurable"],
              "gain_spread": None, "hotspot": False}


def _real_hotspot() -> dict:
    """The canned 'specular_hotspot' result is the REAL Python result for the glare-patch screen scene (text and all)."""
    off, on = _with_glare_patch(0.25)
    return retroreflection(off, on, PLATE, hsrp_expected=True).as_dict()


RETRO_HOTSPOT = _real_hotspot()
ATTACK_CONTEXT = {"score": 3.0, "url": True, "ui_words": ["DETAILS", "IMAGES", "RELATED", "SEARCH"], "text_lines": 31,
                  "reasons": ["a web address (URL) is visible in the scene", "browser / app interface words are visible: DETAILS, IMAGES, RELATED, SEARCH",
                              "31 separate lines of text are visible, far more than a vehicle scene normally has"]}
URL_CONTEXT = {"score": 1.0, "url": True, "ui_words": [], "text_lines": 3, "reasons": ["a web address (URL) is visible in the scene"]}
WORDS_CONTEXT = {"score": 1.5, "url": False, "ui_words": ["DOWNLOAD", "SEARCH", "SETTINGS"], "text_lines": 6,
                 "reasons": ["browser / app interface words are visible: DOWNLOAD, SEARCH, SETTINGS"]}
LINES_CONTEXT = {"score": 0.5, "url": False, "ui_words": [], "text_lines": 15, "reasons": ["15 separate lines of text are visible, far more than a vehicle scene normally has"]}
NO_CONTEXT = {"score": 0.0, "url": False, "ui_words": [], "text_lines": 0, "reasons": []}
PLAN_NP = {"verdict": "non_planar", "residual_px": 2.1, "baseline_px": 6.0, "tiles": 80, "inlier_frac": 0.4, "reasons": []}
PLAN_FLAT = {"verdict": "planar", "residual_px": 0.2, "baseline_px": 4.0, "tiles": 80, "inlier_frac": 0.97, "reasons": ["97% of tiles move as ONE plane (motion 4.0px). Consistent with a screen/print - but also with a flat, shallow scene or a pure camera rotation."]}
PLAN_LOW = {"verdict": "inconclusive", "residual_px": None, "baseline_px": 0.6, "tiles": 70, "inlier_frac": None, "reasons": ["camera moved only 0.60px at 512px width: not enough baseline"]}
PLAN_NP_TINY = {"verdict": "non_planar", "residual_px": 0.5, "baseline_px": 1.0, "tiles": 70, "inlier_frac": 0.5, "reasons": []}


def fr(role, name, rec=None, prov=None, roi=True, size=(200, 200)):
    return {"role": role, "name": name, "recapture": rec, "provenance": prov, "roi": roi, "size": list(size)}


def bundle3(rec=None, prov=None):
    r = rec if isinstance(rec, list) else [rec] * 3
    return [fr("primary", "front", r[0], prov), fr("flash_on", "front_flash", r[1]), fr("angle", "left", r[2])]


CASES = [
    dict(name="genuine_hsrp_bundle_no_model", frames=bundle3(), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True),
    dict(name="genuine_hsrp_bundle_clean_model", frames=bundle3(0.05), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True),
    dict(name="screen_replay_flash_and_flat", frames=bundle3(), retro=RETRO_BAD, planarity=[PLAN_FLAT], hsrp=True),
    dict(name="screen_replay_high_model_score", frames=bundle3(0.97), retro=RETRO_BAD, planarity=[PLAN_FLAT], hsrp=True),
    dict(name="print_replay_flash_only", frames=bundle3()[:2], retro=RETRO_BAD, planarity=[], hsrp=True),
    dict(name="parallax_alone_no_model", frames=[fr("primary", "front"), fr("angle", "left")], retro=None, planarity=[PLAN_NP], hsrp=False),
    dict(name="parallax_with_clean_recapture", frames=[fr("primary", "front", 0.05), fr("angle", "left", 0.05)], retro=None, planarity=[PLAN_NP], hsrp=False),
    dict(name="parallax_with_suspicious_recapture", frames=[fr("primary", "front", 0.6), fr("angle", "left", 0.6)], retro=None, planarity=[PLAN_NP], hsrp=False),
    dict(name="parallax_clean_but_flash_fails", frames=bundle3(0.05), retro=RETRO_BAD, planarity=[PLAN_NP], hsrp=True),
    dict(name="single_image_no_flags", frames=[fr("primary", "front")], retro=None, planarity=[], hsrp=False),
    dict(name="single_image_require_liveness", frames=[fr("primary", "front")], retro=None, planarity=[], hsrp=False, require=True),
    dict(name="conflicting_reads", frames=[fr("primary", "front"), fr("other", "second")], retro=None, planarity=[], hsrp=False,
         readings=["MH12AB1234", "KA01XY9876"]),
    dict(name="reads_with_unknown_are_compatible", frames=[fr("primary", "front"), fr("other", "second")], retro=None, planarity=[], hsrp=False,
         readings=["MH12?B1234", "MH12AB1234"]),
    dict(name="reads_conflict_with_dropped_char", frames=[fr("primary", "front"), fr("other", "second")], retro=None, planarity=[], hsrp=False,
         readings=["MH12AB1234", "MH12AB123"]),
    dict(name="hard_provenance_flag", frames=[fr("primary", "front", None, [HARD_SCREEN])], retro=None, planarity=[], hsrp=False),
    dict(name="soft_provenance_flag", frames=[fr("primary", "front", None, [SOFT_EDITOR])], retro=None, planarity=[], hsrp=False),
    dict(name="same_hard_flag_in_two_frames", frames=[fr("primary", "front", None, [HARD_SCREEN]), fr("other", "second", None, [HARD_SCREEN])], retro=None, planarity=[], hsrp=False),
    dict(name="flash_inconclusive_saturated", frames=bundle3()[:2], retro=RETRO_NONE, planarity=[], hsrp=True),
    dict(name="flash_frame_without_roi", frames=[fr("primary", "front", roi=False), fr("flash_on", "front_flash", roi=False)], retro=None, planarity=[], hsrp=True,
         retro_skip="no plate region (ROI) for the primary frame, so no plate mask can be built"),
    dict(name="two_flat_angles_count_once", frames=[fr("primary", "front"), fr("angle", "a"), fr("angle", "b")], retro=None, planarity=[PLAN_FLAT, PLAN_FLAT], hsrp=False),
    dict(name="mixed_planarity_non_planar_wins", frames=[fr("primary", "front"), fr("angle", "a"), fr("angle", "b")], retro=None, planarity=[PLAN_FLAT, PLAN_NP], hsrp=False),
    dict(name="non_planar_below_baseline_is_ignored", frames=[fr("primary", "front"), fr("angle", "a")], retro=None, planarity=[PLAN_NP_TINY], hsrp=False),
    dict(name="angle_without_enough_motion", frames=[fr("primary", "front"), fr("angle", "a")], retro=None, planarity=[PLAN_LOW], hsrp=False),
    dict(name="angle_frame_aspect_mismatch", frames=[fr("primary", "front"), fr("angle", "wide", size=(100, 200))], retro=None, planarity=[], hsrp=False, skipped=["wide"]),
    dict(name="two_flash_frames", frames=[fr("primary", "front"), fr("flash_on", "f1"), fr("flash_on", "f2")], retro=RETRO_OK, planarity=[], hsrp=True),
    dict(name="one_frame_mildly_suspicious_model", frames=bundle3([0.1, 0.55, 0.1]), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True),
    dict(name="clean_model_no_flash_no_angle", frames=[fr("primary", "front", 0.05)], retro=None, planarity=[], hsrp=True),
    # --- the real attack (photo of a monitor): glare patch under flash, depth from the office behind it, PadNet 'clean',
    # --- browser text in the scene. Liveness is NEVER confirmed by parallax or by a low PadNet score.
    dict(name="real_attack_screen_photo", frames=bundle3(0.003), retro=RETRO_HOTSPOT, planarity=[PLAN_NP], hsrp=True, context=ATTACK_CONTEXT),
    dict(name="real_attack_four_frames", frames=[fr("primary", "front", 0.003), fr("flash_on", "front_flash", 0.003), fr("angle", "left", 0.003), fr("angle", "right", 0.003)],
         retro=RETRO_HOTSPOT, planarity=[PLAN_NP, PLAN_NP], hsrp=True, context=ATTACK_CONTEXT),
    dict(name="parallax_clean_padnet_and_screen_context", frames=[fr("primary", "front", 0.003), fr("angle", "left", 0.003)], retro=None, planarity=[PLAN_NP], hsrp=False,
         context=ATTACK_CONTEXT),
    dict(name="hotspot_only_is_review", frames=bundle3()[:2], retro=RETRO_HOTSPOT, planarity=[], hsrp=True),
    dict(name="hotspot_with_depth_and_clean_model", frames=bundle3(0.05), retro=RETRO_HOTSPOT, planarity=[PLAN_NP], hsrp=True),
    dict(name="uniform_retro_with_screen_context_is_still_a_spoof", frames=bundle3(), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True, context=ATTACK_CONTEXT),
    dict(name="uniform_retro_with_weak_lines_context_confirms", frames=bundle3(), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True, context=LINES_CONTEXT),
    dict(name="uniform_retro_with_no_context_confirms", frames=bundle3(), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True, context=NO_CONTEXT),
    dict(name="single_url_is_review", frames=bundle3(), retro=RETRO_OK, planarity=[PLAN_NP], hsrp=True, context=URL_CONTEXT),
    dict(name="interface_words_alone_review", frames=[fr("primary", "front")], retro=None, planarity=[], hsrp=False, context=WORDS_CONTEXT),
    dict(name="screen_context_on_a_single_image", frames=[fr("primary", "front")], retro=None, planarity=[], hsrp=False, context=ATTACK_CONTEXT),
    dict(name="padnet_clean_alone_never_confirms", frames=[fr("primary", "front", 0.003)], retro=None, planarity=[], hsrp=False),
    dict(name="padnet_clean_and_depth_never_confirms", frames=[fr("primary", "front", 0.003), fr("angle", "left", 0.003)], retro=None, planarity=[PLAN_NP], hsrp=False),
]


def run_case(c: dict) -> dict:
    retro = c.get("retro")
    plans = iter(c.get("planarity", []))
    recs = [f["recapture"] for f in c["frames"]]
    B.retroreflection = lambda *a, **k: RetroResult(**retro)
    B.planarity = lambda a, b, **k: PlanarityResult(**next(plans))
    B.analyze_pixels = lambda rgb, was_jpeg=False: ForensicsReport()
    B.analyze_provenance = lambda payload: ProvenanceReport(findings=[Finding(**f) for f in json.loads(payload)])
    scores = iter([r for r in recs if r is not None])
    auth = Authenticator(pad_model=(lambda patches: next(scores))) if any(r is not None for r in recs) else Authenticator()
    frames = []
    for f in c["frames"]:
        h, w = f["size"][1], f["size"][0]
        frames.append(Frame(np.zeros((h, w, 3), np.uint8), f["role"], payload=(json.dumps(f["provenance"]).encode() if f["provenance"] is not None else None),
                            roi=((20, 20, w - 20, h - 20) if f["roi"] else None), name=f["name"]))
    d = assess_bundle(frames, authenticator=auth, hsrp_expected=c.get("hsrp", False), require_liveness=c.get("require", False), readings=c.get("readings"),
                      screen_context=c.get("context"))
    return {"status": d.status, "liveness": d.liveness, "pa_score": d.pa_score, "positive": list(d.positive), "not_run": list(d.not_run),
            "basis": d.evidence["liveness_basis"], "reasons": list(d.reasons), "not_run_reasons": d.evidence["not_run_reasons"],
            "read_conflicts": d.evidence["read_conflicts"], "screen_context": d.evidence.get("screen_context")}


def cases_golden() -> dict:
    out = []
    for c in CASES:
        # the recorded retro result is only fed to Python when a flash frame exists and a plate region does
        exp = run_case(c)
        out.append({"name": c["name"], "frames": [{k: f[k] for k in ("role", "name", "recapture", "provenance")} for f in c["frames"]],
                    "retro": c.get("retro"), "retro_skip_reason": c.get("retro_skip"), "planarity": c.get("planarity", []),
                    "skipped_angle_frames": c.get("skipped", []), "hsrp": c.get("hsrp", False), "require_liveness": c.get("require", False),
                    "readings": c.get("readings"), "screen_context": c.get("context"), "expect": exp})
    return {"cases": out}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("liveness_planarity", planarity_golden()), ("liveness_retro", retro_golden()), ("liveness_cases", cases_golden())):
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KB")
    for n, v in json.loads((OUT / "liveness_planarity.json").read_text())["pairs"].items():
        print("planarity", n, v["expect"])


if __name__ == "__main__":
    main()
