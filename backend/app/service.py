"""One capture (1-6 photos, each with a role) -> readings + authenticity / liveness verdict.

Roles say what a photo is for, because that decides which checks are meaningful:

  plate        the number plate, normal exposure (flash off). The first one is the primary frame.
  plate_flash  the same pose with the flash ON - used only for the retroreflection (liveness) test.
  angle        the same plate from another position - read and fused with the others, and used for
               the parallax / planarity test.
  vin          a VIN / chassis photo: read for a VIN only (short legacy chassis numbers allowed).
  auto         unknown: look for both a plate and a VIN. Gets the passive checks only.

Active liveness checks (flash, parallax) run only when the roles make them meaningful. Nothing
here decides anything: ``status`` / ``liveness`` say what the evidence supports and what was not
checked, and the caller applies its own policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vinplate.authenticity import Authenticator, Frame, assess_bundle
from vinplate.scene import FieldReading, SceneReader, SceneResult

ROLES = ("auto", "plate", "plate_flash", "angle", "vin")


@dataclass
class ImageIn:
    name: str
    payload: bytes
    rgb: np.ndarray
    role: str = "auto"
    taken_at: str | None = None


def jsonable(x: Any) -> Any:
    """Recursively convert numpy scalars / arrays / tuples so the response always serialises."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _better(a: FieldReading | None, b: FieldReading | None) -> FieldReading | None:
    if a is None or b is None:
        return a or b
    return a if a.rank() >= b.rank() else b


def field_out(fr: FieldReading | None, image_index: int) -> dict | None:
    """The public shape of one reading. ``value`` is only set when there is something to act on
    (accept / review); ``masked`` is what to DISPLAY unless status is accept."""
    if fr is None:
        return None
    d = fr.as_dict()
    d["image"] = image_index
    d["value"] = d["value"] if fr.status in ("accept", "review") else None
    parsed = fr.parsed or {}
    d["display"] = parsed.get("display") or d["value"]
    d["needs_review"] = fr.status != "accept"
    if fr.domain == "vin":
        d["check_digit_ok"] = parsed.get("check_digit_ok")
        d["read_length"] = len(fr.reading.text)
    return d


def _plate_box(res: SceneResult) -> tuple[int, int, int, int] | None:
    return tuple(res.plate.box) if res and res.plate else None


def analyse(images: list[ImageIn], reader: SceneReader, authenticator: Authenticator | None, *,
            hsrp_expected: bool = False, require_liveness: bool = False) -> dict:
    """Read every photo, fuse readings of the same identifier, then assess authenticity."""
    groups: dict[str, list[int]] = {r: [] for r in ROLES}
    for i, im in enumerate(images):
        groups[im.role].append(i)

    # ---- readings -------------------------------------------------------------------------
    plate_ids = groups["plate"] + groups["angle"]
    plate_res = reader.read_frames([images[i].rgb for i in plate_ids], domains=("plate",)) if plate_ids else None
    vin_res = (reader.read_frames([images[i].rgb for i in groups["vin"]], domains=("vin",), allow_legacy_vin=True)
               if groups["vin"] else None)
    auto_res = (reader.read_frames([images[i].rgb for i in groups["auto"]], domains=("plate", "vin"))
                if groups["auto"] else None)

    def locate(res: SceneResult | None, ids: list[int], fr: FieldReading | None) -> tuple[FieldReading | None, int]:
        return (fr, ids[fr.image] if fr is not None and fr.image < len(ids) else -1)

    cands_plate = [locate(plate_res, plate_ids, plate_res.plate if plate_res else None),
                   locate(auto_res, groups["auto"], auto_res.plate if auto_res else None)]
    cands_vin = [locate(vin_res, groups["vin"], vin_res.vin if vin_res else None),
                 locate(auto_res, groups["auto"], auto_res.vin if auto_res else None)]
    plate, plate_img = max(cands_plate, key=lambda t: (t[0].rank() if t[0] else (-1, 0, 0)))
    vin, vin_img = max(cands_vin, key=lambda t: (t[0].rank() if t[0] else (-1, 0, 0)))

    conflicts = [c for r in (plate_res, vin_res, auto_res) if r for c in r.conflicts]
    timings = {k: round(sum((r.timings.get(k, 0.0) for r in (plate_res, vin_res, auto_res) if r)), 1)
               for k in ("detect", "recognise", "decode")}

    # per-photo plate boxes (the flash test needs the plate region of the primary frame)
    plate_boxes: dict[int, tuple[int, int, int, int]] = {}
    for res, ids in ((plate_res, plate_ids), (auto_res, groups["auto"])):
        if res is None:
            continue
        for k, single in enumerate(res.per_image or [res]):
            box = _plate_box(single)
            if box is not None and k < len(ids):
                plate_boxes[ids[k]] = box

    # ---- authenticity ----------------------------------------------------------------------
    primary_idx = groups["plate"][0] if groups["plate"] else None
    frames: list[Frame] = []
    order: list[int] = []
    for i, im in enumerate(images):
        if im.role == "plate":
            role = "primary" if i == primary_idx else "other"
        elif im.role == "plate_flash":
            role = "flash_on"
        elif im.role == "angle":
            role = "angle"
        else:
            role = "other"
        frames.append(Frame(im.rgb, role, payload=im.payload, roi=plate_boxes.get(i), name=f"{i}:{im.name}"))
        order.append(i)
    # the primary must come first so it is the reference for flash / parallax
    if primary_idx is not None:
        frames.insert(0, frames.pop(primary_idx))
        order.insert(0, order.pop(primary_idx))
    plate_reads = [(s.plate.reading.masked if s.plate else None)
                   for res, ids in ((plate_res, plate_ids),) if res for s in (res.per_image or [res])]
    # what the SCENE looks like (web-page text, many lines of UI text): the strongest cue against a photo of a
    # screen that the plate itself cannot give. Take the most screen-like photo group.
    contexts = [r.screen_context for r in (plate_res, auto_res, vin_res) if r is not None and r.screen_context]
    screen_context = max(contexts, key=lambda c: c.get("score", 0.0), default=None)
    auth = assess_bundle(frames, authenticator=authenticator, hsrp_expected=hsrp_expected,
                         require_liveness=require_liveness,
                         readings=plate_reads if len(plate_reads) > 1 else None,
                         screen_context=screen_context)

    return jsonable({
        "fields": {"plate": field_out(plate, plate_img), "vin": field_out(vin, vin_img)},
        "authenticity": auth.as_dict(),
        "conflicts": conflicts,
        "images": [{"index": i, "name": im.name, "role": im.role, "taken_at": im.taken_at,
                    "size": [int(im.rgb.shape[1]), int(im.rgb.shape[0])]} for i, im in enumerate(images)],
        "lines": [{"image": ids[k] if k < len(ids) else -1, **ln}
                  for res, ids in ((plate_res, plate_ids), (vin_res, groups["vin"]), (auto_res, groups["auto"])) if res
                  for k, single in enumerate(res.per_image or [res]) for ln in single.lines],
        "screen_context": screen_context or {"score": 0.0, "url": False, "ui_words": [], "text_lines": 0, "reasons": []},
        "timings_ms": timings,
    })
