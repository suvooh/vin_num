"""Provenance and metadata evidence for an uploaded image. Deterministic, no model.

What each signal is worth (so callers do not over-read it):

* A *positive* AI/screen declaration (IPTC DigitalSourceType, a C2PA "created by AI" action,
  a Stable-Diffusion/ComfyUI PNG chunk) is strong evidence - the generator wrote it. Absence
  of such a marker means NOTHING: anyone can strip metadata, screenshots and messaging apps
  do it automatically, and most Indian phones sign no C2PA at all.
* C2PA is only trust when it VERIFIES (signature + hashes); a manifest that is merely present
  is reported as ``unverified``. A manifest that fails validation is a hard flag.
* "no EXIF" is information, not a flag: WhatsApp and screenshots strip it.
* Editor names in EXIF/XMP are soft flags (a legitimate crop shows up the same way).
* An EXIF thumbnail that does not match the image means the pixels were changed after the
  camera wrote the file (many editors keep the stale thumbnail) - a real tamper cue.

Not available, and not faked: Google SynthID. Its image detector is a portal only (no API or
offline model) so it can only be a manual escalation step.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

HARD, SOFT, INFO = "hard", "soft", "info"

_AI_TOOLS = ("stable diffusion", "midjourney", "dall-e", "dall·e", "firefly", "imagen", "gemini",
             "flux", "comfyui", "automatic1111", "novelai", "invokeai", "leonardo.ai", "ideogram",
             "sdxl", "openai", "bing image creator", "dreamstudio", "nano banana")
_EDITORS = ("photoshop", "gimp", "lightroom", "snapseed", "pixlr", "paint.net", "canva", "picsart",
            "affinity", "photopea", "facetune", "meitu", "capture one", "luminar", "fotor", "pixelmator")
_DST_HARD = {"trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia", "algorithmicmedia",
             "compositesynthetic", "datadrivenmedia", "screencapture", "print"}
_DST_SOFT = {"composite", "algorithmicallyenhanced", "humanedits", "digitalcreation", "virtualrecording"}
_PNG_AI_KEYS = {"parameters": "Automatic1111/SD-WebUI", "prompt": "ComfyUI", "workflow": "ComfyUI",
                "invokeai_metadata": "InvokeAI", "sd-metadata": "InvokeAI", "dream": "Dream",
                "negative_prompt": "Stable Diffusion"}


@dataclass
class Finding:
    code: str
    severity: str          # hard | soft | info
    detail: str


@dataclass
class ProvenanceReport:
    findings: list[Finding] = field(default_factory=list)
    c2pa: dict = field(default_factory=lambda: {"present": False, "verified": None})
    exif: dict = field(default_factory=dict)
    sha256: str = ""
    dhash: str = ""

    def add(self, code, sev, detail):
        self.findings.append(Finding(code, sev, detail))

    @property
    def hard(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == HARD]

    @property
    def soft(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SOFT]

    def as_dict(self) -> dict:
        return {"findings": [f.__dict__ for f in self.findings], "c2pa": self.c2pa, "exif": self.exif,
                "sha256": self.sha256, "dhash": self.dhash}


# ------------------------------------------------------------------ helpers
def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash: near-duplicate / replay detection against a store of past captures."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def exif_thumbnail(jpeg: bytes) -> bytes | None:
    """Embedded JPEG thumbnail from IFD1 of the Exif APP1 segment, if any."""
    if not jpeg.startswith(b"\xff\xd8"):
        return None
    i = 2
    while i + 4 <= len(jpeg) and jpeg[i] == 0xFF:
        marker, ln = jpeg[i + 1], struct.unpack(">H", jpeg[i + 2 : i + 4])[0]
        seg = jpeg[i + 4 : i + 2 + ln]
        if marker == 0xE1 and seg.startswith(b"Exif\x00\x00"):
            tiff = seg[6:]
            end = "<" if tiff[:2] == b"II" else ">"
            try:
                ifd0 = struct.unpack(end + "I", tiff[4:8])[0]
                n = struct.unpack(end + "H", tiff[ifd0 : ifd0 + 2])[0]
                nxt = struct.unpack(end + "I", tiff[ifd0 + 2 + 12 * n : ifd0 + 6 + 12 * n])[0]
                if not nxt:
                    return None
                n1 = struct.unpack(end + "H", tiff[nxt : nxt + 2])[0]
                off = length = None
                for k in range(n1):
                    tag, _t, _c, val = struct.unpack(end + "HHII", tiff[nxt + 2 + 12 * k : nxt + 14 + 12 * k])
                    if tag == 0x0201:
                        off = val
                    elif tag == 0x0202:
                        length = val
                if off is not None and length:
                    return tiff[off : off + length]
            except struct.error:
                return None
        if marker == 0xDA:
            break
        i += 2 + ln
    return None


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-6 else 0.0


def dst_tokens(text: str) -> set[str]:
    """IPTC DigitalSourceType values present in ``text`` (XMP or C2PA JSON), lower-cased.
    Matches the canonical URL form ``.../digitalsourcetype/trainedAlgorithmicMedia`` and the
    bare element/attribute form. Requires the 'digitalsourcetype' key to be present at all, so
    an unrelated word such as 'print' elsewhere in the metadata cannot trigger it."""
    low = text.lower()
    if "digitalsourcetype" not in low:
        return set()
    found = set(re.findall(r"digitalsourcetype/([a-z]+)", low))
    for t in _DST_HARD | _DST_SOFT:
        if re.search(rf">\s*{t}\s*<", low) or re.search(rf"=\s*[\"']{t}[\"']", low):
            found.add(t)
    return found


def _xmp_packets(blob: bytes) -> list[str]:
    out, pos = [], 0
    while True:
        a = blob.find(b"<x:xmpmeta", pos)
        if a < 0:
            return out
        b = blob.find(b"</x:xmpmeta>", a)
        if b < 0:
            return out
        out.append(blob[a : b + 12].decode("utf-8", "replace"))
        pos = b + 12


def _c2pa(payload: bytes, fmt: str, rep: ProvenanceReport) -> None:
    present = b"c2pa" in payload and b"jumb" in payload
    rep.c2pa = {"present": present, "verified": None}
    if not present:
        return
    try:
        import c2pa  # optional: pip install c2pa-python==0.38.0 (0.38.1/0.38.2 ship no Windows wheels)
    except Exception:
        rep.c2pa["note"] = "manifest present but NOT verified: c2pa-python not installed"
        rep.add("c2pa_unverified", INFO, rep.c2pa["note"])
        return
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(fmt, "image/jpeg")
    try:
        data = json.loads(c2pa.Reader(mime, io.BytesIO(payload)).json())
    except Exception as exc:
        rep.c2pa.update(verified=False, error=str(exc)[:200])
        rep.add("c2pa_unreadable", HARD, f"C2PA data present but could not be parsed/validated: {str(exc)[:120]}")
        return
    state = str(data.get("validation_state", "")).lower()
    errors = [s for s in data.get("validation_status", []) or [] if "fail" in json.dumps(s).lower() or "mismatch" in json.dumps(s).lower()]
    active = (data.get("manifests") or {}).get(data.get("active_manifest"), {})
    rep.c2pa.update(verified=(state in ("valid", "trusted") and not errors), validation_state=state,
                    generator=active.get("claim_generator") or active.get("claim_generator_info"),
                    title=active.get("title"))
    if errors or state == "invalid":
        rep.add("c2pa_invalid", HARD, f"C2PA validation failed: {json.dumps(errors)[:200]}")
    for dst in sorted(dst_tokens(json.dumps(active)) & _DST_HARD):
        rep.add(f"c2pa_{dst}", HARD, f"C2PA actions declare digital source type '{dst}'")
    if state == "valid":
        rep.add("c2pa_signed_unknown_issuer", INFO, "valid C2PA signature but issuer is not on a trust list")
    if state == "trusted":
        rep.add("c2pa_trusted", INFO, "valid C2PA manifest from a trusted issuer")


# ------------------------------------------------------------------ main entry
def analyze_provenance(payload: bytes) -> ProvenanceReport:
    rep = ProvenanceReport(sha256=hashlib.sha256(payload).hexdigest())
    try:
        img = Image.open(io.BytesIO(payload))
        img.load()
    except Exception as exc:
        rep.add("undecodable", HARD, f"image could not be decoded: {exc}")
        return rep
    fmt = (img.format or "").upper()
    rep.dhash = dhash(img)

    # ---- EXIF
    ex = img.getexif()
    sub = {}
    try:
        from PIL import ExifTags
        sub = ex.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        pass
    get = lambda tag: ex.get(tag) or sub.get(tag)  # noqa: E731
    rep.exif = {k: str(v)[:80] for k, v in {
        "make": get(0x010F), "model": get(0x0110), "software": get(0x0131),
        "datetime_original": get(0x9003), "datetime_modified": get(0x0132),
    }.items() if v}
    if not rep.exif.get("make") and not rep.exif.get("model"):
        rep.add("no_camera_exif", INFO, "no camera make/model in EXIF (stripped by a messenger/screenshot, or not a camera file)")
    sw = (rep.exif.get("software") or "").lower()
    if any(t in sw for t in _AI_TOOLS):
        rep.add("ai_software_exif", HARD, f"EXIF Software names an AI tool: {rep.exif['software']}")
    elif any(e in sw for e in _EDITORS):
        rep.add("editor_software_exif", SOFT, f"EXIF Software names an image editor: {rep.exif['software']}")
    dm, do = rep.exif.get("datetime_modified"), rep.exif.get("datetime_original")
    if dm and do and dm > do and sw:
        rep.add("modified_after_capture", SOFT, f"modified {dm} after capture {do} by '{rep.exif.get('software')}'")

    # ---- XMP / IPTC / PNG text
    for xmp in _xmp_packets(payload):
        low = xmp.lower()
        for v in sorted(dst_tokens(xmp)):
            if v in _DST_HARD:
                rep.add(f"xmp_{v}", HARD, f"XMP DigitalSourceType = {v}")
            elif v in _DST_SOFT:
                rep.add(f"xmp_{v}", SOFT, f"XMP DigitalSourceType = {v}")
        if any(t in low for t in _AI_TOOLS):
            rep.add("ai_tool_xmp", HARD, "XMP mentions an AI image tool")
        elif "photoshop:history" in low or any(e in low for e in _EDITORS):
            rep.add("editor_xmp", SOFT, "XMP records an image-editor history/tool")
    if fmt == "PNG":
        for k, v in (getattr(img, "text", {}) or {}).items():
            kl = k.lower()
            if kl in _PNG_AI_KEYS:
                rep.add("ai_png_chunk", HARD, f"PNG text chunk '{k}' is written by {_PNG_AI_KEYS[kl]}")
            elif any(t in str(v).lower() for t in _AI_TOOLS) and kl in ("software", "comment", "description", "source"):
                rep.add("ai_png_chunk", HARD, f"PNG '{k}' names an AI tool")
    com = re.findall(rb"\xff\xfe..([\x20-\x7e]{4,200})", payload[:200000], flags=re.S) if fmt == "JPEG" else []
    for c in com:
        t = c.decode("latin1").lower()
        if any(a in t for a in _AI_TOOLS):
            rep.add("ai_jpeg_comment", HARD, f"JPEG comment names an AI tool: {c.decode('latin1')[:60]}")

    # ---- C2PA
    _c2pa(payload, fmt, rep)

    # ---- EXIF thumbnail vs pixels
    if fmt == "JPEG":
        th = exif_thumbnail(payload)
        if th:
            try:
                t = Image.open(io.BytesIO(th)).convert("L")
                main = img.convert("L").resize(t.size, Image.LANCZOS)
                if abs(t.size[0] / t.size[1] - img.size[0] / img.size[1]) > 0.08:
                    rep.add("thumbnail_aspect_mismatch", SOFT, "embedded EXIF thumbnail has a different aspect ratio than the image (cropped after capture?)")
                else:
                    c = _norm_corr(np.asarray(t, np.float32), np.asarray(main, np.float32))
                    rep.exif["thumbnail_correlation"] = f"{c:.3f}"
                    if c < 0.85:
                        rep.add("thumbnail_mismatch", SOFT, f"embedded EXIF thumbnail does not match the image (corr {c:.2f}): pixels changed after capture")
            except Exception:
                pass
    return rep
