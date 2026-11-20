"""Audit installed Python packages and the non-pip asset manifest against licences/policy.json.

    python scripts/license_audit.py [--strict]

Exit code 1 if any component is denied (or, with --strict, needs review). The point is to make
"commercially viable" a checked property of the build, not a belief. This reads PACKAGE METADATA
(License / License-Expression / classifiers), which authors fill in imperfectly: 'unknown' means
"go and read the licence file", not "fine". It is not legal advice.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "licenses"

# (regex on normalised licence text, canonical id, class)
RULES = [
    (r"\bagpl|affero", "AGPL", "deny"),
    (r"\blgpl|lesser general public", "LGPL", "review"),
    (r"\bgpl\b|gnu general public", "GPL", "deny"),
    (r"non-?commercial|\bcc-by-nc|\bnc-nd|\bnc-sa|research use only|research-only", "non-commercial", "deny"),
    (r"\bsspl|commons clause", "SSPL", "deny"),
    (r"mpl|mozilla public", "MPL-2.0", "review"),
    (r"apache", "Apache-2.0", "allow"),
    (r"\bmit-cmu|hpnd|historical permission|pil software", "HPND", "allow"),
    (r"\bmit\b|mit license|expat", "MIT", "allow"),
    (r"\bbsd|berkeley", "BSD", "allow"),
    (r"\bisc\b", "ISC", "allow"),
    (r"\bpsf|python software foundation|python-2\.0", "PSF", "allow"),
    (r"zlib", "Zlib", "allow"),
    (r"unlicense|public domain|\bcc0", "Unlicense", "allow"),
    (r"bsl-1\.0|boost software", "BSL-1.0", "allow"),
]
# only if nothing permissive matched: "All rights reserved" appears in plenty of BSD/MIT texts
FALLBACK = [(r"proprietary|qualcomm|all rights reserved", "proprietary", "review")]


def classify(text: str, policy: dict) -> tuple[str, str]:
    t = re.sub(r"\s+", " ", text.lower())
    # dual licences ("MIT OR Apache-2.0"): the *most permissive* alternative governs
    hits = [(c, k) for pat, c, k in RULES if re.search(pat, t)]
    if not hits:
        hits = [(c, k) for pat, c, k in FALLBACK if re.search(pat, t)]
    if not hits:
        return "unknown", "review"
    if " or " in t or " / " in t:
        ok = [h for h in hits if h[1] == "allow"]
        if ok:
            hits = ok
    order = {"deny": 0, "review": 1, "allow": 2}
    c, k = min(hits, key=lambda h: order[h[1]])
    # policy overrides: an id listed under a different class moves the verdict
    for cls in ("deny", "review", "allow"):
        if c in policy.get(cls, []) or any(c.startswith(x) for x in policy.get(cls, [])):
            k = cls
            break
    return c, k


def licence_text(dist) -> str:
    md = dist.metadata
    cls = [c.split("::")[-1] for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]
    # structured fields first; the free-text License field is often a pasted licence body
    parts = [md.get("License-Expression") or ""] + cls
    if not any(p.strip() for p in parts):
        parts = [md.get("License") or ""]
    s = " ".join(p for p in parts if p and p.strip().lower() != "unknown")
    return s[:400] if len(s) > 400 else s  # some packages paste the whole licence text here


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    policy = json.loads((ROOT / "policy.json").read_text())
    assets = json.loads((ROOT / "assets.json").read_text())

    rows, bad, rev = [], 0, 0
    for d in sorted(metadata.distributions(), key=lambda x: (x.metadata["Name"] or "").lower()):
        name = d.metadata["Name"]
        c, k = classify(licence_text(d), policy)
        rows.append((name, d.version, c, k))
    print(f"{'package':32s} {'version':14s} {'licence':16s} verdict")
    for name, ver, c, k in rows:
        print(f"{name[:31]:32s} {ver[:13]:14s} {c[:15]:16s} {k.upper() if k != 'allow' else 'ok'}")
        bad += k == "deny"
        rev += k == "review"

    print("\n-- non-pip assets (licenses/assets.json) --")
    for sect in ("runtimes", "code", "datasets", "fonts"):
        for it in assets.get(sect, []):
            k = it.get("verdict") or ("review" if it["license"] == "proprietary" else "allow")
            print(f"[{sect[:-1]:8s}] {it['name'][:58]:58s} {it['license'][:30]:30s} {k.upper() if k != 'allow' else 'ok'} (basis {it['basis']})")
            bad += k == "deny" and sect in ("runtimes", "fonts")   # denied code/data are *listed to avoid*, not in the build
            rev += k == "review"
    print(f"\n{bad} denied in the build, {rev} need review. Basis: V verified, A agent-reported, R recalled/UNVERIFIED.")
    return 1 if bad or (a.strict and rev) else 0


if __name__ == "__main__":
    sys.exit(main())
