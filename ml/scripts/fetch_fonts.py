"""Download OFL-licensed fonts for synthetic plate / VIN rendering.

Only SIL Open Font License families from google/fonts are fetched, each together with its
OFL.txt, so the licence travels with the files. Fonts are used to *render training images*;
no font software ships in the model.

Uses raw.githubusercontent.com with known file names (the GitHub REST API's anonymous quota
is tiny and shared). Missing files are skipped, not fatal.

    python scripts/fetch_fonts.py            # -> synth/fonts/<family>/
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "synth" / "fonts"
RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl"

# directory -> (file prefix, role, static styles to try; "VAR" = variable font file name)
FAMILIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # DIN-like / highway-sign sans: closest in spirit to the HSRP typeface
    "barlowcondensed": ("BarlowCondensed", "hsrp", ("Medium", "SemiBold", "Bold", "ExtraBold")),
    "barlowsemicondensed": ("BarlowSemiCondensed", "hsrp", ("Medium", "SemiBold", "Bold")),
    "barlow": ("Barlow", "hsrp", ("Medium", "SemiBold", "Bold")),
    "sairasemicondensed": ("SairaSemiCondensed", "hsrp", ("Medium", "SemiBold", "Bold")),
    "titilliumweb": ("TitilliumWeb", "hsrp", ("SemiBold", "Bold")),
    "rajdhani": ("Rajdhani", "hsrp", ("Medium", "SemiBold", "Bold")),
    "robotocondensed": ("RobotoCondensed", "hsrp", ("VAR[wght]",)),
    "archivonarrow": ("ArchivoNarrow", "hsrp", ("VAR[wght]",)),
    "overpass": ("Overpass", "hsrp", ("VAR[wght]",)),
    # heavy / painted / stylised: older and hand-painted plates
    "oswald": ("Oswald", "painted", ("VAR[wght]",)),
    "anton": ("Anton", "painted", ("Regular",)),
    "russoone": ("RussoOne", "painted", ("Regular",)),
    "blackopsone": ("BlackOpsOne", "painted", ("Regular",)),
    "alfaslabone": ("AlfaSlabOne", "painted", ("Regular",)),
    "bebasneue": ("BebasNeue", "painted", ("Regular",)),
    "chakrapetch": ("ChakraPetch", "painted", ("Medium", "SemiBold", "Bold")),
    "fjallaone": ("FjallaOne", "painted", ("Regular",)),
    "passionone": ("PassionOne", "painted", ("Regular", "Bold")),
    "staatliches": ("Staatliches", "painted", ("Regular",)),
    # monospace / OCR-like: printed VIN stickers and label text
    "sharetechmono": ("ShareTechMono", "print", ("Regular",)),
    "b612mono": ("B612Mono", "print", ("Regular", "Bold")),
    "ibmplexmono": ("IBMPlexMono", "print", ("Regular", "Medium", "SemiBold", "Bold")),
    "spacemono": ("SpaceMono", "print", ("Regular", "Bold")),
    "cutivemono": ("CutiveMono", "print", ("Regular",)),
    "robotomono": ("RobotoMono", "print", ("VAR[wght]",)),
    "sourcecodepro": ("SourceCodePro", "print", ("VAR[wght]",)),
    "inconsolata": ("Inconsolata", "print", ("VAR[wdth,wght]",)),
    # stencil / punched-die look: stamped chassis numbers
    "stardosstencil": ("StardosStencil", "stamp", ("Regular", "Bold")),
    "allertastencil": ("AllertaStencil", "stamp", ("Regular",)),
    "sairastencilone": ("SairaStencilOne", "stamp", ("Regular",)),
    "bigshouldersstencildisplay": ("BigShouldersStencilDisplay", "stamp", ("VAR[wght]",)),
    "sirinstencil": ("SirinStencil", "stamp", ("Regular",)),
}


def _get(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "vinplate-font-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def main() -> int:
    ok = 0
    for fam, (prefix, role, styles) in FAMILIES.items():
        lic = _get(f"{RAW}/{fam}/OFL.txt")
        if lic is None:
            print(f"skip {fam}: no OFL.txt (not OFL, or renamed)")
            continue
        d = OUT / fam
        d.mkdir(parents=True, exist_ok=True)
        (d / "OFL.txt").write_bytes(lic)
        (d / "ROLE").write_text(role)
        got = 0
        for st in styles:
            name = f"{prefix}{st[3:]}.ttf" if st.startswith("VAR") else f"{prefix}-{st}.ttf"
            blob = _get(f"{RAW}/{fam}/{urllib.parse.quote(name)}")
            if blob:
                (d / name).write_bytes(blob)
                got += 1
        print(f"{fam:28s} {role:8s} {got}/{len(styles)} files")
        ok += bool(got)
    print(f"done: {ok} families -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
