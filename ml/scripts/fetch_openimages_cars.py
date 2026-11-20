"""Fetch a small set of real vehicle photos (with a labelled registration plate) from Open Images.

Used as the *bona fide* real-camera pool for recapture (presentation-attack) experiments, and as
the content that gets re-photographed in simulated screen/print attacks. NOT Indian data.

Licence handling: Google states annotations are CC BY 4.0 and images are listed CC BY 2.0 but
says to verify per image. This script keeps only rows whose License column is a plain
creativecommons.org/licenses/by/2.0 URL and writes ATTRIBUTION.csv (author, title, source page,
licence) next to the images, which CC BY requires you to retain. Everything under --out is
untrusted content: read it with `python -I` and decode only with Pillow.

    python -I scripts/fetch_openimages_cars.py --out data/openimages --n 260
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://storage.googleapis.com/openimages"
FILES = {
    "classes.csv": f"{BASE}/v5/class-descriptions-boxable.csv",
    "val_bbox.csv": f"{BASE}/v5/validation-annotations-bbox.csv",
    "val_images.csv": f"{BASE}/2018_04/validation/validation-images-with-rotation.csv",
}


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vinplate-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=260)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        if not (out / name).exists():
            print("fetching", name, flush=True)
            (out / name).write_bytes(get(url, 300))

    with open(out / "classes.csv", newline="", encoding="utf-8") as f:
        mids = {row[1].lower(): row[0] for row in csv.reader(f)}
    plate = mids.get("vehicle registration plate")
    if not plate:
        print("class 'Vehicle registration plate' not found", file=sys.stderr)
        return 1

    wanted: set[str] = set()
    with open(out / "val_bbox.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] == plate and row["Confidence"] == "1":
                wanted.add(row["ImageID"])
    print(f"{len(wanted)} validation images carry a plate box", flush=True)

    rows = []
    with open(out / "val_images.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ImageID"] in wanted and r["License"].rstrip("/").endswith("licenses/by/2.0"):
                rows.append(r)
    print(f"{len(rows)} of those are plain CC BY 2.0", flush=True)
    rows = rows[: a.n * 3]

    import collections
    import time

    fails: collections.Counter = collections.Counter()

    def fetch(url: str):
        for attempt in range(3):
            try:
                return get(url, 25)
            except urllib.error.HTTPError as e:
                fails[e.code] += 1
                if e.code == 429:
                    time.sleep(2 + 3 * attempt)      # Flickr rate limit: back off
                    continue
                return None
            except Exception as e:
                fails[type(e).__name__] += 1
                return None
        return None

    def grab(r):
        # ~640px rendition first (the full-size farm URLs are heavily rate-limited), then original
        for key in ("Thumbnail300KURL", "OriginalURL"):
            blob = fetch(r[key]) if r.get(key) else None
            if not blob:
                continue
            try:
                im = Image.open(io.BytesIO(blob))
                im.load()
            except Exception:
                fails["undecodable"] += 1
                continue
            if min(im.size) < 320:
                fails["too_small"] += 1
                continue
            (out / "images" / f"{r['ImageID']}.jpg").write_bytes(blob)
            time.sleep(0.15)
            return r
        return None

    kept = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(grab, rows):
            if r:
                kept.append(r)
                if len(kept) % 25 == 0:
                    print(f"  {len(kept)} images (failures so far: {dict(fails)})", flush=True)
            if len(kept) >= a.n:
                break
    print("failure counts:", dict(fails))
    with open(out / "ATTRIBUTION.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ImageID", "Author", "Title", "AuthorProfileURL", "OriginalLandingURL", "License"])
        for r in kept:
            w.writerow([r["ImageID"], r["Author"], r["Title"], r["AuthorProfileURL"], r["OriginalLandingURL"], r["License"]])
    print(f"done: {len(kept)} images -> {out}/images (+ ATTRIBUTION.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
