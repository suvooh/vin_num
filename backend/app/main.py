"""VIN and number-plate reader: one small FastAPI service, fully offline.

Photos go through plate / VIN detection, deskew, a recogniser whose posteriors are kept, and a
grammar-constrained decoder (Indian plate layouts, ISO 3779 VIN) that marks damaged characters
``?`` instead of guessing them. A capture bundle (normal + flash + other angles) is additionally
checked for presentation attacks (a photo of a screen or a print). See ``app.service`` for roles.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import engine_health, get_engine
from app.imaging import decode
from app.service import ROLES, ImageIn, analyse

BACKEND_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BACKEND_ROOT.parent / "frontend"
CAPTURES_DIR = Path(os.environ.get("VIN_NUM_CAPTURES") or BACKEND_ROOT / "data" / "captures")

ALLOWED = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_BYTES = 15 * 1024 * 1024
MAX_FILES = 6

# The models use every CPU core already, so captures are analysed one at a time.
_gate = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:                                    # load now so a missing model shows up in /api/health at once
        await run_in_threadpool(get_engine)
    except RuntimeError:
        pass
    yield


app = FastAPI(
    title="VIN & Number Plate Reader",
    version="2.0.0",
    description="Offline VIN and Indian registration-plate reading with damage-aware decoding and "
                "presentation-attack (screen / print) checks.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{h}:{p}" for h in ("localhost", "127.0.0.1") for p in (3000, 5173, 8002)],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


@app.get("/api/health")
def health() -> dict[str, Any]:
    h = engine_health()
    return {"ok": bool(h.get("ok")), "ocr": h}


@app.post("/api/extract")
async def extract(
    files: list[UploadFile] = File(...),
    roles: str = Form("", description="comma list, one per file: " + "|".join(ROLES)),
    hsrp: str = Form("false", description="the plate is a retroreflective HSRP plate (enables the flash test)"),
    require_liveness: str = Form("false", description="flag the capture for review unless liveness is confirmed"),
) -> dict[str, Any]:
    """Read 1-6 photos and return the best VIN and plate plus an authenticity / liveness verdict."""
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(400, f"send 1-{MAX_FILES} images")
    role_list = [r.strip().lower() for r in roles.split(",")] if roles.strip() else ["auto"] * len(files)
    if len(role_list) != len(files):
        raise HTTPException(400, f"{len(files)} files but {len(role_list)} roles")
    bad = [r for r in role_list if r not in ROLES]
    if bad:
        raise HTTPException(400, f"unknown role {bad[0]!r}; use one of {', '.join(ROLES)}")
    if ("plate_flash" in role_list or "angle" in role_list) and "plate" not in role_list:
        raise HTTPException(400, "'plate_flash' and 'angle' photos need a 'plate' photo to compare against")
    if role_list.count("plate_flash") > 1:
        raise HTTPException(400, "send at most one 'plate_flash' photo")

    capture_id = uuid.uuid4().hex[:12]
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    images: list[ImageIn] = []
    for i, (f, role) in enumerate(zip(files, role_list)):
        if f.content_type not in ALLOWED:
            raise HTTPException(415, f"{f.filename}: only JPEG, PNG or WebP")
        payload = await f.read()
        if len(payload) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: image too large")
        try:
            rgb, taken = await run_in_threadpool(decode, payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        (CAPTURES_DIR / f"{capture_id}_{i}{ALLOWED[f.content_type]}").write_bytes(payload)
        images.append(ImageIn(f.filename or f"photo{i}", payload, rgb, role, taken.isoformat() if taken else None))

    try:
        eng = await run_in_threadpool(get_engine)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    async with _gate:
        result = await run_in_threadpool(
            analyse, images, eng.reader, eng.authenticator,
            hsrp_expected=_flag(hsrp), require_liveness=_flag(require_liveness))
    return {"capture_id": capture_id, "processed_at": datetime.now().replace(microsecond=0).isoformat(), **result}


@app.get("/api/captures/{capture_id}/{index}")
def capture_image(capture_id: str, index: int) -> FileResponse:
    if not capture_id.isalnum():
        raise HTTPException(400, "bad capture id")
    for ext in ALLOWED.values():
        path = CAPTURES_DIR / f"{capture_id}_{index}{ext}"
        if path.is_file():
            return FileResponse(path)
    raise HTTPException(404, "capture not found")


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("VIN_NUM_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("VIN_NUM_API_PORT", "8002")),
        reload=False,
    )


if __name__ == "__main__":
    main()
