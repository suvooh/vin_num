"""Process-wide models: the plate/VIN scene reader and the authenticity checks. Loaded once, lazily."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from vinplate.authenticity import Authenticator
from vinplate.ocr import PpOcrEngine
from vinplate.scene import SceneReader


@dataclass
class Engine:
    reader: SceneReader
    authenticator: Authenticator
    provider: str


_engine: Engine | None = None
_error: str | None = None
_lock = threading.Lock()


def get_engine(provider: str = "cpu") -> Engine:
    global _engine, _error
    with _lock:
        if _engine is None and _error is None:
            try:
                ocr = PpOcrEngine(provider=provider)
                _engine = Engine(SceneReader(ocr), Authenticator.from_default(provider="cpu"), provider)
            except Exception as exc:  # models missing, onnxruntime not installed, ...
                _error = f"OCR engine unavailable: {exc}"
        if _engine is None:
            raise RuntimeError(_error or "OCR engine unavailable")
        return _engine


def engine_health() -> dict[str, Any]:
    out: dict[str, Any] = {"ok": _error is None, "engine": "pp-ocr v3 (onnxruntime) + grammar decoder",
                           "loaded": _engine is not None, "error": _error}
    if _engine is not None:
        out["provider"] = _engine.provider
        out["recapture_model"] = _engine.authenticator.pad is not None
        out["recapture_model_error"] = getattr(_engine.authenticator, "pad_error", None)
    return out
