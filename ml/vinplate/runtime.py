"""ONNX Runtime session factory and model-file lookup, shared by every inference module.

Lives inside ``vinplate`` (not ``export/``) so an installed ``vinplate`` package has no import
of a sibling top-level package. ``export.bench`` re-exports ``make_session`` for the CLI.

``cpu`` is the portable default (MIT-licensed ONNX Runtime only). ``npu`` / ``qnn`` / ``gpu`` use
the Qualcomm QNN plugin execution provider and raise ``RuntimeError`` when it is not usable, so a
caller can fall back to CPU explicitly instead of silently running somewhere unexpected.
"""

from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort

# <repo>/ml/vinplate/runtime.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRS = (REPO_ROOT / "models", REPO_ROOT / "app" / "assets" / "models")


def find_model(name: str, extra_dirs: tuple[Path, ...] = ()) -> Path:
    """Resolve a model file: ``$VINPLATE_MODELS`` first, then ``<repo>/models`` and the Flutter
    asset folder (the single place the shipped PP-OCR / PadNet files live)."""
    dirs = []
    if os.environ.get("VINPLATE_MODELS"):
        dirs.append(Path(os.environ["VINPLATE_MODELS"]))
    dirs += [*extra_dirs, *MODEL_DIRS]
    for d in dirs:
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"model '{name}' not found in: " + ", ".join(str(d) for d in dirs))


def make_session(path: str, provider: str = "cpu", perf: str = "burst",
                 threads: int | None = None) -> ort.InferenceSession:
    """provider: cpu | npu (alias qnn) | gpu. npu/gpu use the QNN *plugin* execution provider
    (onnxruntime-qnn >= 2.x): register its library, pick the device, attach it explicitly."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    threads = threads or int(os.environ.get("VINPLATE_THREADS", "0")) or None     # cap CPU use (e.g. parallel benchmarks)
    if threads:
        so.intra_op_num_threads = threads
    if provider == "cpu":
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    try:
        import onnxruntime_qnn
    except ImportError as exc:
        raise RuntimeError("onnxruntime-qnn is not installed (needed for npu/gpu providers)") from exc

    want = {"npu": ort.OrtHardwareDeviceType.NPU, "qnn": ort.OrtHardwareDeviceType.NPU,
            "gpu": ort.OrtHardwareDeviceType.GPU}[provider]
    name = onnxruntime_qnn.get_ep_name()
    try:
        ort.register_execution_provider_library(name, onnxruntime_qnn.get_library_path())
    except Exception:  # already registered in this process
        pass
    dev = [d for d in ort.get_ep_devices() if d.ep_name == name and d.device.type == want]
    if not dev:
        raise RuntimeError(f"no QNN {provider.upper()} device found (needs a Snapdragon X / Windows-on-ARM machine)")
    opts = {"htp_performance_mode": perf, "enable_htp_fp16_precision": "1"} if want == ort.OrtHardwareDeviceType.NPU else {}
    so.add_provider_for_devices(dev[:1], opts)
    return ort.InferenceSession(str(path), so)
