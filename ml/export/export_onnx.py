"""Export the recogniser to ONNX, then (optionally) to a static-INT8 QDQ model for the NPU.

    python -m export.export_onnx --ckpt runs/v1/rec_ema.pt --out build/ --calib 512

Produces, for each line shape (plate 48x256, VIN 48x512):
    rec_plate_fp32.onnx / rec_vin_fp32.onnx
    rec_plate_int8.onnx / rec_vin_int8.onnx      (QDQ, per-channel weights, uint8 activations)

Why static shapes: Qualcomm's HTP compiles a graph per input shape; dynamic widths fall back
to CPU. Two fixed shapes cover every line the pipeline produces (fit_line pads/squeezes to
them), which is also why both training and inference go through ``fit_line``.

Calibration uses SYNTHETIC lines by default. For a production quantisation, calibrate on real
captures - activation ranges on real glare/dirt differ from the synthetic ones, and the INT8
accuracy gap must be measured on a real held-out set (``eval/``) before shipping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.models import LineRecognizer
from vinplate.geometry.lines import PLATE_LINE_SHAPE, VIN_LINE_SHAPE, to_gray

SHAPES = {"plate": PLATE_LINE_SHAPE, "vin": VIN_LINE_SHAPE}


def load_model(ckpt: str | None, width: float = 1.0) -> LineRecognizer:
    if ckpt is None:
        m = LineRecognizer(width)
    else:
        d = torch.load(ckpt, map_location="cpu")
        m = LineRecognizer(d.get("width", width))
        m.load_state_dict(d["model"])
    return m.eval()


def export_fp32(model: LineRecognizer, shape: tuple[int, int], path: Path) -> None:
    x = torch.rand(1, 1, *shape)
    torch.onnx.export(
        model, x, str(path), input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )


class _Calib:
    def __init__(self, domain: str, n: int, seed: int = 4242):
        from synth.sampler import sample_plate_lines, sample_vin_line

        rng = np.random.default_rng(seed)
        fn = sample_plate_lines if domain == "plate" else sample_vin_line
        self.data, self.i = [], 0
        while len(self.data) < n:
            self.data += [to_gray(l.img)[None, None].astype(np.float32) / 255 for l in fn(rng)]
        self.data = self.data[:n]

    def get_next(self):
        if self.i >= len(self.data):
            return None
        x = self.data[self.i]
        self.i += 1
        return {"image": x}


def quantize(fp32: Path, int8: Path, domain: str, n_calib: int) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)

    class Reader(_Calib, CalibrationDataReader):
        pass

    quantize_static(
        str(pre), str(int8), Reader(domain, n_calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8,
        per_channel=True, calibrate_method=CalibrationMethod.Percentile,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": False,
                       "CalibPercentile": 99.999},
    )
    pre.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint; omit for random weights (latency only)")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--no-int8", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.ckpt, a.width)
    for dom, shape in SHAPES.items():
        fp = out / f"rec_{dom}_fp32.onnx"
        export_fp32(model, shape, fp)
        print(f"wrote {fp} ({fp.stat().st_size / 1e6:.2f} MB)")
        if not a.no_int8:
            q = out / f"rec_{dom}_int8.onnx"
            quantize(fp, q, dom, a.calib)
            print(f"wrote {q} ({q.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
