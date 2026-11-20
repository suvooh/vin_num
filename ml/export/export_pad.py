"""Export the recapture classifier (PadNet) to ONNX, quantise to INT8 and CHECK what that cost.

    python -I -X utf8 export/export_pad.py --ckpt runs/pad/pad.pt --images data/openimages/images --out build_pad

The model takes K=4 native-resolution 96x96 patches per call (static shape for the NPU/GPU
compilers) and returns 4 logits; the app averages them. Quantisation can silently destroy a
detector that keys on faint high-frequency structure, so this script reports AUC for the torch
FP32 model, ONNX FP32 and ONNX INT8 on the same held-out simulated patches, and the latency of
each. Calibration and evaluation data are simulated attacks on real photos (see train_pad.py);
calibrate on real captures before shipping.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch

from training.pad_model import PadNet
from training.train_pad import PATCH, auc, load_pool, make_sample

K = 4


class _Reader:
    def __init__(self, pool, n, seed=31):
        rng = np.random.default_rng(seed)
        kinds = ["real", "screen", "print"]
        self.items = []
        for _ in range(n):
            xs = np.stack([make_sample(pool, rng, kinds[int(rng.integers(3))]) for _ in range(K)])
            self.items.append(torch.from_numpy(xs).permute(0, 3, 1, 2).float().div_(255).numpy())
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        self.i += 1
        return {"patches": self.items[self.i - 1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib", type=int, default=64)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(a.images)
    cut = int(len(pool) * 0.75)
    train_pool, val_pool = pool[:cut], pool[cut:]

    d = torch.load(a.ckpt, map_location="cpu")
    model = PadNet(d.get("width", 1.0))
    model.load_state_dict(d["model"])
    model.eval()
    fp32, int8 = a.out / "pad_fp32.onnx", a.out / "pad_int8.onnx"
    torch.onnx.export(model, torch.rand(K, 3, PATCH, PATCH), str(fp32), input_names=["patches"],
                      output_names=["logits"], opset_version=17, dynamo=False)

    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = a.out / "pad_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)

    class Reader(_Reader, CalibrationDataReader):
        pass

    quantize_static(str(pre), str(int8), Reader(train_pool, a.calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.999})
    pre.unlink(missing_ok=True)
    print(f"fp32 {fp32.stat().st_size / 1e3:.0f} KB   int8 {int8.stat().st_size / 1e3:.0f} KB")

    # ---- held-out accuracy of each variant on identical patches
    rng = np.random.default_rng(777)
    n = 240
    sets = {k: np.stack([make_sample(val_pool, rng, k, (0.8, 2.0)) for _ in range(n)]) for k in ("real", "screen", "print")}
    sets["screen_ood"] = np.stack([make_sample(val_pool, rng, "screen", (2.0, 3.2)) for _ in range(n)])

    def torch_scores(x):
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255)
        with torch.no_grad():
            return torch.cat([model(t[i : i + 64]) for i in range(0, n, 64)]).numpy()

    def onnx_scores(path, x):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255).numpy()
        return np.concatenate([s.run(None, {"patches": t[i : i + K]})[0] for i in range(0, n, K)])

    print(f"\nheld-out AUC vs real photos (simulated attacks, {n} patches/class):")
    print(f"{'variant':10s} {'screen':>8s} {'screen-OOD':>11s} {'print':>7s}")
    for name, fn in (("torch fp32", torch_scores), ("onnx fp32", lambda x: onnx_scores(fp32, x)), ("onnx int8", lambda x: onnx_scores(int8, x))):
        real = fn(sets["real"])
        print(f"{name:10s} {auc(fn(sets['screen']), real):8.3f} {auc(fn(sets['screen_ood']), real):11.3f} {auc(fn(sets['print']), real):7.3f}")


if __name__ == "__main__":
    main()
