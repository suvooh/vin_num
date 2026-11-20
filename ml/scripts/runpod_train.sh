#!/usr/bin/env bash
# Train the line recogniser on a GPU pod (Runpod PyTorch template, Linux, CUDA).
#
#   bash scripts/runpod_train.sh [PLATE_LINES] [VIN_LINES] [STEPS] [WIDTH]
#
# Stages: (1) tests, (2) synthetic shards (CPU-bound: ~15 lines/s/core), (3) training,
# (4) ONNX + INT8 export, (5) one tarball to download. Safe to re-run: finished shards are
# skipped by name. Everything lands under /workspace (the persistent volume).
set -euo pipefail
cd "$(dirname "$0")/.."

PLATE=${1:-800000}
VIN=${2:-500000}
STEPS=${3:-40000}
WIDTH=${4:-2.0}
OUT=/workspace
mkdir -p "$OUT/runs" "$OUT/shards"

echo "== env"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
pip install -q -r requirements-train.txt

echo "== tests (must pass before spending GPU time)"
python -m pytest tests -q

echo "== fonts"; [ -d synth/fonts ] && [ "$(ls synth/fonts | wc -l)" -gt 5 ] || python scripts/fetch_fonts.py

echo "== shards ($PLATE plate-row lines, $VIN VIN lines)"
python scripts/make_shards.py --out "$OUT/shards" --plate "$PLATE" --vin "$VIN" --workers "$(nproc)"

echo "== train (width $WIDTH, $STEPS steps)"
python -m training.train_rec --data "$OUT/shards" --steps "$STEPS" --batch 256 \
    --workers "$(( $(nproc) > 16 ? 16 : $(nproc) ))" --amp --width "$WIDTH" \
    --val-every 1000 --val-batches 24 --out "$OUT/runs/v1" 2>&1 | tee "$OUT/runs/v1.log"

echo "== export (synthetic calibration: re-calibrate on REAL captures before shipping)"
python -m export.export_onnx --ckpt "$OUT/runs/v1/rec_ema.pt" --out "$OUT/build" --calib 512

echo "== synthetic plumbing eval (NOT a real-world number)"
python -m eval.run_eval --synthetic 600 --plate-model "$OUT/build/rec_plate_int8.onnx" \
    --vin-model "$OUT/build/rec_vin_int8.onnx" --provider cpu --json "$OUT/runs/v1_synth_eval.json" || true

tar czf "$OUT/vinplate_run_v1.tgz" -C "$OUT" runs/v1 runs/v1.log runs/v1_synth_eval.json build
echo "== done: $OUT/vinplate_run_v1.tgz"
