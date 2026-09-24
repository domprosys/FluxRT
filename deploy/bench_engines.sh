#!/usr/bin/env bash
# Benchmark every engine config on the current pod with the same input clip.
#   bash deploy/bench_engines.sh /root/clip.mp4 [OUT_DIR]
# Assumes runpod_setup.sh already installed all engines (no SKIP_*), WITH_BF16=1.
set -uo pipefail
CLIP=${1:?clip path}
OUT=${2:-/workspace/bench_engines}
export HF_HOME=${HF_HOME:-/workspace/hf}
cd /workspace/fluxrt
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader > "$OUT/gpu.txt"
for cfg in sd_config sd_controlnet_config web_config web_bf16_config sdv2_config; do
  echo "=== $cfg $(date +%H:%M:%S)"
  timeout 1200 .venv/bin/python scripts/test_backend.py --config configs/$cfg.json --video "$CLIP" \
    --seconds 45 --warmup 10 --samples 60,120,180 --json "$OUT/$cfg.json" --out "$OUT/$cfg" 2>&1 \
    | grep -E "ready in|BENCH|died|Error|Traceback|OutOfMemory" | cut -c1-400
  pkill -9 -f "spawn_mai[n]|sd_worke[r]|sdv2_worke[r]"; sleep 3
done
echo "=== done $(date +%H:%M:%S)"
