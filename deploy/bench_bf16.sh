#!/usr/bin/env bash
# Compare FluxRT bf16 vs int8 on the current pod. Run from /workspace/fluxrt after runpod_setup.sh.
set -uo pipefail
export HF_HOME=/workspace/hf
cd /workspace/fluxrt
log(){ echo "[bf16bench $(date +%H:%M:%S)] $*"; }
log "idle gpu: $(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader)"
if [ ! -f FLUX.2-klein-4B/transformer/config.json ] || [ ! -f FLUX.2-klein-4B/text_encoder/config.json ]; then
  log "downloading bf16 transformer + text encoder"
  t0=$(date +%s)
  .venv/bin/hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B --include "transformer/*" "text_encoder/*" > /dev/null 2>&1
  log "download done in $(( $(date +%s) - t0 ))s ($(du -sh FLUX.2-klein-4B | cut -f1))"
fi
for cfg in web_bf16_config web_config; do
  log "=== $cfg"
  t0=$(date +%s)
  timeout 1200 .venv/bin/python scripts/test_backend.py --config configs/$cfg.json --device -1 --seconds 25 --out /workspace/bf16bench_$cfg 2>&1 | grep -E "ready in|proc=|Error|Traceback|OutOfMemory" | tail -4
  log "$cfg total $(( $(date +%s) - t0 ))s"
  pkill -9 -f "spawn_mai[n]"; sleep 3
done
log "DONE"
