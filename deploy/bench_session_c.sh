#!/usr/bin/env bash
# Session C (RTX PRO 6000, open issues): does gc.freeze() remove the latency spikes (TensorRT SD+ControlNet
# 57-168 ms every 5-10 s, cached attention p95 83 vs p50 44 ms in Session A)? And FaceID-PlusV2 vs FaceID.
# Container-disk layout (WS=/root/ws), like the template.
#   WS=/root/ws bash deploy/bench_session_c.sh /root/clip.mp4 [OUT]
set -uo pipefail
CLIP=${1:?clip}
export WS=${WS:-/root/ws}
OUT=${2:-$WS/sessC}
export HF_HOME=$WS/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd "$WS/fluxrt" || exit 1
mkdir -p "$OUT"
TAG=sessC
CLIENT=/dev/null  # no WebRTC runs here
source deploy/bench_lib.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt"

EXTRA=$(python3 - <<'EOF'
import json
seen = []
for c in ["sd_controlnet_config", "sd_controlnet_faceid_config", "sd_controlnet_faceidplus_config"]:
    for m in json.load(open(f"configs/{c}.json")).get("hf_models", []):
        if m not in seen: seen.append(m)
print(" ".join(seen))
EOF
)
say "install start"
t0=$(date +%s)
if SKIP_SD=0 SKIP_SDV2=1 SKIP_FLUXRT_WEIGHTS=1 WITH_TRT=1 WITH_FACEID=1 EXTRA_HF_MODELS="$EXTRA" \
   bash deploy/runpod_setup.sh > "$OUT/install.log" 2>&1; then
  say "install OK in $(( $(date +%s) - t0 ))s"
else
  say "INSTALL FAILED after $(( $(date +%s) - t0 ))s"; tail -n 30 "$OUT/install.log"; exit 1
fi

T_SECONDS=60 T_WARMUP=10
# ── 1. gc.freeze A/B on PyTorch cached attention; plain SD+CN as the no-spike reference ──
C4=$(cfgvar configs/sd_controlnet_config.json cache4 worker.use_cached_attn=true worker.cache_maxframes=4 worker.max_cache_maxframes=16 worker.cache_interval=1)
FLUXRT_GC_FREEZE=0 T cache4_nofreeze "$C4"
T cache4_freeze "$C4"
T sdcn_freeze configs/sd_controlnet_config.json

# ── 2. TensorRT SD+ControlNet: the first run builds the UNet + ControlNet engines (~9 min) ──
export SD_ACCELERATION=tensorrt
T trt_sdcn_build configs/sd_controlnet_config.json
FLUXRT_GC_FREEZE=0 T trt_sdcn_nofreeze configs/sd_controlnet_config.json
T trt_sdcn_freeze configs/sd_controlnet_config.json
unset SD_ACCELERATION

# ── 3. identity: FaceID vs FaceID-PlusV2, both capturing the face at input frame 60 ──
T faceid     configs/sd_controlnet_faceid_config.json --set 60:faceid_capture=true
T faceidplus configs/sd_controlnet_faceidplus_config.json --set 60:faceid_capture=true
say "=== done"
