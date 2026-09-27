#!/usr/bin/env bash
# Session D (RTX PRO 6000): generation fps at different resolutions and aspect ratios for
# SD+ControlNet (TensorRT), SDXL+ControlNet (TensorRT), FluxRT bf16 and StreamDiffusionV2 1.3B,
# plus one end-to-end WebRTC stream at 1024x576 output (aiortc's CPU VP8 encoder may be the limit).
#   WS=/root/ws bash deploy/bench_session_d.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
# Needs HF_TOKEN in pid 1's env (template-style secret) for the TensorRT engine cache.
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client}
export WS=${WS:-/root/ws}
OUT=${3:-$WS/sessD}
export HF_HOME=$WS/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FLUXRT_THREADS=4
export TRT_CACHE_REPO=alexcloak/fluxrt-trt-engines
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(HF_TOKEN|TRT_CACHE_REPO)=' | sed "s/^\([A-Z_]*\)=\(.*\)$/export \1='\2'/")"
cd "$WS/fluxrt" || exit 1
mkdir -p "$OUT"
TAG=sessD
source deploy/bench_lib.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt"

say "install start"
t0=$(date +%s)
eval "$(python3 deploy/config_needs.py configs/multi_all_config.json)"
if WITH_TRT=1 bash deploy/runpod_setup.sh > "$OUT/install.log" 2>&1; then
  say "install OK in $(( $(date +%s) - t0 ))s"
else
  say "INSTALL FAILED"; tail -n 30 "$OUT/install.log"; exit 1
fi
bash deploy/trt_engine_cache.sh pull 2>&1 | grep trt-cache
export SD_ACCELERATION=tensorrt

# ── end-to-end: SD + SDXL at 1024x576 in, 1024x576 out, over WebRTC ──
C=$(python3 - <<'EOF'
import json
c = {"backend": "multi", "resolution": {"height": 576, "width": 1024}, "out_resolution": {"height": 576, "width": 1024},
     "default_engine": "sd", "crossfade_s": 1.0, "input_smoothing_alpha": 1.0,
     "engines": {"sd": {"config": "sd_controlnet_config", "resolution": {"height": 576, "width": 1024}},
                 "sdxl": {"config": "sdxl_controlnet_config", "resolution": {"height": 576, "width": 1024}}}}
json.dump(c, open("configs/_bench_multi1024.json", "w"), indent=1)
print("configs/_bench_multi1024.json")
EOF
)
MULTI_WAIT_S=600 multi_run webrtc1024 "$C" 40 "20:sdxl"

# ── resolution sweep (priority order: the cap may cut the tail) ──
T_SECONDS=25 T_WARMUP=5
for r in 1024x576 768x432 640x384 960x544 576x1024 384x640 768x576 640x480 768x768 512x512; do
  w=${r%x*}; h=${r#*x}
  T sd_$r   "$(cfgvar configs/sd_controlnet_config.json   sd_$r   resolution.width=$w resolution.height=$h)"
  T sdxl_$r "$(cfgvar configs/sdxl_controlnet_config.json sdxl_$r resolution.width=$w resolution.height=$h)"
  T_SAMPLES=400,500,600 T_SECONDS=30 T sdv2_$r "$(cfgvar configs/sdv2_config.json sdv2_$r resolution.width=$w resolution.height=$h)"
  T flux_$r "$(cfgvar configs/web_bf16_config.json      flux_$r resolution.width=$w resolution.height=$h)"
done
say "=== done"
