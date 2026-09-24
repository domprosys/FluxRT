#!/usr/bin/env bash
# Session A follow-up (run after bench_session_a.sh, same pod): cached attention after the
# inference_mode fix, LivePortrait with the face box taken from the webcam, and the
# all-engines server with TensorRT (reuses the engines phase 4 built).
#   bash deploy/bench_session_a2.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client script}; OUT=${3:-/workspace/sessA}
export WS=/workspace HF_HOME=/workspace/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd /workspace/fluxrt
mkdir -p "$OUT"
TAG=sessA2
source deploy/bench_lib.sh

# Ordered by priority: the pod's 3 h cap may cut the tail.
# ── LivePortrait: face box from the webcam frame instead of detecting it in the stylised one ──
T flux_lp_driving     "$(cfgvar configs/web_bf16_liveportrait_config.json lp_driving 'lip_transfer.source_crop="driving"')"

# ── all engines resident with TensorRT (the template's ENABLE_TRT=1 path) ──
# first start builds the FaceID+CN and SDXL+CN UNets and the SDXL ControlNet engines (serialized)
export SD_ACCELERATION=tensorrt
MULTI_WAIT_S=1800 multi_run multi_trt configs/multi_config.json 50 "12:sdxl 15:flux 15:sd"
unset SD_ACCELERATION

T flux_lp_driving_exp "$(cfgvar configs/web_bf16_liveportrait_config.json lp_driving_exp 'lip_transfer.source_crop="driving"' 'lip_transfer.region="exp"')"

# ── cached attention (StreamV2V), moving input only: SD is deterministic on a still frame.
# cache12 already ran in bench_session_a.sh (its worker started after the fix landed).
T sdcn_cache4 "$(cfgvar configs/sd_controlnet_config.json cache4 worker.use_cached_attn=true worker.cache_maxframes=4 worker.max_cache_maxframes=16 worker.cache_interval=1)"
du -sh /workspace/engines 2>/dev/null > "$OUT/engines_size.txt"
say "=== done"
