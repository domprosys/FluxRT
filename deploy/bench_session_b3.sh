#!/usr/bin/env bash
# Session B diagnostics (same pod, after bench_session_b2.sh):
#  - every sdv2 stream reset costs ~12 s (first-chunk VAE encode ~8 s + decode ~4 s): cudnn autotune?
#  - the 14B ignores the input at 512x512: upstream runs it at 480x832; or is it our scene prompt?
#   WS=/root/ws bash deploy/bench_session_b3.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client script}
export WS=${WS:-/root/ws}
OUT=${3:-$WS/sessB}
export HF_HOME=$WS/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd "$WS/fluxrt" || exit 1
TAG=sessB3
source deploy/bench_lib.sh
T_SECONDS=60 T_WARMUP=15 T_SAMPLES=500,900,1300  # sdv2's first output comes ~12 s in
SCENE="An astronaut in an orange NASA spacesuit with a chrome cyborg face and glowing blue eyes, smiling, an American flag and a model space shuttle behind her, studio portrait photo"

T sdv2_nobench      "$(cfgvar configs/sdv2_config.json d_nobench worker.cudnn_benchmark=false)"
T sdv2_14b_832      "$(cfgvar configs/sdv2_14b_config.json d_832 resolution.width=832 resolution.height=480)"
T sdv2_14b_832_n07  "$(cfgvar configs/sdv2_14b_config.json d_832n07 resolution.width=832 resolution.height=480 worker.noise_scale=0.7 worker.normalize_latents=true)"
T sdv2_14b_scene    configs/sdv2_14b_config.json --prompt "$SCENE"
T sdv2_scene        configs/sdv2_config.json --prompt "$SCENE"
say "=== done"
