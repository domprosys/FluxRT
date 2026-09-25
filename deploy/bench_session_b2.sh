#!/usr/bin/env bash
# Session B follow-up (same pod, after bench_session_b.sh): the 14B at noise_scale 0.8 / 1 step
# ignored the input (a prompt-only robot, dark frames at 2 steps and 448x448). Upstream pairs
# --normalize_latents ("fixes dark/color-shifted outputs") with a lower noise scale (0.6).
#   WS=/root/ws bash deploy/bench_session_b2.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client script}
export WS=${WS:-/root/ws}
OUT=${3:-$WS/sessB}
export HF_HOME=$WS/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd "$WS/fluxrt" || exit 1
TAG=sessB2
source deploy/bench_lib.sh
T_SECONDS=60 T_WARMUP=15

T sdv2_14b_ns06      "$(cfgvar configs/sdv2_14b_config.json ns06 worker.noise_scale=0.6)"
T sdv2_14b_ns06_norm "$(cfgvar configs/sdv2_14b_config.json ns06n worker.noise_scale=0.6 worker.normalize_latents=true)"
T sdv2_14b_ns07_norm "$(cfgvar configs/sdv2_14b_config.json ns07n worker.noise_scale=0.7 worker.normalize_latents=true)"
T sdv2_ns06_norm     "$(cfgvar configs/sdv2_config.json ns06n worker.noise_scale=0.6 worker.normalize_latents=true)"
T sdv2_samples       configs/sdv2_config.json   # stage-1 runs predate the sample-saving fix
say "=== done"
