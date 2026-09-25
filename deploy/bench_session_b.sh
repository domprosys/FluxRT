#!/usr/bin/env bash
# Session B benchmark (RTX PRO 6000): StreamDiffusionV2 1.3B on Blackwell, then the 14B, then all
# four engines resident. Runs from $WS/fluxrt with WS on the container disk: on RunPod /workspace
# can be a network fs even without a volume (EU-RO-1: MooseFS, ~0.4 GB/s reads vs ~6 GB/s local),
# so this also measures the local-disk layout for the template.
#   WS=/root/ws bash deploy/bench_session_b.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
# Installs are staged (1.3B, then the 14B checkpoint, then the other engines) so every stage
# reports before the next download starts.
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client script}
export WS=${WS:-/root/ws}
OUT=${3:-$WS/sessB}
export HF_HOME=$WS/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd "$WS/fluxrt" || exit 1
mkdir -p "$OUT"
TAG=sessB
source deploy/bench_lib.sh

# install name  (runpod_setup.sh knobs come from the caller's environment)
install() {
  local name=$1 t0; t0=$(date +%s)
  say "install $name start"
  if bash deploy/runpod_setup.sh > "$OUT/install_$name.log" 2>&1; then
    say "install $name OK in $(( $(date +%s) - t0 ))s"
    echo "{\"install_s\": $(( $(date +%s) - t0 ))}" > "$OUT/install_$name.json"
  else
    say "INSTALL $name FAILED after $(( $(date +%s) - t0 ))s"; tail -n 30 "$OUT/install_$name.log"; return 1
  fi
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt"
df -hT "$WS" /workspace > "$OUT/disks.txt" 2>&1

# ── stage 1: StreamDiffusionV2 1.3B ─────────────────────────────────────────────
SKIP_SD=1 SKIP_SDV2=0 SKIP_FLUXRT_WEIGHTS=1 SDV2_14B=0 install sdv2 || exit 1
T_SECONDS=60 T_WARMUP=15 T_SAMPLES=500,900,1300  # sdv2's first output comes ~12 s in
T sdv2          configs/sdv2_config.json
T sdv2_static   configs/sdv2_config.json --static
T sdv2_s1       "$(cfgvar configs/sdv2_config.json s1 worker.steps=1)"
T sdv2_taehv    "$(cfgvar configs/sdv2_config.json taehv 'worker.vae_type="lightvae_taehv"')"
T sdv2_640      "$(cfgvar configs/sdv2_config.json r640 resolution.width=640 resolution.height=352)"

# ── stage 2: 14B checkpoint (28.6 GB) ───────────────────────────────────────────
if SKIP_SD=1 SKIP_SDV2=0 SKIP_FLUXRT_WEIGHTS=1 SDV2_14B=1 install sdv2_14b; then
  T sdv2_14b        configs/sdv2_14b_config.json
  T sdv2_14b_static configs/sdv2_14b_config.json --static
  T sdv2_14b_448    "$(cfgvar configs/sdv2_14b_config.json r448 resolution.width=448 resolution.height=448)"
  T sdv2_14b_s2     "$(cfgvar configs/sdv2_14b_config.json s2 worker.steps=2)"
fi
T_SECONDS=45 T_WARMUP=10

# ── stage 3: the other engines, then all four resident + switching ──────────────
# config_needs.py is what the template's pod_start.sh runs, so this exercises that path too
eval "$(python3 deploy/config_needs.py configs/multi_all_config.json)"
export SDV2_14B=1  # keep the stage-2 checkpoint "wanted" so the stamp check does not complain
install all || exit 1
# Session A follow-ups: SDXL-Turbo without ControlNet drifted off the input at t_index 22;
# FaceID locked but the likeness was weak at scale 0.8
T sdxl_t32   "$(cfgvar configs/sdxl_config.json t32 'worker.t_index_list=[32]')"
T faceid_s12 "$(cfgvar configs/sd_controlnet_faceid_config.json s12 worker.ipadapter.scale=1.2)" --set 60:faceid_capture=true
multi_run multi_all configs/multi_all_config.json 80 "12:flux 18:sdxl 18:sdv2 18:sd"
# the 14B next to the others, if it fits (flux ~18 + sd ~5 + sdxl ~12 + 14B ~50 GB): an OOM is a result too
c=$(cfgvar configs/multi_all_config.json multi_14b 'engines.sdv2.config="sdv2_14b_config"' 'engines.sdv2.resolution={"height":448,"width":448}')
multi_run multi_14b "$c" 50 "12:sdv2 18:flux"
say "disk usage:"; du -sh "$WS"/* /root/venvs 2>/dev/null | sort -h | tail -8
say "=== done"
