#!/usr/bin/env bash
# Bootstrap a RunPod pod whose network volume is mounted at /workspace.
# Idempotent: re-running only fills in what is missing. Everything persistent
# (code, venvs, models, HF cache, uv cache) lives on the volume so a new pod
# attached to the same volume is ready after this script's quick checks.
#
#   bash /workspace/fluxrt/deploy/runpod_setup.sh            # full setup
#   bash /workspace/fluxrt/deploy/runpod_setup.sh --check    # verify only
set -euo pipefail

WS=/workspace
REPO=$WS/fluxrt
SD=$WS/StreamDiffusion-daydream
export HF_HOME=$WS/hf
export UV_CACHE_DIR=$WS/.uv-cache
export PATH="$HOME/.local/bin:$PATH"
CHECK_ONLY=${1:-}

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

# ── 1. base tools ────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
if ! command -v rsync >/dev/null || ! dpkg -s libgl1 >/dev/null 2>&1; then
  log "apt: rsync, libgl1 (opencv), git-lfs"
  apt-get update -qq && apt-get install -y -qq rsync libgl1 libglib2.0-0 git-lfs >/dev/null
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── 2. FluxRT venv (torch cu128) ────────────────────────────────────────────
[ -d "$REPO" ] || { echo "repo missing at $REPO (rsync it first)"; exit 1; }
cd "$REPO"
if [ ! -x .venv/bin/python ]; then
  [ -n "$CHECK_ONLY" ] && { echo "MISSING: fluxrt venv"; exit 1; }
  log "creating FluxRT venv"
  uv venv --python 3.12 .venv >/dev/null
  uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python .venv/bin/python -r requirements.txt aiortc fastapi "uvicorn[standard]"
  uv pip install --python .venv/bin/python -e .
fi
.venv/bin/python -c "import torch, aiortc, fluxrt; print('fluxrt venv ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── 3. FluxRT models (GPU path: int8 transformer + int8 text encoder) ───────
need_flux=0
for f in FLUX.2-klein-4B-int8/diffusion_pytorch_model.safetensors FLUX.2-klein-4B-int8/text_encoder/model.safetensors \
         FLUX.2-klein-4B/vae/diffusion_pytorch_model.safetensors FLUX.2-klein-4B/scheduler/scheduler_config.json \
         RIFE-safetensors/flownet.safetensors taef2/taef2.safetensors; do
  [ -f "$f" ] || { echo "MISSING: $f"; need_flux=1; }
done
if [ $need_flux = 1 ]; then
  [ -n "$CHECK_ONLY" ] && exit 1
  log "downloading FluxRT weights (~9 GB)"
  HF=.venv/bin/hf
  $HF download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B --include "scheduler/*" "vae/*" "tokenizer/*" "model_index.json"
  $HF download black-forest-labs/FLUX.2-klein-4B scheduler/scheduler_config.json --local-dir FLUX.2-klein-4B
  $HF download aydin99/FLUX.2-klein-4B-int8 --local-dir FLUX.2-klein-4B-int8
  $HF download TensorForger/RIFE-safetensors --local-dir RIFE-safetensors
  $HF download madebyollin/taef2 taef2.safetensors --local-dir taef2
fi

# ── 4. classic StreamDiffusion venv + models ────────────────────────────────
[ -d "$SD" ] || { echo "StreamDiffusion clone missing at $SD (rsync it first)"; exit 1; }
cd "$SD"
if [ ! -x .venv/bin/python ]; then
  [ -n "$CHECK_ONLY" ] && { echo "MISSING: sd venv"; exit 1; }
  log "creating StreamDiffusion venv"
  uv venv --python 3.11 .venv >/dev/null
  uv pip install --python .venv/bin/python torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python .venv/bin/python -e ".[xformers,controlnet]" peft
fi
.venv/bin/python -c "import streamdiffusion, torch; print('sd venv ok, torch', torch.__version__)"
if [ ! -d "$HF_HOME/hub/models--Lykon--dreamshaper-8" ]; then
  [ -n "$CHECK_ONLY" ] && { echo "MISSING: SD models"; exit 1; }
  log "downloading SD models (~13 GB)"
  HF=$REPO/.venv/bin/hf
  for m in Lykon/dreamshaper-8 latent-consistency/lcm-lora-sdv1-5 madebyollin/taesd depth-anything/Depth-Anything-V2-Small-hf \
           lllyasviel/control_v11f1p_sd15_depth lllyasviel/control_v11p_sd15_canny lllyasviel/control_v11f1e_sd15_tile \
           lllyasviel/control_v11p_sd15_softedge lllyasviel/control_v11p_sd15_openpose; do
    $HF download "$m" >/dev/null && log "  $m"
  done
fi

# ── 5. summary ───────────────────────────────────────────────────────────────
cd "$REPO"
log "volume usage:"; du -sh $WS/* 2>/dev/null | sort -h | tail -8
log "setup complete. run:  cd $REPO && HF_HOME=$HF_HOME .venv/bin/python scripts/serve_web.py --config configs/sd_controlnet_config.json"
