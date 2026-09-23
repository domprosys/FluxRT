#!/usr/bin/env bash
# Bootstrap a RunPod pod whose network volume is mounted at /workspace.
#
# Layout (why): the network volume is fast for big sequential files (weights,
# HF cache: ~450 MB/s) but very slow for the thousands of small files a Python
# venv import touches (cold `import torch` took many minutes). So:
#   /workspace/fluxrt, /workspace/StreamDiffusion-daydream   code (git)
#   /workspace/hf, /workspace/fluxrt/FLUX.2-*, ...            weights (volume)
#   /workspace/venvs/venvs.tar                                 venv snapshot (volume)
#   /root/venvs/{fluxrt,sd} + /root/uvpython                   venvs (container disk,
#                                   restored from the tarball at every pod start)
#   <repo>/.venv -> /root/venvs/fluxrt  (symlink, so configs/scripts are unchanged)
#
# Idempotent: re-running only fills in what is missing.
#   bash /workspace/fluxrt/deploy/runpod_setup.sh            # full setup / restore
#   bash /workspace/fluxrt/deploy/runpod_setup.sh --check    # verify only (no builds)
#   bash /workspace/fluxrt/deploy/runpod_setup.sh --snapshot # re-create venvs.tar
set -euo pipefail

WS=/workspace
REPO=$WS/fluxrt
SD=$WS/StreamDiffusion-daydream
VENVS=/root/venvs
export UV_PYTHON_INSTALL_DIR=/root/uvpython
export HF_HOME=$WS/hf
export UV_CACHE_DIR=$WS/.uv-cache
export PATH="$HOME/.local/bin:$PATH"
MODE=${1:-}

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
if [ ! -d "$REPO/.git" ]; then
  [ "$MODE" = "--check" ] && { echo "MISSING: repo"; exit 1; }
  log "cloning FluxRT fork (installation-controls)"
  git clone -q --branch installation-controls https://github.com/domprosys/FluxRT.git "$REPO"
fi
if [ ! -d "$SD" ]; then
  [ "$MODE" = "--check" ] && { echo "MISSING: StreamDiffusion clone"; exit 1; }
  log "cloning daydreamlive/StreamDiffusion"
  git clone -q --depth 1 https://github.com/daydreamlive/StreamDiffusion.git "$SD"
fi

# ── 2. venvs: restore snapshot from the volume, else build on local disk ─────
if [ "$MODE" != "--snapshot" ] && [ ! -x $VENVS/fluxrt/bin/python ]; then
  if [ -f $WS/venvs/venvs.tar ]; then
    log "restoring venvs from $WS/venvs/venvs.tar ($(du -h $WS/venvs/venvs.tar | cut -f1))"
    tar -xf $WS/venvs/venvs.tar -C /
  elif [ "$MODE" = "--check" ]; then
    echo "MISSING: venvs (no snapshot on volume)"; exit 1
  fi
fi
mkdir -p $VENVS
if [ ! -x $VENVS/fluxrt/bin/python ]; then
  log "building FluxRT venv on local disk"
  uv venv --python 3.12 $VENVS/fluxrt >/dev/null
  uv pip install --python $VENVS/fluxrt/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python $VENVS/fluxrt/bin/python -r $REPO/requirements.txt
  uv pip install --python $VENVS/fluxrt/bin/python -e $REPO
  NEW_VENV=1
fi
if [ ! -x $VENVS/sd/bin/python ]; then
  log "building StreamDiffusion venv on local disk"
  uv venv --python 3.11 $VENVS/sd >/dev/null
  uv pip install --python $VENVS/sd/bin/python torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  (cd $SD && uv pip install --python $VENVS/sd/bin/python -e ".[xformers,controlnet]" peft "mediapipe==0.10.21")
  NEW_VENV=1
fi
# symlinks so the repo layout (and the configs' ../StreamDiffusion-daydream/.venv) keep working
for pair in "$REPO/.venv:$VENVS/fluxrt" "$SD/.venv:$VENVS/sd"; do
  link=${pair%%:*}; target=${pair##*:}
  if [ -d "$link" ] && [ ! -L "$link" ]; then log "removing old on-volume venv $link"; rm -rf "$link"; fi
  [ -L "$link" ] || ln -s "$target" "$link"
done
$VENVS/fluxrt/bin/python -c "import torch, aiortc, fluxrt; print('fluxrt venv ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$VENVS/sd/bin/python -c "import streamdiffusion, torch, mediapipe; print('sd venv ok, torch', torch.__version__)"
if [ "${NEW_VENV:-}" = 1 ] || [ "$MODE" = "--snapshot" ] || [ ! -f $WS/venvs/venvs.tar ]; then
  log "snapshotting venvs + interpreters to $WS/venvs/venvs.tar"
  mkdir -p $WS/venvs
  tar -cf $WS/venvs/venvs.tar.tmp -C / root/venvs root/uvpython && mv $WS/venvs/venvs.tar.tmp $WS/venvs/venvs.tar
  log "snapshot: $(du -h $WS/venvs/venvs.tar | cut -f1)"
fi

# ── 3. FluxRT models (GPU path: int8 transformer + int8 text encoder) ───────
cd "$REPO"
need_flux=0
for f in FLUX.2-klein-4B-int8/diffusion_pytorch_model.safetensors FLUX.2-klein-4B-int8/text_encoder/model.safetensors \
         FLUX.2-klein-4B/vae/diffusion_pytorch_model.safetensors FLUX.2-klein-4B/scheduler/scheduler_config.json \
         RIFE-safetensors/flownet.safetensors taef2/taef2.safetensors; do
  [ -f "$f" ] || { echo "MISSING: $f"; need_flux=1; }
done
if [ $need_flux = 1 ]; then
  [ "$MODE" = "--check" ] && exit 1
  log "downloading FluxRT weights (~9 GB)"
  HF=$VENVS/fluxrt/bin/hf
  $HF download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B --include "scheduler/*" "vae/*" "tokenizer/*" "model_index.json"
  $HF download black-forest-labs/FLUX.2-klein-4B scheduler/scheduler_config.json --local-dir FLUX.2-klein-4B
  $HF download aydin99/FLUX.2-klein-4B-int8 --local-dir FLUX.2-klein-4B-int8
  $HF download TensorForger/RIFE-safetensors --local-dir RIFE-safetensors
  $HF download madebyollin/taef2 taef2.safetensors --local-dir taef2
fi

# ── 4. SD / ControlNet models (HF cache on the volume) ──────────────────────
if [ ! -d "$HF_HOME/hub/models--lllyasviel--control_v11p_sd15_openpose" ]; then
  [ "$MODE" = "--check" ] && { echo "MISSING: SD models"; exit 1; }
  log "downloading SD models (~13 GB)"
  HF=$VENVS/fluxrt/bin/hf
  for m in Lykon/dreamshaper-8 latent-consistency/lcm-lora-sdv1-5 madebyollin/taesd depth-anything/Depth-Anything-V2-Small-hf \
           lllyasviel/control_v11f1p_sd15_depth lllyasviel/control_v11p_sd15_canny lllyasviel/control_v11f1e_sd15_tile \
           lllyasviel/control_v11p_sd15_softedge lllyasviel/control_v11p_sd15_openpose; do
    $HF download "$m" >/dev/null && log "  $m"
  done
fi

# ── 5. summary ───────────────────────────────────────────────────────────────
log "volume usage:"; du -sh $WS/* 2>/dev/null | sort -h | tail -8
log "setup complete. run:  cd $REPO && HF_HOME=$HF_HOME .venv/bin/python scripts/serve_web.py --config configs/sd_controlnet_config.json"
