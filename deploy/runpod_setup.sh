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

WS=${WS:-/workspace}
REPO=$WS/fluxrt
SD=$WS/StreamDiffusion-daydream
SDV2=$WS/StreamDiffusionV2
WHEELS=$WS/wheels      # optional wheelhouse (diffusers fork, flash-attn) for hosts with flaky GitHub
LOCKS=$REPO/deploy/locks
VENVS=${VENVS:-/root/venvs}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/root/uvpython}
export HF_HOME=${HF_HOME:-$WS/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR_POD:-/root/.uv-cache}   # local disk: /workspace may be a network fs (slow small files)
export PATH="$HOME/.local/bin:$PATH"
MODE=${1:-}
SKIP_SD=${SKIP_SD:-0}   # SKIP_SD=1: FluxRT only (no StreamDiffusion venv/models)
SKIP_SDV2=${SKIP_SDV2:-0}  # SKIP_SDV2=1: no StreamDiffusionV2 venv/models
SKIP_FLUXRT_WEIGHTS=${SKIP_FLUXRT_WEIGHTS:-0}  # 1: server venv only, no FLUX weights (SD/SDV2-only pods)
WITH_BF16=${WITH_BF16:-0}  # 1: also fetch the bf16 FLUX transformer + text encoder (web_bf16_config)
SDV2_REPO=https://github.com/chenfengxu714/StreamDiffusionV2.git; SDV2_REF=bcb894b
SD_DIFFUSERS="diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922"
# snapshots only make sense on a network volume (it outlives the pod)
if mount | grep -E " $WS " | grep -qiE "mfs|nfs|fuse"; then NETVOL=1; else NETVOL=0; fi
retry() { local n; for n in 1 2 3; do "$@" && return 0; log "attempt $n failed: $*"; sleep 10; done; return 1; }

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
  retry git clone -q --branch installation-controls https://github.com/domprosys/FluxRT.git "$REPO"
fi
if [ "$SKIP_SD" != 1 ] && [ ! -d "$SD" ]; then
  [ "$MODE" = "--check" ] && { echo "MISSING: StreamDiffusion clone"; exit 1; }
  log "cloning daydreamlive/StreamDiffusion"
  retry git clone -q --depth 1 https://github.com/daydreamlive/StreamDiffusion.git "$SD"
fi
if [ "$SKIP_SDV2" != 1 ] && [ ! -d "$SDV2" ]; then
  [ "$MODE" = "--check" ] && { echo "MISSING: StreamDiffusionV2 clone"; exit 1; }
  log "cloning StreamDiffusionV2 @ $SDV2_REF"
  rm -rf "$SDV2.tmp"; retry git clone -q "$SDV2_REPO" "$SDV2.tmp" && git -C "$SDV2.tmp" checkout -q "$SDV2_REF" && mv "$SDV2.tmp" "$SDV2"
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
  retry uv pip install --python $VENVS/fluxrt/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python $VENVS/fluxrt/bin/python -r $REPO/requirements.txt
  uv pip install --python $VENVS/fluxrt/bin/python -e $REPO
  NEW_VENV=1
fi
if [ "$SKIP_SD" != 1 ] && [ ! -x $VENVS/sd/bin/python ]; then
  log "building StreamDiffusion venv on local disk"
  uv venv --python 3.11 $VENVS/sd >/dev/null
  retry uv pip install --python $VENVS/sd/bin/python torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  DIFF_WHL=$(ls $WHEELS/diffusers-*.whl 2>/dev/null | head -1 || true)
  if [ -f $LOCKS/sd.lock ]; then
    DIFF_WHL=${DIFF_WHL:-$SD_DIFFUSERS}
    log "SD venv from lock + local diffusers wheel"
    # the lock is a full freeze of a working venv; install it verbatim (--no-deps), because its
    # metadata isn't self-consistent (mediapipe pins protobuf below what onnx declares)
    retry uv pip install --python $VENVS/sd/bin/python --no-deps "$DIFF_WHL" -r $LOCKS/sd.lock
    retry uv pip install --python $VENVS/sd/bin/python --no-deps -e $SD
  else
    (cd $SD && retry uv pip install --python $VENVS/sd/bin/python -e ".[xformers,controlnet]" peft "mediapipe==0.10.21")
  fi
  NEW_VENV=1
fi
# symlinks so the repo layout (and the configs' ../StreamDiffusion-daydream/.venv) keep working
if [ "$SKIP_SDV2" != 1 ] && [ -d "$SDV2" ] && [ ! -x $VENVS/sdv2/bin/python ]; then
  log "building StreamDiffusionV2 venv on local disk"
  uv venv --python 3.10 $VENVS/sdv2 >/dev/null
  retry uv pip install --python $VENVS/sdv2/bin/python torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  retry uv pip install --python $VENVS/sdv2/bin/python --no-deps -r $LOCKS/sdv2.lock
  FA_WHL=$(ls $WHEELS/flash_attn-2.7.4*cp310*.whl 2>/dev/null | head -1 || true)
  FA_WHL=${FA_WHL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}
  retry uv pip install --python $VENVS/sdv2/bin/python --no-deps "$FA_WHL"
  retry uv pip install --python $VENVS/sdv2/bin/python --no-deps -e $SDV2 --config-settings editable_mode=compat
  NEW_VENV=1
fi
pairs="$REPO/.venv:$VENVS/fluxrt"; [ "$SKIP_SD" != 1 ] && pairs="$pairs $SD/.venv:$VENVS/sd"
[ "$SKIP_SDV2" != 1 ] && [ -d "$SDV2" ] && pairs="$pairs $SDV2/.venv:$VENVS/sdv2"
for pair in $pairs; do
  link=${pair%%:*}; target=${pair##*:}
  if [ -d "$link" ] && [ ! -L "$link" ]; then log "removing old on-volume venv $link"; rm -rf "$link"; fi
  [ -L "$link" ] || ln -s "$target" "$link"
done
$VENVS/fluxrt/bin/python -c "import torch, aiortc, fluxrt; print('fluxrt venv ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
{ [ "$SKIP_SDV2" = 1 ] || [ ! -d "$SDV2" ]; } || $VENVS/sdv2/bin/python -c "import torch, flash_attn, causvid; print('sdv2 venv ok, torch', torch.__version__)"
[ "$SKIP_SD" = 1 ] || $VENVS/sd/bin/python -c "import streamdiffusion, torch, mediapipe; print('sd venv ok, torch', torch.__version__)"
if [ "$MODE" = "--snapshot" ] || { [ -f $WS/venvs/venvs.tar ] && [ "${NEW_VENV:-}" = 1 ] && [ "$SKIP_SD" != 1 ] && [ "$SKIP_SDV2" != 1 ]; }; then
  log "snapshotting venvs + interpreters to $WS/venvs/venvs.tar"
  mkdir -p $WS/venvs
  tar -cf $WS/venvs/venvs.tar.tmp -C / root/venvs root/uvpython && mv $WS/venvs/venvs.tar.tmp $WS/venvs/venvs.tar
  log "snapshot: $(du -h $WS/venvs/venvs.tar | cut -f1)"
fi

# ── 3. FluxRT models (GPU path: int8 transformer + int8 text encoder) ───────
cd "$REPO"
need_flux=0
if [ "$SKIP_FLUXRT_WEIGHTS" != 1 ]; then
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
if [ "$WITH_BF16" = 1 ]; then
  for sub in transformer text_encoder; do   # one pattern per call (multi-pattern --include drops files)
    [ -f FLUX.2-klein-4B/$sub/config.json ] || { [ "$MODE" = "--check" ] && { echo "MISSING: bf16 $sub"; exit 1; }; log "downloading bf16 $sub"; retry $VENVS/fluxrt/bin/hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B --include "$sub/*" >/dev/null; }
    [ -f FLUX.2-klein-4B/$sub/config.json ] || { echo "MISSING after download: bf16 $sub"; exit 1; }
  done
fi
fi  # SKIP_FLUXRT_WEIGHTS

# ── 4. SD / ControlNet models (HF cache on the volume) ──────────────────────
if [ "$SKIP_SD" != 1 ] && [ ! -d "$HF_HOME/hub/models--lllyasviel--control_v11p_sd15_openpose" ]; then
  [ "$MODE" = "--check" ] && { echo "MISSING: SD models"; exit 1; }
  log "downloading SD models (~13 GB)"
  HF=$VENVS/fluxrt/bin/hf
  for m in Lykon/dreamshaper-8 latent-consistency/lcm-lora-sdv1-5 madebyollin/taesd depth-anything/Depth-Anything-V2-Small-hf \
           lllyasviel/control_v11f1p_sd15_depth lllyasviel/control_v11p_sd15_canny lllyasviel/control_v11f1e_sd15_tile \
           lllyasviel/control_v11p_sd15_softedge lllyasviel/control_v11p_sd15_openpose; do
    retry $HF download "$m" >/dev/null && log "  $m" || { echo "MISSING after download: $m"; exit 1; }
  done
fi

# ── 4b. StreamDiffusionV2 weights (inside the SDV2 clone, where the worker expects them) ──
if [ "$SKIP_SDV2" != 1 ] && [ -d "$SDV2" ]; then
  HF=$VENVS/fluxrt/bin/hf
  [ -f $SDV2/wan_models/Wan2.1-T2V-1.3B/config.json ] || { [ "$MODE" = "--check" ] && { echo "MISSING: Wan2.1"; exit 1; }; log "downloading Wan2.1-T2V-1.3B (~17 GB)"; retry $HF download Wan-AI/Wan2.1-T2V-1.3B --local-dir $SDV2/wan_models/Wan2.1-T2V-1.3B >/dev/null; }
  [ -f $SDV2/ckpts/wan_causal_dmd_v2v/model.pt ] || { [ "$MODE" = "--check" ] && { echo "MISSING: SDV2 ckpt"; exit 1; }; log "downloading SDV2 checkpoint (~11 GB)"; retry $HF download daydreamlive/StreamDiffusionV2 wan_causal_dmd_v2v/model.pt --local-dir $SDV2/ckpts >/dev/null; }
  [ -f $SDV2/wan_models/Autoencoders/lightvaew2_1.pth ] || { [ "$MODE" = "--check" ] && { echo "MISSING: lightvae"; exit 1; }; retry $HF download lightx2v/Autoencoders lightvaew2_1.pth --local-dir $SDV2/wan_models/Autoencoders >/dev/null; }
  for f in wan_models/Wan2.1-T2V-1.3B/config.json ckpts/wan_causal_dmd_v2v/model.pt wan_models/Autoencoders/lightvaew2_1.pth; do
    [ -f "$SDV2/$f" ] || { echo "MISSING after download: $SDV2/$f"; exit 1; }
  done
fi

# ── 5. summary ───────────────────────────────────────────────────────────────
log "volume usage:"; du -sh $WS/* 2>/dev/null | sort -h | tail -8
log "setup complete. run:  cd $REPO && HF_HOME=$HF_HOME .venv/bin/python scripts/serve_web.py --config configs/sd_controlnet_config.json"
