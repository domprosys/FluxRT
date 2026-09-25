#!/usr/bin/env bash
# TensorRT engine cache on a private Hugging Face repo, so a fresh pod with ENABLE_TRT=1 downloads its
# engines (~1 min for ~21 GB) instead of building them (~30 min for the multi set).
#
#   bash deploy/trt_engine_cache.sh pull   # before the server starts: fetch this GPU's engines
#   bash deploy/trt_engine_cache.sh push   # after a build: upload this GPU's engines (only new files are sent)
#
# Env: TRT_CACHE_REPO (e.g. <hf-user>/fluxrt-trt-engines, a private model repo), HF_TOKEN (read for pull,
# write for push), WS (default /workspace), VENVS (default /root/venvs).
# Engines only work on the GPU architecture and TensorRT version that built them, so they live under the
# same "<sm..>-trt<version>" folder name that sd_worker.py uses below $WS/engines.
set -euo pipefail
MODE=${1:?usage: $0 pull|push}
WS=${WS:-/workspace}; VENVS=${VENVS:-/root/venvs}
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1
log() { echo "[trt-cache $(date +%H:%M:%S)] $*"; }
[ -n "${TRT_CACHE_REPO:-}" ] || { log "TRT_CACHE_REPO not set: nothing to do"; exit 0; }
[ -n "${HF_TOKEN:-}" ] || { log "HF_TOKEN not set: cannot reach the private repo $TRT_CACHE_REPO"; exit 0; }

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')
TRT=$("$VENVS/sd/bin/python" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null) \
  || { log "TensorRT not installed in $VENVS/sd (WITH_TRT=1 first)"; exit 0; }
TAG="sm${CC}-trt${TRT}"
DIR="$WS/engines/$TAG"
mkdir -p "$DIR"

case "$MODE" in
  pull)
    t0=$(date +%s)
    if hf download "$TRT_CACHE_REPO" --repo-type model --include "$TAG/*" --local-dir "$WS/engines" >/dev/null; then
      log "pulled $(find "$DIR" -name '*.engine' | wc -l) engines for $TAG ($(du -sh "$DIR" | cut -f1)) in $(( $(date +%s) - t0 ))s"
    else
      log "no cached engines for $TAG yet (or the repo is unreachable): they will be built on first start"
    fi
    ;;
  push)
    n=$(find "$DIR" -name '*.engine' | wc -l)
    [ "$n" -gt 0 ] || { log "no engines in $DIR"; exit 0; }
    t0=$(date +%s)
    # skip lock files and per-build scratch; the engine dirs carry their own onnx-free layout
    hf upload "$TRT_CACHE_REPO" "$DIR" "$TAG" --repo-type model --exclude "*.lock" "*.onnx" "*.onnx.data" \
      --commit-message "engines for $TAG" >/dev/null
    log "pushed $n engines for $TAG ($(du -sh "$DIR" | cut -f1)) in $(( $(date +%s) - t0 ))s"
    ;;
  *) echo "usage: $0 pull|push" >&2; exit 2 ;;
esac
