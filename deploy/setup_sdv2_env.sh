#!/usr/bin/env bash
# StreamDiffusionV2 environment + weights for the "sdv2" backend (src/fluxrt/backends/sdv2_worker.py).
# Works on Ada (RTX 4090, sm_89) and Blackwell (RTX PRO 6000 / 5090, sm_120). Idempotent.
#
# What gets built (and why):
#   $VENVS/sdv2   python 3.10 (upstream's; its numpy==1.24.4 pin has no wheels for 3.12)
#                 torch 2.11.0 + torchvision 0.26.0 from the PyTorch cu128 index: the Blackwell pins
#                   from upstream's README; cu128 rather than PyPI's default cu130 build so hosts with
#                   570-series drivers work. The cu128 wheels carry sm_120 kernels (sm_86 ones run on sm_89).
#                 the rest of upstream's pinned deps: deploy/locks/sdv2-cu128.lock (--no-deps)
#                 upstream StreamDiffusionV2 @ 6961a5c, installed NON-editable from the clone (fast imports
#                   from local disk). Not PyPI: streamdiffusionv2 0.1.1 (2026-05-18) predates the KV
#                   ring-buffer RoPE re-alignment fix (2026-09) that long live streams need.
#                 flash-attn: NOT installed. There is no prebuilt wheel for torch 2.11 (Dao-AILab ships up
#                   to torch 2.8; PyPI has only the sdist) and upstream falls back to torch SDPA, whose
#                   built-in flash / cuDNN kernels cover sm_89 and sm_120. To try one anyway:
#                   SDV2_FLASH_ATTN_WHEEL=<path|url> (cp310, torch 2.11, cu12; dropped again if it won't import).
#   $SDV2         the upstream clone at the pinned commit (weights live inside it; .venv -> $VENVS/sdv2)
#   weights       $SDV2/wan_models/Wan2.1-T2V-1.3B/{config.json,UMT5,tokenizer,Wan2.1_VAE.pth}
#                 $SDV2/ckpts/wan_causal_dmd_v2v/model.pt            (1.3B DMD generator, 5.7 GB)
#                 $SDV2/wan_models/Autoencoders/lightvaew2_1.pth     (LightVAE, 32 MB)
#                 $SDV2/ckpts/taew2_1.pth                            (TAEHV decoder, 23 MB)
#                 SDV2_14B=1: $SDV2/ckpts/wan_causal_dmd_v2v_14b/model.pt (28.6 GB) + the 14B config.json;
#                   UMT5 / tokenizer / VAE are byte-identical to the 1.3B ones and are symlinked.
#                 The worker builds the generator from config.json and fills it from the DMD checkpoint
#                 (it holds the whole generator), so the base diffusion weights (1.3B: 5.7 GB, 14B: 57 GB)
#                 are NOT needed. SDV2_BASE_WEIGHTS=1 fetches them anyway, for upstream's own scripts
#                 (run_v2v.sh, demo/), which load them through diffusers from_pretrained.
#   Every file is fetched from a pinned revision and size-checked (a truncated download once passed).
#
# Env (runpod_setup.sh exports VENVS WS REPO SDV2 HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR):
#   SDV2_14B=1 | SDV2_BASE_WEIGHTS=1 | SDV2_TORCHAO=1 (torchao for worker.fp8) |
#   SDV2_FLASH_ATTN_WHEEL=... | SDV2_REBUILD=1 (force a venv rebuild) | SDV2_SKIP_GPU_CHECK=1
# Usage:
#   bash deploy/setup_sdv2_env.sh            # clone + venv + weights
#   bash deploy/setup_sdv2_env.sh --venv     # clone + venv only
#   bash deploy/setup_sdv2_env.sh --weights  # weights only
#   bash deploy/setup_sdv2_env.sh --check    # verify everything, build/download nothing (exit 1 if incomplete)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=${WS:-/workspace}
REPO=${REPO:-$(dirname "$SCRIPT_DIR")}
SDV2=${SDV2:-$WS/StreamDiffusionV2}
VENVS=${VENVS:-/root/venvs}
WHEELS=${WHEELS:-$WS/wheels}
REPO=$(realpath -m "$REPO"); SDV2=$(realpath -m "$SDV2"); VENVS=$(realpath -m "$VENVS"); WHEELS=$(realpath -m "$WHEELS")
export HF_HOME=${HF_HOME:-$WS/hf}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/root/uvpython}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.uv-cache}
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}  # no 10 GB xet chunk cache on the volume
MODE=${1:-}
case "$MODE" in ""|--venv|--weights|--check) ;; *) echo "usage: $0 [--venv|--weights|--check]" >&2; exit 2 ;; esac
SDV2_14B=${SDV2_14B:-0}
SDV2_BASE_WEIGHTS=${SDV2_BASE_WEIGHTS:-0}

SDV2_GIT=https://github.com/chenfengxu714/StreamDiffusionV2.git
SDV2_SHA=6961a5cf2045d1dda05a04ef229698bdc04e873a     # 2026-09-14 "Fix position refresh in streaming inference"
TORCH_PINS=(torch==2.11.0 torchvision==0.26.0)
TORCH_INDEX=https://download.pytorch.org/whl/cu128
TORCHAO_PIN=torchao==0.15.0  # newest torchao diffusers 0.35.1 can import: 0.16+ moved torchao.dtypes.* and
                             # diffusers' fallback then dies (NameError: logger). Its C++ ops may not load on
                             # torch 2.11 (warning only); the fp8 dynamic-quant path is pure torch._scaled_mm.
LOCK=$REPO/deploy/locks/sdv2-cu128.lock
PY=$VENVS/sdv2/bin/python
STAMP=$VENVS/sdv2/.fluxrt-sdv2-stamp
# HF revisions (pinned 2026-09-24) and TAEHV commit
REV_SDV2=2373eb2b39278b3a1aa174964a724ee78ead96f0     # jerryfeng/StreamDiffusionV2
REV_W13=37ec512624d61f7aa208f7ea8140a131f93afc9a      # Wan-AI/Wan2.1-T2V-1.3B
REV_W14=a064a6c71f5be440641209c07bf2a5ce7a2ff5e4      # Wan-AI/Wan2.1-T2V-14B
REV_LX=02cbfd1a0a336bbd87da49fd8cc155ed11ff123e       # lightx2v/Autoencoders
TAEHV_URL=https://raw.githubusercontent.com/madebyollin/taehv/1a88a7dcafa06f46866661c0d687654aafa5521b/taew2_1.pth

log() { echo "[sdv2-setup $(date +%H:%M:%S)] $*"; }
die() { echo "[sdv2-setup] FAILED: $*" >&2; exit 1; }
retry() { local n; for n in 1 2 3 4 5; do "$@" && return 0; log "attempt $n/5 failed: $*"; sleep $((n * 8)); done; return 1; }
is_check() { [ "$MODE" = "--check" ]; }
MISSING=0
missing() { echo "MISSING: $*"; MISSING=1; }

command -v uv >/dev/null || { is_check && die "uv not installed"; log "installing uv"; retry sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null'; }
[ -f "$LOCK" ] || die "lock file not found: $LOCK"
stamp_want() { echo "$SDV2_SHA $(sha1sum < "$LOCK" | cut -c1-12) ${TORCH_PINS[*]}"; }

# ── 1. upstream source at the pinned commit ──────────────────────────────────────────────
src_ok() { [ "$(cat "$SDV2/.fluxrt-sdv2-ref" 2>/dev/null)" = "$SDV2_SHA" ] && [ -f "$SDV2/pyproject.toml" ] && [ -f "$SDV2/models/wan/causal_model.py" ]; }
fetch_source() {
  mkdir -p "$SDV2"
  # a) git: shallow fetch of exactly the pinned commit into the existing folder. Keeps the untracked
  #    weights (wan_models/, ckpts/) and .venv; an old bcb894b checkout's tracked files are replaced.
  if [ ! -d "$SDV2/.git" ]; then git -C "$SDV2" init -q && git -C "$SDV2" remote add origin "$SDV2_GIT"; fi
  git -C "$SDV2" remote set-url origin "$SDV2_GIT"
  if retry timeout 600 git -C "$SDV2" fetch -q --depth 1 origin "$SDV2_SHA" && git -C "$SDV2" checkout -q -f "$SDV2_SHA"; then
    echo "$SDV2_SHA" > "$SDV2/.fluxrt-sdv2-ref"; return 0
  fi
  # b) codeload tarball: another GitHub endpoint, sometimes alive when git-over-https stalls
  log "git fetch failed; trying the GitHub tarball"
  local tmp; tmp=$(mktemp -d)
  if retry curl -fsSL --max-time 600 -o "$tmp/src.tgz" "https://codeload.github.com/chenfengxu714/StreamDiffusionV2/tar.gz/$SDV2_SHA" \
     && tar -xzf "$tmp/src.tgz" -C "$tmp" && [ -f "$tmp/StreamDiffusionV2-$SDV2_SHA/pyproject.toml" ]; then
    cp -a "$tmp/StreamDiffusionV2-$SDV2_SHA/." "$SDV2/"
    rm -rf "$tmp"; echo "$SDV2_SHA" > "$SDV2/.fluxrt-sdv2-ref"; return 0
  fi
  rm -rf "$tmp"
  die "could not fetch StreamDiffusionV2 @ ${SDV2_SHA:0:7} from GitHub (git and tarball). Retry later, or put a
       checkout of that commit at $SDV2 and write the SHA into $SDV2/.fluxrt-sdv2-ref"
}

# ── 2. venv ──────────────────────────────────────────────────────────────────────────────
venv_ok() {
  [ -x "$PY" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$(stamp_want)" ] || return 1
  cd /  # import the installed package, not a clone that happens to be the cwd
  "$PY" - <<'EOF' >/dev/null 2>&1
import torch, accelerate, transformers, diffusers, sentencepiece  # noqa
import models.wan.causal_stream_inference, models.wan.taehv_wrapper  # noqa
from models.wan.causal_model import KV_POS_EMPTY  # noqa: the RoPE re-alignment fix
assert torch.__version__.startswith("2.11.0") and "sm_120" in getattr(torch._C, "_cuda_getArchFlags", str)().split()
EOF
}
mkvenv() { rm -rf "$VENVS/sdv2"; mkdir -p "$VENVS"; uv venv -q --python 3.10 "$VENVS/sdv2"; }
build_venv() {
  log "building $VENVS/sdv2: python 3.10, torch 2.11.0+cu128, StreamDiffusionV2 @ ${SDV2_SHA:0:7}"
  retry mkvenv
  retry uv pip install -q --python "$PY" "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
  retry uv pip install -q --python "$PY" --no-deps -r "$LOCK"
  local fa=${SDV2_FLASH_ATTN_WHEEL:-}
  if [ -z "$fa" ]; then
    for f in "$WHEELS"/flash_attn-*torch2.11*cp310*.whl; do if [ -f "$f" ]; then fa=$f; break; fi; done
  fi
  if [ -n "$fa" ]; then
    log "flash-attn wheel: $fa"
    if ! retry uv pip install -q --python "$PY" --no-deps "$fa" || ! "$PY" -c "import flash_attn, flash_attn_2_cuda" 2>/dev/null; then
      log "WARNING: flash-attn wheel unusable with torch 2.11; removing it (SDPA fallback)"
      uv pip uninstall -q --python "$PY" flash-attn flash_attn >/dev/null 2>&1 || true
    fi
  fi
  if [ "${SDV2_TORCHAO:-0}" = 1 ]; then retry uv pip install -q --python "$PY" --no-deps "$TORCHAO_PIN"; fi
  # the upstream package itself (setuptools comes via build isolation from PyPI)
  retry uv pip install -q --python "$PY" --no-deps --reinstall-package streamdiffusionv2 "$SDV2"
  stamp_want > "$STAMP"
  VENV_BUILT=1
}
verify_venv() {
  cd /
  "$PY" - <<'EOF'
import os, re, sys
import torch
flags = getattr(torch._C, "_cuda_getArchFlags", str)().split()  # compiled-in archs, no GPU needed
print(f"sdv2 venv: python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}, archs {' '.join(flags)}")
assert "sm_120" in flags, "torch build lacks sm_120 (Blackwell)"
assert any(re.fullmatch(r"sm_8[0-9]", a) for a in flags), "torch build lacks sm_8x (Ada/Ampere)"
import numpy, diffusers, transformers, accelerate
import models.wan.causal_stream_inference, models.wan.taehv_wrapper  # noqa
import models.wan.causal_model as cm
assert hasattr(cm, "KV_POS_EMPTY"), "StreamDiffusionV2 build predates the RoPE re-alignment fix"
print(f"  numpy {numpy.__version__}, diffusers {diffusers.__version__}, transformers {transformers.__version__}, "
      f"attention: {'flash_attn' if cm.FLASH_ATTN_AVAILABLE else 'torch SDPA (no flash-attn)'}")
if torch.cuda.is_available() and os.environ.get("SDV2_SKIP_GPU_CHECK") != "1":
    maj, mnr = torch.cuda.get_device_capability(0)
    have = [a for a in flags if (m := re.fullmatch(r"sm_(\d+)", a)) and int(m[1]) // 10 == maj and int(m[1]) % 10 <= mnr]
    assert have, f"no kernels for this GPU (sm_{maj}{mnr})"
    import torch.nn.functional as F
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(2, 12, 1024, 128, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum() + F.scaled_dot_product_attention(q, q, q).float().sum()
    c = torch.nn.Conv3d(16, 16, 3, padding=1).cuda().to(torch.bfloat16)(torch.randn(1, 16, 2, 32, 32, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    print(f"  GPU smoke test ok on {torch.cuda.get_device_name(0)} (sm_{maj}{mnr}, kernels {have[-1]})")
EOF
}

# ── 3. weights ───────────────────────────────────────────────────────────────────────────
HF=""
find_hf() {
  local c
  for c in "$VENVS/fluxrt/bin/hf" "$PY_HF" "$(command -v hf 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then HF=$c; return 0; fi
  done
  die "no 'hf' CLI found (huggingface_hub >= 0.34; the sdv2 venv has one once built)"
}
PY_HF=$VENVS/sdv2/bin/hf
size_ok() { local f=$1 sizes=$2 s; [ -f "$f" ] || return 1; s=$(stat -Lc %s "$f"); [[ ",$sizes," == *",$s,"* ]]; }
first_size() { echo "${1%%,*}"; }
NEED_BYTES=0
plan() {  # dest sizes: account a missing/bad file
  size_ok "$1" "$2" && return 0
  if is_check; then missing "$1"; else NEED_BYTES=$((NEED_BYTES + $(first_size "$2"))); fi
}
fetch_hf() {  # local_dir repo repo_path revision accepted_sizes
  local dir=$1 repo=$2 path=$3 rev=$4 sizes=$5 dest="$1/$3" try extra
  size_ok "$dest" "$sizes" && return 0
  is_check && return 0
  for try in 1 2; do
    extra=()
    if [ "$try" = 2 ]; then extra=(--force-download); fi
    if [ -e "$dest" ]; then
      log "  bad size for $dest ($(stat -Lc %s "$dest") B, want $sizes): re-downloading"
      rm -f "$dest" "$dir/.cache/huggingface/download/$path.metadata"
    fi
    log "  $repo/$path -> $dir"
    retry "$HF" download "$repo" "$path" --revision "$rev" --local-dir "$dir" "${extra[@]}" >/dev/null || true
    size_ok "$dest" "$sizes" && return 0
  done
  die "download of $repo/$path did not produce a file of size $sizes: $dest"
}
fetch_url() {  # dest url accepted_sizes [hf fallback: dir repo path rev]
  local dest=$1 url=$2 sizes=$3
  size_ok "$dest" "$sizes" && return 0
  is_check && return 0
  mkdir -p "$(dirname "$dest")"
  log "  $url -> $dest"
  if retry curl -fsSL --max-time 300 -o "$dest.part" "$url" && size_ok "$dest.part" "$sizes"; then mv "$dest.part" "$dest"; return 0; fi
  rm -f "$dest.part"
  if [ $# -ge 7 ]; then
    log "  GitHub download failed; using the HF mirror $5/$6"
    fetch_hf "$4" "$5" "$6" "$7" "$sizes"
    [ "$4/$6" = "$dest" ] || cp -f "$4/$6" "$dest"
    size_ok "$dest" "$sizes" && return 0
  fi
  die "download of $url did not produce a file of size $sizes: $dest"
}
link_shared() {  # 14B folder: symlink the files shared with the 1.3B
  local d=$SDV2/wan_models/Wan2.1-T2V-14B f
  mkdir -p "$d/google"
  for f in models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth google/umt5-xxl; do
    [ -e "$d/$f" ] && [ ! -L "$d/$f" ] && continue   # a real copy (e.g. full repo download) is fine too
    ln -sfn "$( [ "$f" = google/umt5-xxl ] && echo ../.. || echo .. )/Wan2.1-T2V-1.3B/$f" "$d/$f"
  done
}

W13=$SDV2/wan_models/Wan2.1-T2V-1.3B
W14=$SDV2/wan_models/Wan2.1-T2V-14B
AE=$SDV2/wan_models/Autoencoders
# sizes from the HF API at the pinned revisions; the old daydreamlive 1.3B checkpoint (generator + critic,
# 11.35 GB, still on some volumes) and daydreamlive's 14B copy are accepted as equivalent
S_CFG13=249; S_CFG14=250; S_T5=11361920418; S_VAE=507609880; S_LVAE=32208043
S_TOK_SPM=4548313; S_TOK_JSON=16837417; S_TOK_CFG=61728; S_TOK_MAP=6623
S_CK13=5676282858,11352649716; S_CK14=28577332702,28577335997; S_TAEHV=22678901,22679486
S_BASE13=5676070424
BASE14=(diffusion_pytorch_model.safetensors.index.json:96805
        diffusion_pytorch_model-00001-of-00006.safetensors:9887603256 diffusion_pytorch_model-00002-of-00006.safetensors:9839059648
        diffusion_pytorch_model-00003-of-00006.safetensors:9839059744 diffusion_pytorch_model-00004-of-00006.safetensors:9839059744
        diffusion_pytorch_model-00005-of-00006.safetensors:9839059744 diffusion_pytorch_model-00006-of-00006.safetensors:7910235256)
TOK=("google/umt5-xxl/spiece.model:$S_TOK_SPM" "google/umt5-xxl/tokenizer.json:$S_TOK_JSON"
     "google/umt5-xxl/tokenizer_config.json:$S_TOK_CFG" "google/umt5-xxl/special_tokens_map.json:$S_TOK_MAP")

weights() {
  # space check first (only what is missing)
  plan "$W13/config.json" $S_CFG13; plan "$W13/models_t5_umt5-xxl-enc-bf16.pth" $S_T5; plan "$W13/Wan2.1_VAE.pth" $S_VAE
  local e; for e in "${TOK[@]}"; do plan "$W13/${e%%:*}" "${e##*:}"; done
  plan "$SDV2/ckpts/wan_causal_dmd_v2v/model.pt" $S_CK13; plan "$AE/lightvaew2_1.pth" $S_LVAE; plan "$SDV2/ckpts/taew2_1.pth" $S_TAEHV
  [ "$SDV2_BASE_WEIGHTS" = 1 ] && plan "$W13/diffusion_pytorch_model.safetensors" $S_BASE13
  if [ "$SDV2_14B" = 1 ]; then
    plan "$W14/config.json" $S_CFG14; plan "$SDV2/ckpts/wan_causal_dmd_v2v_14b/model.pt" $S_CK14
    [ "$SDV2_BASE_WEIGHTS" = 1 ] && for e in "${BASE14[@]}"; do plan "$W14/${e%%:*}" "${e##*:}"; done
  fi
  if is_check; then
    [ "$SDV2_14B" = 1 ] && for e in models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth google/umt5-xxl/tokenizer.json; do
      [ -e "$W14/$e" ] || missing "$W14/$e (symlink to the 1.3B copy)"; done
    return 0
  fi
  mkdir -p "$W13" "$AE" "$SDV2/ckpts"
  if [ "$NEED_BYTES" -gt 0 ]; then
    local avail; avail=$(df -B1 --output=avail "$SDV2" | tail -1 | tr -d ' ')
    log "weights to download: $((NEED_BYTES / 1000000000)) GB (free on $(df --output=target "$SDV2" | tail -1): $((avail / 1000000000)) GB)"
    [ "$avail" -gt $((NEED_BYTES + 2000000000)) ] || die "not enough disk space for the SDV2 weights"
  fi
  find_hf
  fetch_hf "$W13" Wan-AI/Wan2.1-T2V-1.3B config.json $REV_W13 $S_CFG13
  for e in "${TOK[@]}"; do fetch_hf "$W13" Wan-AI/Wan2.1-T2V-1.3B "${e%%:*}" $REV_W13 "${e##*:}"; done
  fetch_hf "$W13" Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth $REV_W13 $S_VAE
  fetch_hf "$W13" Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth $REV_W13 $S_T5
  fetch_hf "$SDV2/ckpts" jerryfeng/StreamDiffusionV2 wan_causal_dmd_v2v/model.pt $REV_SDV2 $S_CK13
  fetch_hf "$AE" lightx2v/Autoencoders lightvaew2_1.pth $REV_LX $S_LVAE
  fetch_url "$SDV2/ckpts/taew2_1.pth" "$TAEHV_URL" $S_TAEHV "$AE" lightx2v/Autoencoders taew2_1.pth $REV_LX
  [ "$SDV2_BASE_WEIGHTS" = 1 ] && fetch_hf "$W13" Wan-AI/Wan2.1-T2V-1.3B diffusion_pytorch_model.safetensors $REV_W13 $S_BASE13
  if [ "$SDV2_14B" = 1 ]; then
    mkdir -p "$W14"
    fetch_hf "$W14" Wan-AI/Wan2.1-T2V-14B config.json $REV_W14 $S_CFG14
    link_shared
    fetch_hf "$SDV2/ckpts" jerryfeng/StreamDiffusionV2 wan_causal_dmd_v2v_14b/model.pt $REV_SDV2 $S_CK14
    if [ "$SDV2_BASE_WEIGHTS" = 1 ]; then
      for e in "${BASE14[@]}"; do fetch_hf "$W14" Wan-AI/Wan2.1-T2V-14B "${e%%:*}" $REV_W14 "${e##*:}"; done
    fi
  fi
  log "weights ok: $(du -sh "$SDV2/wan_models" "$SDV2/ckpts" 2>/dev/null | awk '{printf "%s %s  ", $1, $2}')"
}

# ── main ─────────────────────────────────────────────────────────────────────────────────
VENV_BUILT=0
if [ "$MODE" != "--weights" ]; then
  if ! src_ok; then
    if is_check; then missing "StreamDiffusionV2 @ ${SDV2_SHA:0:7} at $SDV2"
    else log "fetching StreamDiffusionV2 @ ${SDV2_SHA:0:7} -> $SDV2"; fetch_source; fi
  fi
  if [ "${SDV2_REBUILD:-0}" = 1 ] || ! venv_ok; then
    if is_check; then missing "sdv2 venv ($VENVS/sdv2: torch 2.11 cu128 + StreamDiffusionV2 @ ${SDV2_SHA:0:7})"
    else build_venv; fi
  fi
  if ! is_check; then
    if [ -d "$SDV2/.venv" ] && [ ! -L "$SDV2/.venv" ]; then log "removing old on-volume venv $SDV2/.venv"; rm -rf "$SDV2/.venv"; fi
    ln -sfn "$VENVS/sdv2" "$SDV2/.venv"
    verify_venv || die "sdv2 venv verification failed"
  elif [ -x "$PY" ]; then
    SDV2_SKIP_GPU_CHECK=1 verify_venv || missing "working sdv2 venv"
  fi
fi
if [ "$MODE" != "--venv" ]; then
  weights
fi
if is_check; then
  if [ "$MISSING" = 0 ]; then log "check ok"; else log "check: incomplete"; exit 1; fi
else
  [ "$VENV_BUILT" = 1 ] && log "NOTE: the sdv2 venv was rebuilt; re-snapshot venvs if you keep a venvs.tar"
  log "done. test: cd $REPO && .venv/bin/python scripts/test_backend.py --config configs/sdv2_config.json --device -1 --seconds 30"
fi
