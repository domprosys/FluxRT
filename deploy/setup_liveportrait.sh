#!/usr/bin/env bash
# LivePortrait add-on for FluxRT: lip / expression transfer from the webcam onto the generated frame
# (config block "lip_transfer", e.g. configs/web_bf16_liveportrait_config.json).
#
# Run by deploy/runpod_setup.sh when WITH_LIVEPORTRAIT=1 (after the venvs exist), or by hand:
#   bash /workspace/fluxrt/deploy/setup_liveportrait.sh
# Env (exported by runpod_setup.sh; defaults for manual runs): VENVS (/root/venvs), WS (/workspace),
# REPO ($WS/fluxrt), HF_HOME ($WS/hf), UV_CACHE_DIR. Optional:
#   LP_MODELS_DIR     where the weights go (default $WS/models/liveportrait; on the volume if there is one)
#   LP_SKIP_SELFTEST=1  skip the GPU self-test at the end
#   LP_STRICT=1       also fail when onnxruntime cannot use CUDA for the face detector (CPU fallback)
#
# Idempotent; each step only fills in what is missing:
#  1. deps (requirements_lipsync.txt) into $VENVS/fluxrt. The venv's current packages are passed as
#     constraints, so torch/numpy/opencv/... are never changed (a conflict fails loudly instead), and
#     deploy/locks/liveportrait.lock pins the packages that are new. onnxruntime-gpu 1.26.0 is the last
#     CUDA-12 build; with torch imported first it reuses torch's CUDA/cuDNN libraries.
#  2. LivePortrait code at a pinned commit into $REPO/LivePortrait-code (where src/fluxrt/__init__.py
#     looks for it): src/ + LICENSE + 2 sample images, via a sparse shallow git fetch (~3 MB);
#     fallbacks: $WS/wheels/LivePortrait-<sha>.tar.gz, then the GitHub codeload tarball.
#  3. human-face weights (~660 MB) from Hugging Face KlingTeam/LivePortrait (ex KwaiVGI) at a pinned
#     revision into $LP_MODELS_DIR/{liveportrait,insightface}; $REPO/LivePortrait -> $LP_MODELS_DIR, so
#     the config's "models_dir": "LivePortrait/liveportrait" works.
#  4. import check, then a GPU self-test (loads the processor from the config, re-animates a sample
#     face, prints ms/frame; writes $WS/liveportrait_selftest.{jpg,log}).
set -euo pipefail

WS=${WS:-/workspace}
REPO=${REPO:-$WS/fluxrt}
VENVS=${VENVS:-/root/venvs}
PY=$VENVS/fluxrt/bin/python
export HF_HOME=${HF_HOME:-$WS/hf}
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH="$HOME/.local/bin:$PATH"
LP_MODELS_DIR=${LP_MODELS_DIR:-$WS/models/liveportrait}
LP_SHA=9b294b3d0536135442ea73cb01e6cb3ca7029dd3   # KlingAIResearch/LivePortrait HEAD 2026-06 (src/ unchanged since 6c4a883, 2025-02-02)
LP_GIT_URLS="https://github.com/KlingAIResearch/LivePortrait.git https://github.com/KwaiVGI/LivePortrait.git"
HF_REPOS="KlingTeam/LivePortrait KwaiVGI/LivePortrait"   # KwaiVGI/LivePortrait redirects to KlingTeam
HF_REV=82a4fa6735ca58432b6ce39301b4b9ee066dea47
LP_CODE=$REPO/LivePortrait-code
LOCK=$REPO/deploy/locks/liveportrait.lock
CONFIG=$REPO/configs/web_bf16_liveportrait_config.json

log() { echo "[liveportrait $(date +%H:%M:%S)] $*"; }
die() { echo "[liveportrait] FAILED: $*" >&2; exit 1; }
retry() { local n; for n in 1 2 3 4; do "$@" && return 0; log "attempt $n failed: $*"; sleep $((n * 10)); done; return 1; }

[ -x "$PY" ] || die "FluxRT venv python not found at $PY (run deploy/runpod_setup.sh first)"
[ -d "$REPO/src/fluxrt" ] || die "FluxRT repo not found at $REPO"
command -v uv >/dev/null || die "uv not on PATH"
TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT

# ── 1. python deps ───────────────────────────────────────────────────────────
deps_ok() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import importlib.metadata as md
import onnxruntime, onnx, skimage, scipy, imageio, yaml, rich, requests, tqdm  # noqa: F401
assert md.version("onnxruntime-gpu") == "1.26.0"
try:
    md.version("onnxruntime")
    raise SystemExit(1)  # CPU build installed too: same module name, clobbers the GPU one
except md.PackageNotFoundError:
    pass
EOF
}
if deps_ok; then
  log "deps: already installed"
else
  log "deps: installing requirements_lipsync.txt into $VENVS/fluxrt"
  if "$PY" -c "import importlib.metadata as m; m.version('onnxruntime')" >/dev/null 2>&1; then
    log "removing the CPU onnxruntime package (clashes with onnxruntime-gpu)"
    uv pip uninstall --python "$PY" onnxruntime onnxruntime-gpu || true
  fi
  pip_check() { uv pip check --python "$PY" 2>&1 | grep -vE '^(Checked|Using Python|All installed)' || true; }
  pip_check > "$TMPD/check.before"
  uv pip freeze --python "$PY" | grep -vE '^(-e |#)|@ ' > "$TMPD/freeze"
  # constraints = the venv as it is (minus the packages this add-on owns) + lock pins for new packages
  "$PY" - "$TMPD/freeze" "$LOCK" > "$TMPD/constraints" <<'EOF'
import re, sys
norm = lambda s: re.sub(r"[-_.]+", "-", s.strip().lower())
owned = {"onnxruntime", "onnxruntime-gpu", "onnx", "scikit-image", "imageio"}
def reqs(path):
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if "==" in line:
            out[norm(line.split("==")[0])] = line
    return out
have, lock = reqs(sys.argv[1]), reqs(sys.argv[2])
for name, line in have.items():
    if name not in owned:
        print(line)
for name, line in lock.items():
    if name in owned or name not in have:
        print(line)
EOF
  retry uv pip install --python "$PY" -r "$REPO/requirements_lipsync.txt" -c "$TMPD/constraints" \
    || die "dependency install failed (a conflict with the existing FluxRT venv is reported above)"
  deps_ok || die "deps installed but the import check fails: $("$PY" -c 'import onnxruntime, onnx, skimage, scipy, imageio' 2>&1 | tail -1)"
  pip_check > "$TMPD/check.after"
  if ! diff -q "$TMPD/check.before" "$TMPD/check.after" >/dev/null; then
    log "WARNING: 'uv pip check' changed after the install:"; diff "$TMPD/check.before" "$TMPD/check.after" || true
  fi
fi

# ── 2. LivePortrait code (pinned) ────────────────────────────────────────────
code_ok() {
  [ "$(cat "$LP_CODE/.fluxrt-sha" 2>/dev/null)" = "$LP_SHA" ] &&
  for f in src/__init__.py src/live_portrait_wrapper.py src/config/models.yaml src/utils/resources/lip_array.pkl \
           src/utils/resources/mask_template.png src/utils/dependencies/insightface/app/face_analysis.py; do
    [ -f "$LP_CODE/$f" ] || return 1
  done
}
extract_tarball() {  # $1 = codeload-style tar.gz of the repo at $LP_SHA
  rm -rf "$TMPD/code" && mkdir -p "$TMPD/code" &&
  tar -xzf "$1" -C "$TMPD/code" --strip-components=1 --wildcards \
      '*/src/*' '*/LICENSE' '*/assets/examples/source/s9.jpg' '*/assets/examples/driving/d19.jpg'
}
git_fetch() {  # $1 = repo url: shallow, blobless fetch of one commit, sparse checkout of what we need
  rm -rf "$TMPD/code" && git init -q "$TMPD/code" &&
  git -C "$TMPD/code" remote add origin "$1" &&
  git -C "$TMPD/code" config core.sparseCheckout true &&
  printf '/src/\n/LICENSE\n/assets/examples/source/s9.jpg\n/assets/examples/driving/d19.jpg\n' > "$TMPD/code/.git/info/sparse-checkout" &&
  timeout 300 git -C "$TMPD/code" fetch -q --depth 1 --filter=blob:none origin "$LP_SHA" &&
  timeout 300 git -C "$TMPD/code" checkout -q FETCH_HEAD &&
  rm -rf "$TMPD/code/.git"
}
if code_ok; then
  log "code: LivePortrait-code @ ${LP_SHA:0:7} present"
else
  log "code: fetching LivePortrait @ ${LP_SHA:0:7}"
  got=0
  if [ -f "$WS/wheels/LivePortrait-$LP_SHA.tar.gz" ] && extract_tarball "$WS/wheels/LivePortrait-$LP_SHA.tar.gz"; then
    got=1; log "code: from $WS/wheels"
  fi
  if [ $got = 0 ]; then
    for url in $LP_GIT_URLS; do retry git_fetch "$url" && { got=1; break; }; done
  fi
  if [ $got = 0 ]; then
    log "code: git failed, trying the codeload tarball"
    for name in KlingAIResearch KwaiVGI; do
      if curl -fsSL --retry 4 --retry-delay 10 --connect-timeout 20 --max-time 600 \
           -o "$TMPD/lp.tar.gz" "https://codeload.github.com/$name/LivePortrait/tar.gz/$LP_SHA" &&
         extract_tarball "$TMPD/lp.tar.gz"; then got=1; break; fi
    done
  fi
  [ $got = 1 ] || die "could not fetch the LivePortrait code (GitHub unreachable?). Put the tarball of
  https://codeload.github.com/KlingAIResearch/LivePortrait/tar.gz/$LP_SHA at $WS/wheels/LivePortrait-$LP_SHA.tar.gz and re-run."
  # LivePortrait's src/ has no __init__.py; make it a regular package so fluxrt's `import src`
  # (aliased to `liveportrait`) can never resolve to some other "src" on sys.path
  touch "$TMPD/code/src/__init__.py"
  echo "$LP_SHA" > "$TMPD/code/.fluxrt-sha"
  rm -rf "$LP_CODE.old"; [ -e "$LP_CODE" ] && mv "$LP_CODE" "$LP_CODE.old"
  mv "$TMPD/code" "$LP_CODE" && rm -rf "$LP_CODE.old"
  code_ok || die "LivePortrait-code is incomplete after the fetch"
fi

# ── 3. weights ───────────────────────────────────────────────────────────────
LINK=$REPO/LivePortrait
if [ -d "$LINK" ] && [ ! -L "$LINK" ]; then
  LP_MODELS_DIR=$LINK   # README-style manual install inside the repo: use it in place
  log "weights: using the existing $LINK directory"
fi
mkdir -p "$LP_MODELS_DIR"
"$PY" - "$LP_MODELS_DIR" "$HF_REV" $HF_REPOS <<'EOF' || die "weights download failed"
import os, shutil, sys, time
dst, rev, repos = sys.argv[1], sys.argv[2], sys.argv[3:]
FILES = {  # path in the HF repo -> size in bytes (revision 82a4fa67)
    "liveportrait/base_models/appearance_feature_extractor.pth": 3387959,
    "liveportrait/base_models/motion_extractor.pth": 112545506,
    "liveportrait/base_models/spade_generator.pth": 221813590,
    "liveportrait/base_models/warping_module.pth": 182180086,
    "liveportrait/retargeting_models/stitching_retargeting_module.pth": 2393098,
    "liveportrait/landmark.onnx": 114666491,  # only used by LivePortrait's own cropper; kept for scripts/tools
    "insightface/models/buffalo_l/det_10g.onnx": 16923827,
    "insightface/models/buffalo_l/2d106det.onnx": 5030888,
}
ok = lambda f: os.path.isfile(os.path.join(dst, f)) and os.path.getsize(os.path.join(dst, f)) == FILES[f]
todo = [f for f in FILES if not ok(f)]
if not todo:
    print(f"[liveportrait] weights: all {len(FILES)} files present in {dst}", flush=True)
    sys.exit(0)
need = sum(FILES[f] for f in todo)
free = shutil.disk_usage(dst).free
if free < need + (256 << 20):
    sys.exit(f"not enough disk space in {dst}: need {need >> 20} MB, free {free >> 20} MB")
from huggingface_hub import hf_hub_download
print(f"[liveportrait] weights: downloading {len(todo)} files ({need >> 20} MB) into {dst}", flush=True)
for f in todo:  # one file per call (multi-pattern downloads have silently dropped files before)
    for attempt in range(6):
        repo = repos[attempt % len(repos)]
        try:
            hf_hub_download(repo, f, revision=rev, local_dir=dst)
            if ok(f):
                break
            print(f"  {f}: size mismatch after download, retrying", flush=True)
            os.remove(os.path.join(dst, f))
        except Exception as e:  # noqa: BLE001
            print(f"  {f} from {repo}: attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(5 * (attempt + 1))
bad = [f for f in FILES if not ok(f)]
if bad:
    sys.exit(f"missing or incomplete after download: {bad}")
print(f"[liveportrait] weights: ok ({len(FILES)} files)", flush=True)
EOF
extra=$(find "$LP_MODELS_DIR/insightface/models/buffalo_l" -name '*.onnx' ! -name det_10g.onnx ! -name 2d106det.onnx | wc -l)
[ "$extra" = 0 ] || log "note: $extra extra .onnx files in buffalo_l/ (ignored by the processor, cost load time only)"
if [ "$LP_MODELS_DIR" != "$LINK" ]; then
  ln -sfn "$LP_MODELS_DIR" "$LINK"
  log "weights: $LINK -> $LP_MODELS_DIR"
fi

# ── 4. verify ────────────────────────────────────────────────────────────────
cd "$REPO"
"$PY" - <<'EOF' || die "import check failed"
import torch, onnxruntime as ort  # torch first, as in the FluxRT worker
import fluxrt, liveportrait
from fluxrt.stream_processor.postprocessors.liveportrait import _import_liveportrait, resolve_models_dir
_import_liveportrait()
path = list(liveportrait.__path__)
assert fluxrt.LIVEPORTRAIT_AVAILABLE and any("LivePortrait-code" in p for p in path), path
print(f"[liveportrait] imports ok: onnxruntime {ort.__version__} {ort.get_available_providers()}, "
      f"torch {torch.__version__}, liveportrait from {path[0]}, models {resolve_models_dir('LivePortrait/liveportrait')}", flush=True)
EOF

if [ "${LP_SKIP_SELFTEST:-0}" = 1 ]; then
  log "self-test skipped (LP_SKIP_SELFTEST=1)"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  log "self-test skipped (no GPU visible)"
else
  log "GPU self-test (first run on Blackwell also JIT-compiles onnxruntime's sm_90 PTX kernels)"
  cfg_arg=(); [ -f "$CONFIG" ] && cfg_arg=(--config "$CONFIG")
  set +e
  "$PY" -m fluxrt.stream_processor.postprocessors.liveportrait_selftest "${cfg_arg[@]}" \
      --out "$WS/liveportrait_selftest.jpg" 2>&1 | tee "$WS/liveportrait_selftest.log"
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || die "self-test failed (rc=$rc); see $WS/liveportrait_selftest.log"
  if grep -q "could not use CUDA for the face detector" "$WS/liveportrait_selftest.log"; then
    [ "${LP_STRICT:-0}" = 1 ] && die "face detector fell back to CPU (LP_STRICT=1)"
    log "WARNING: the face detector runs on the CPU (onnxruntime CUDA EP unavailable); lip transfer works but slower"
  fi
fi
log "done: LivePortrait ready (config: $CONFIG)"
