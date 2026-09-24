#!/usr/bin/env bash
# IP-Adapter FaceID add-on for the StreamDiffusion venv (configs/sd_controlnet_faceid_config.json).
# Run by runpod_setup.sh when WITH_FACEID=1 (config_needs.py sets it for configs with
# worker.use_ipadapter). Idempotent. Standalone:
#   VENVS=/root/venvs WS=/workspace HF_HOME=/workspace/hf bash deploy/setup_faceid.sh
#
# 1. deploy/locks/faceid.lock verbatim (--no-deps): onnxruntime-gpu 1.23.2 (the fork's pin; PyPI builds
#    target CUDA 12.x + cuDNN 9, both already in the venv via torch 2.7.0+cu128), albumentations 1.4.24
#    + deps (insightface.app imports it at import time), cython (to build insightface), onnx 1.18.0
#    (insightface imports onnx; sd.lock's onnx 1.23.0 cannot be imported next to protobuf 4.25).
#    A CPU `onnxruntime` dist (e.g. from the fork's tensorrt extra) shares the module dir and would
#    shadow the GPU build: it is removed and onnxruntime-gpu reinstalled.
# 2. insightface 0.7.3: PyPI has only the sdist; its Cython extension is imported by insightface.app,
#    so it is compiled here against the venv's numpy 1.26 (--no-build-isolation). Needs g++ (the pod
#    image is a CUDA devel image; build-essential is apt-installed if missing). A wheel in
#    $WS/wheels (insightface-0.7.3-cp311-*.whl) is used instead when present.
# 3. diffusers-ipadapter @ 405f87d (the fork's pin): wheelhouse, else GitHub archive, else git.
# 4. insightface "buffalo_l" detection + recognition models (det_10g.onnx, w600k_r50.onnx) from the HF
#    mirror public-data/insightface (deepghs/insightface, then the GitHub release zip as fallbacks;
#    sha256-checked) into ~/.insightface/models/models/buffalo_l, the dir Diffusers_IPAdapter's
#    FaceAnalysis(name, root="~/.insightface/models") resolves. Only these two: FaceID needs the
#    5-point detection + the ArcFace embedding; the other buffalo_l models only add load time.
# 5. verify: imports, ONNX Runtime providers, a detection on insightface's sample photo.
# The IP-Adapter weights themselves (h94/IP-Adapter-FaceID, h94/IP-Adapter image encoder) come from the
# config's hf_models (runpod_setup.sh section 4a).
set -euo pipefail

VENVS=${VENVS:-/root/venvs}
WS=${WS:-/workspace}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
WHEELS=${WHEELS:-$WS/wheels}
export HF_HOME=${HF_HOME:-$WS/hf}
export NO_ALBUMENTATIONS_UPDATE=1   # albumentations (imported by insightface) phones PyPI at import
PY=$VENVS/sd/bin/python
LOCK=$REPO/deploy/locks/faceid.lock
IPA_REF=405f87da42932e30bd55ee8dca3ce502d7834a99
IPA_TGZ=https://github.com/livepeer/Diffusers_IPAdapter/archive/$IPA_REF.tar.gz
IPA_GIT="git+https://github.com/livepeer/Diffusers_IPAdapter.git@$IPA_REF"
IF_DIR=$HOME/.insightface/models/models/buffalo_l

log() { echo "[setup_faceid $(date +%H:%M:%S)] $*"; }
die() { echo "[setup_faceid] FAILED: $*" >&2; exit 1; }
retry() { local n; for n in 1 2 3 4; do "$@" && return 0; log "attempt $n/4 failed: $*"; [ $n = 4 ] || sleep $((n * 10)); done; return 1; }

[ -x "$PY" ] || die "SD venv python not found: $PY (build the venvs first: deploy/runpod_setup.sh)"
command -v uv >/dev/null || die "uv not on PATH"
[ -f "$LOCK" ] || die "missing $LOCK"

# ── 1. pinned pure/binary deps ────────────────────────────────────────────────
lock_satisfied() {
  "$PY" - "$LOCK" <<'EOF'
import sys
from importlib import metadata
bad = []
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if "==" not in line:
        continue
    name, ver = line.split("==")
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = None
    if have != ver:
        bad.append(f"{name} {have or '-'} (want {ver})")
if bad:
    print("  to install:", ", ".join(bad))
sys.exit(1 if bad else 0)
EOF
}
if "$PY" -c "from importlib.metadata import version; version('onnxruntime')" 2>/dev/null; then
  log "removing the CPU onnxruntime dist (it shadows onnxruntime-gpu)"
  uv pip uninstall --python "$PY" onnxruntime || die "cannot uninstall onnxruntime"
  uv pip uninstall --python "$PY" onnxruntime-gpu >/dev/null 2>&1 || true   # its files were shared: reinstall below
fi
if lock_satisfied; then
  log "pinned deps already installed ($LOCK)"
else
  log "installing $LOCK"
  retry uv pip install --python "$PY" --no-deps -r "$LOCK" || die "uv pip install -r $LOCK failed"
  lock_satisfied || die "packages still missing after install"
fi
if ! "$PY" -c "import onnxruntime as o; import sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" 2>/dev/null; then
  log "onnxruntime-gpu has no CUDA provider (broken/overwritten install?): reinstalling"
  retry uv pip install --python "$PY" --no-deps --reinstall-package onnxruntime-gpu \
    "onnxruntime-gpu==$(grep -E '^onnxruntime-gpu==' "$LOCK" | cut -d= -f3)" || die "cannot reinstall onnxruntime-gpu"
fi

# ── 2. insightface (compiled) ─────────────────────────────────────────────────
if "$PY" -c "import insightface.app" 2>/dev/null; then
  log "insightface present"
else
  IF_WHL=$(ls "$WHEELS"/insightface-0.7.3-cp311-*.whl 2>/dev/null | head -1 || true)
  if [ -n "$IF_WHL" ]; then
    log "insightface from wheelhouse: $IF_WHL"
    uv pip install --python "$PY" --no-deps "$IF_WHL" || die "cannot install $IF_WHL"
  else
    if ! command -v g++ >/dev/null; then
      log "g++ missing: apt-get install build-essential"
      { apt-get update -qq && apt-get install -y -qq build-essential >/dev/null; } || die "no C++ compiler for insightface"
    fi
    log "building insightface 0.7.3 from the sdist (against the venv's numpy $("$PY" -c 'import numpy; print(numpy.__version__)'))"
    retry uv pip install --python "$PY" --no-deps --no-build-isolation insightface==0.7.3 \
      || die "insightface build failed (needs g++, Python headers, cython + numpy + setuptools in the venv)"
  fi
  "$PY" -c "import insightface.app" || die "insightface installed but does not import"
fi

# ── 3. diffusers-ipadapter ────────────────────────────────────────────────────
if "$PY" -c "import diffusers_ipadapter" 2>/dev/null; then
  log "diffusers-ipadapter present"
else
  IPA_WHL=$(ls "$WHEELS"/diffusers_ipadapter-*.whl 2>/dev/null | head -1 || true)
  if [ -n "$IPA_WHL" ]; then
    log "diffusers-ipadapter from wheelhouse: $IPA_WHL"
    uv pip install --python "$PY" --no-deps "$IPA_WHL" || die "cannot install $IPA_WHL"
  else
    log "diffusers-ipadapter @ $IPA_REF (GitHub archive, git as fallback)"
    retry uv pip install --python "$PY" --no-deps "diffusers-ipadapter @ $IPA_TGZ" \
      || retry uv pip install --python "$PY" --no-deps "diffusers-ipadapter @ $IPA_GIT" \
      || die "cannot install diffusers-ipadapter from GitHub (put a wheel in $WHEELS to avoid GitHub)"
  fi
fi

# ── 4. insightface models ─────────────────────────────────────────────────────
log "insightface buffalo_l detection + recognition -> $IF_DIR"
mkdir -p "$IF_DIR"
IF_DIR="$IF_DIR" "$PY" - <<'EOF' || die "insightface models unavailable"
import hashlib, io, os, shutil, sys, time, urllib.request, zipfile
from huggingface_hub import hf_hub_download

dest = os.environ["IF_DIR"]
want = {  # sha256 (identical on both HF mirrors; = the files of insightface's v0.7 buffalo_l.zip)
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
mirrors = [("public-data/insightface", "models/buffalo_l/"), ("deepghs/insightface", "buffalo_l/")]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def ok(name):
    p = os.path.join(dest, name)
    return os.path.isfile(p) and sha(p) == want[name]

for name in want:
    if ok(name):
        print(f"  {name}: present")
        continue
    for repo, prefix in mirrors:
        for attempt in range(3):
            try:
                src = hf_hub_download(repo, prefix + name)
                shutil.copyfile(src, os.path.join(dest, name))
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {repo}/{prefix}{name} attempt {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        if ok(name):
            print(f"  {name}: from {repo}")
            break
missing = [n for n in want if not ok(n)]
if missing:  # last resort: insightface's own GitHub release
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in missing:
                    member = next(m for m in z.namelist() if m.endswith(n))
                    with open(os.path.join(dest, n), "wb") as f:
                        f.write(z.read(member))
            break
        except Exception as e:  # noqa: BLE001
            print(f"  GitHub buffalo_l.zip attempt {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    missing = [n for n in want if not ok(n)]
    if missing:
        sys.exit(f"missing or corrupt after all mirrors: {missing}")
    print("  from the GitHub release")
EOF

# ── 5. verify ─────────────────────────────────────────────────────────────────
log "verifying"
"$PY" - <<'EOF' || die "FaceID verification failed (see the traceback above)"
import torch  # load torch's CUDA/cuDNN libs first, as the worker does
import onnxruntime as ort
import insightface
import diffusers_ipadapter  # noqa: F401
from diffusers_ipadapter.ip_adapter.face_utils import get_insightface_model

prov = ort.get_available_providers()
print(f"onnxruntime {ort.__version__} providers {prov}; insightface {insightface.__version__}; torch {torch.__version__}")
app = get_insightface_model("buffalo_l")
used = app.det_model.session.get_providers()
print(f"FaceAnalysis modules {sorted(app.models)}, detector runs on {used}")
assert {"detection", "recognition"} <= set(app.models), app.models
if "CUDAExecutionProvider" not in used:
    print("WARNING: insightface runs on the CPU (~0.3 s per faceid_capture instead of ~30 ms)")
try:
    from insightface.data import get_image
    img = get_image("t1")
except Exception as e:  # noqa: BLE001  (sample images are package data; not fatal if absent)
    print(f"sample image unavailable ({e}); skipped the detection test")
else:
    faces = app.get(img)
    assert faces, "no face found in insightface's sample photo"
    print(f"sample photo: {len(faces)} faces, embedding {faces[0].normed_embedding.shape}")
print("FaceID stack OK")
EOF
log "done"
