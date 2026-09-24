#!/usr/bin/env bash
# TensorRT add-on for the StreamDiffusion (SD / SD-Turbo / SDXL) venv.
# Run by runpod_setup.sh when WITH_TRT=1 (pod_start.sh: ENABLE_TRT=1, which also exports
# SD_ACCELERATION=tensorrt so every SD/SDXL worker uses TensorRT). Idempotent. Standalone:
#   VENVS=/root/venvs WS=/workspace bash deploy/setup_trt.sh
#
# Installs deploy/locks/trt.lock verbatim (--no-deps, NVIDIA index first):
#   TensorRT 10.12.0.36  the fork's own pin (tools/install-tensorrt.py); Blackwell sm_120 needs >= 10.8.
#                        cu12 build: runs on the venv's CUDA 12.8 runtime; its cuDNN pin (9.7.1.26)
#                        is the one torch 2.7.0+cu128 already brought.
#   cuda-python 12.8.0   matches torch's CUDA 12.8 and still has the `from cuda import cudart` API the
#                        fork uses (removed in cuda-python 13).
#   polygraphy 0.49.24, onnx-graphsurgeon 0.5.8, colored 2.2.4   the fork's pins
#   onnx 1.18.0          the fork's pin. sd.lock's onnx 1.23.0 cannot even be imported next to the
#                        protobuf 4.25 that mediapipe needs, so TensorRT export would die at import.
# plus diffusers-ipadapter (livepeer/Diffusers_IPAdapter @ the fork's commit): the fork's TensorRT
# export package imports it unconditionally, IP-Adapter or not.
# Then verifies the imports and builds a tiny engine on this GPU (proves the arch is supported).
#
# Engines are built by the worker on first use, into $WS/engines/sm<cc>-trt<version>/ (on the network
# volume when one is mounted, so later pods only load them; without a volume every cold start rebuilds).
# First-build times on a 4090 / RTX PRO 6000 class GPU (ONNX export + TensorRT build):
#   SD-Turbo / SD1.5 UNet 3-6 min, SDXL UNet 10-20 min, SD1.5 ControlNet 2-4 min each,
#   SDXL ControlNet 5-8 min each, tiny VAE encoder+decoder ~1 min.
# Sizes: SD1.5 UNet ~1.7 GB, SDXL UNet ~5 GB, ControlNet 0.7 GB (SD1.5) / 2.5 GB (SDXL); the export
# needs about twice the engine size of temporary disk in the engine dir (SDXL: ~15 GB).
set -euo pipefail

VENVS=${VENVS:-/root/venvs}
WS=${WS:-/workspace}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
WHEELS=${WHEELS:-$WS/wheels}
PY=$VENVS/sd/bin/python
LOCK=$REPO/deploy/locks/trt.lock
NV_INDEX=https://pypi.nvidia.com
IPA_REF=405f87da42932e30bd55ee8dca3ce502d7834a99   # = the fork's setup.py pin
IPA_TGZ=https://github.com/livepeer/Diffusers_IPAdapter/archive/$IPA_REF.tar.gz
IPA_GIT="git+https://github.com/livepeer/Diffusers_IPAdapter.git@$IPA_REF"

log() { echo "[setup_trt $(date +%H:%M:%S)] $*"; }
die() { echo "[setup_trt] FAILED: $*" >&2; exit 1; }
retry() { local n; for n in 1 2 3 4; do "$@" && return 0; log "attempt $n/4 failed: $*"; [ $n = 4 ] || sleep $((n * 10)); done; return 1; }

[ -x "$PY" ] || die "SD venv python not found: $PY (build the venvs first: deploy/runpod_setup.sh)"
command -v uv >/dev/null || die "uv not on PATH"
[ -f "$LOCK" ] || die "missing $LOCK"

# 0 = every pin in the lock installed at its exact version (distribution metadata only, no imports)
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

if lock_satisfied; then
  log "TensorRT stack already installed ($LOCK)"
else
  log "installing the TensorRT stack from $LOCK"
  retry uv pip install --python "$PY" --no-deps --extra-index-url "$NV_INDEX" -r "$LOCK" \
    || die "uv pip install -r $LOCK failed (NVIDIA index $NV_INDEX reachable?)"
  lock_satisfied || die "packages still missing after install"
fi

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

log "verifying (imports + a tiny engine build on this GPU)"
"$PY" - <<'EOF' || die "TensorRT verification failed (see the traceback above)"
import importlib, sys
import torch
import tensorrt as trt
from cuda import cudart  # noqa: F401  (legacy API the fork uses)
import onnx, onnx_graphsurgeon, polygraphy  # noqa: F401,E401
import diffusers_ipadapter  # noqa: F401
# the fork's TensorRT modules import cleanly
for m in ("streamdiffusion.acceleration.tensorrt.utilities",
          "streamdiffusion.acceleration.tensorrt.engine_manager",
          "streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export"):
    importlib.import_module(m)
print(f"tensorrt {trt.__version__}  onnx {onnx.__version__}  torch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("CUDA not available to torch")
cc = torch.cuda.get_device_capability()
print(f"GPU {torch.cuda.get_device_name()}  sm_{cc[0]}{cc[1]}")
# tiny fp16 engine: fails fast when this TensorRT does not support the GPU architecture
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(0)
x = net.add_input("x", trt.float16, (1, 8, 16, 16))
y = net.add_elementwise(x, x, trt.ElementWiseOperation.SUM)
net.mark_output(y.get_output(0))
cfg = builder.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
plan = builder.build_serialized_network(net, cfg)
if plan is None:
    sys.exit("tiny engine build failed")
engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
assert engine is not None, "tiny engine does not deserialize"
print(f"tiny engine OK ({plan.nbytes} bytes): TensorRT {trt.__version__} supports sm_{cc[0]}{cc[1]}")
print(f"engine dir used by the workers: $WS/engines/sm{cc[0]}{cc[1]}-trt{trt.__version__}")
EOF
mkdir -p "$WS/engines" 2>/dev/null || true
log "done. Engines build on first use (SD1.5 UNet ~3-6 min, SDXL UNet ~10-20 min, ControlNets 2-8 min each)."
