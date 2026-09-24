#!/usr/bin/env bash
# Session A benchmark (RTX PRO 6000): SDXL-Turbo, FaceID, LivePortrait, cached attention,
# all engines resident + switching, TensorRT. Runs on a pod from /workspace/fluxrt.
#   bash deploy/bench_session_a.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
# Every test writes $OUT/<name>.json (+ .log); failures are recorded and the run continues.
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client script}; OUT=${3:-/workspace/sessA}
export WS=/workspace HF_HOME=/workspace/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd /workspace/fluxrt
mkdir -p "$OUT"
ts() { date +%H:%M:%S; }
say() { echo "[sessA $(ts)] $*"; }
PY=.venv/bin/python

# ── install: everything Session A needs (no StreamDiffusionV2 — that is Session B) ──
EXTRA=$(python3 - <<'EOF'
import json
seen = []
for c in ["sd_config", "sd_controlnet_config", "sdxl_config", "sdxl_controlnet_config", "sd_controlnet_faceid_config"]:
    for m in json.load(open(f"configs/{c}.json")).get("hf_models", []):
        if m not in seen: seen.append(m)
print(" ".join(seen))
EOF
)
say "install start"
t0=$(date +%s)
if SKIP_SD=0 SKIP_SDV2=1 SKIP_FLUXRT_WEIGHTS=0 WITH_BF16=1 WITH_TRT=1 WITH_FACEID=1 WITH_LIVEPORTRAIT=1 \
   EXTRA_HF_MODELS="$EXTRA" bash deploy/runpod_setup.sh > "$OUT/install.log" 2>&1; then
  say "install OK in $(( $(date +%s) - t0 ))s"
else
  say "INSTALL FAILED after $(( $(date +%s) - t0 ))s"; tail -n 30 "$OUT/install.log"; exit 1
fi
echo "{\"install_s\": $(( $(date +%s) - t0 ))}" > "$OUT/install.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt"

# T name config [extra test_backend args...]
T() {
  local name=$1 cfg=$2; shift 2
  say "=== $name ($cfg $*)"
  timeout 1500 $PY scripts/test_backend.py --config "$cfg" --video "$CLIP" --seconds 45 --warmup 10 \
    --samples 60,120,180 --json "$OUT/$name.json" --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  grep -E "^BENCH" "$OUT/$name.log" | cut -c1-300 || { say "  $name FAILED rc=$rc"; grep -E "Error|Traceback|error" "$OUT/$name.log" | tail -5; }
  pkill -9 -f "spawn_mai[n]|sd_worke[r]|sdv2_worke[r]"; sleep 3
}
# cfgvar base.json key=value... -> /tmp/<name>.json with worker.* overrides
cfgvar() {
  local base=$1 name=$2; shift 2
  python3 - "$base" "/tmp/$name.json" "$@" <<'EOF'
import json, sys
base, out, *kv = sys.argv[1:]
c = json.load(open(base))
for item in kv:
    k, v = item.split("=", 1)
    tgt = c
    parts = k.split(".")
    for p in parts[:-1]: tgt = tgt.setdefault(p, {})
    tgt[parts[-1]] = json.loads(v)
# relative python/worker paths are resolved against the repo root, so /tmp configs still work
json.dump(c, open(out, "w"), indent=1)
EOF
  echo "/tmp/$name.json"
}

# ── phase 1: PyTorch quality/speed ──────────────────────────────────────────────
T sdcn_base      configs/sd_controlnet_config.json
T sdcn_base_static configs/sd_controlnet_config.json --static
T sdxl           configs/sdxl_config.json
T sdxlcn         configs/sdxl_controlnet_config.json
T faceid         configs/sd_controlnet_faceid_config.json --set 60:faceid_capture=true
T flux_bf16      configs/web_bf16_config.json
T flux_lp        configs/web_bf16_liveportrait_config.json

# ── phase 2: cached attention (StreamV2V) ───────────────────────────────────────
for mf in 4 12; do
  c=$(cfgvar configs/sd_controlnet_config.json cache$mf worker.use_cached_attn=true worker.cache_maxframes=$mf worker.max_cache_maxframes=16 worker.cache_interval=1)
  T sdcn_cache$mf "$c"
  T sdcn_cache${mf}_static "$c" --static
done

# ── phase 3: all engines resident + switching over WebRTC ───────────────────────
say "=== multi: server with configs/multi_config.json"
setsid $PY scripts/serve_web.py --config configs/multi_config.json --port 8000 > "$OUT/multi_server.log" 2>&1 < /dev/null &
t0=$(date +%s)
for i in $(seq 1 240); do
  st=$(curl -s -m 5 http://127.0.0.1:8000/api/state || true)
  if echo "$st" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('engines') and all(e['ready'] for e in d['engines']) else 1)" 2>/dev/null; then break; fi
  sleep 5
done
echo "$st" > "$OUT/multi_state_ready.json"
say "multi: all engines ready after $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/multi_vram.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT/multi_vram.txt"
( $PY "$CLIENT" --url http://127.0.0.1:8000 --role stage --device "$CLIP" --seconds 70 --out "$OUT/multi_client" > "$OUT/multi_client.log" 2>&1 ) &
CPID=$!
mkdir -p "$OUT/multi_client"
# client streams the clip for 70 s; switch engines at ~12 s, ~30 s, ~48 s of streaming
for pair in "12:flux" "18:sdxl" "18:sd"; do
  sleep "${pair%%:*}"; eng=${pair##*:}
  t=$(date +%s.%N)
  curl -s -m 10 -X POST -H 'content-type: application/json' -d "{\"name\":\"$eng\"}" http://127.0.0.1:8000/api/engine > "$OUT/multi_switch_$eng.json"
  echo "{\"engine\":\"$eng\",\"t\":$t}" >> "$OUT/multi_switches.jsonl"
  say "multi: switched to $eng ($(head -c 120 "$OUT/multi_switch_$eng.json"))"
done
wait $CPID
curl -s -m 5 http://127.0.0.1:8000/api/state > "$OUT/multi_state_end.json"
pkill -INT -f "serve_web.p[y]"; sleep 8; pkill -9 -f "serve_web.p[y]|spawn_mai[n]|sd_worke[r]|sdv2_worke[r]"; sleep 3
grep -E "t=(10|20|30|40|50|60|70)s" "$OUT/multi_client.log" | cut -c1-200

# ── phase 4: TensorRT (first run includes engine builds) ────────────────────────
export SD_ACCELERATION=tensorrt
T trt_sd       configs/sd_config.json
T trt_sdcn     configs/sd_controlnet_config.json
T trt_sdxl     configs/sdxl_config.json
T trt_sdxlcn   configs/sdxl_controlnet_config.json
T trt_sdcn_warm configs/sd_controlnet_config.json
unset SD_ACCELERATION
du -sh /workspace/engines 2>/dev/null > "$OUT/engines_size.txt"
say "=== done"
