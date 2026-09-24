#!/usr/bin/env bash
# Session A follow-up (run after bench_session_a.sh, same pod): cached attention after the
# inference_mode fix, LivePortrait with the face box taken from the webcam, and the
# all-engines server with TensorRT (reuses the engines phase 4 built).
#   bash deploy/bench_session_a2.sh /root/clip.mp4 /root/webrtc_client_test.py [OUT]
set -uo pipefail
CLIP=${1:?clip}; CLIENT=${2:?webrtc client script}; OUT=${3:-/workspace/sessA}
export WS=/workspace HF_HOME=/workspace/hf VENVS=${VENVS:-/root/venvs} PATH="$HOME/.local/bin:$PATH"
cd /workspace/fluxrt
mkdir -p "$OUT"
ts() { date +%H:%M:%S; }
say() { echo "[sessA2 $(ts)] $*"; }
PY=.venv/bin/python

T() {
  local name=$1 cfg=$2; shift 2
  say "=== $name ($cfg $*)"
  timeout 1500 $PY scripts/test_backend.py --config "$cfg" --video "$CLIP" --seconds 45 --warmup 10 \
    --samples 60,120,180 --json "$OUT/$name.json" --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  grep -E "^BENCH" "$OUT/$name.log" | cut -c1-300 || { say "  $name FAILED rc=$rc"; grep -E "Error|Traceback|error" "$OUT/$name.log" | tail -5; }
  pkill -9 -f "spawn_mai[n]|sd_worke[r]|sdv2_worke[r]"; sleep 3
}
# cfgvar base.json name key=value... -> configs/_bench_<name>.json (next to the base, so any
# config-relative path still resolves; JSON values, dotted keys for nesting)
cfgvar() {
  local base=$1 name=$2; shift 2
  local out="$(dirname "$base")/_bench_$name.json"
  python3 - "$base" "$out" "$@" <<'EOF'
import json, sys
base, out, *kv = sys.argv[1:]
c = json.load(open(base))
for item in kv:
    k, v = item.split("=", 1)
    tgt = c
    parts = k.split(".")
    for p in parts[:-1]: tgt = tgt.setdefault(p, {})
    tgt[parts[-1]] = json.loads(v)
json.dump(c, open(out, "w"), indent=1)
EOF
  echo "$out"
}

# ── cached attention (StreamV2V), moving input only: SD is deterministic on a still frame ──
for mf in 4 12; do
  c=$(cfgvar configs/sd_controlnet_config.json cache$mf worker.use_cached_attn=true worker.cache_maxframes=$mf worker.max_cache_maxframes=16 worker.cache_interval=1)
  T sdcn_cache$mf "$c"
done

# ── LivePortrait: face box from the webcam frame instead of detecting it in the stylised one ──
c=$(cfgvar configs/web_bf16_liveportrait_config.json lp_driving 'lip_transfer.source_crop="driving"')
T flux_lp_driving "$c"
c=$(cfgvar configs/web_bf16_liveportrait_config.json lp_driving_exp 'lip_transfer.source_crop="driving"' 'lip_transfer.region="exp"')
T flux_lp_driving_exp "$c"

# ── all engines resident with TensorRT (the template's ENABLE_TRT=1 path) ──
say "=== multi_trt: server with configs/multi_config.json, SD_ACCELERATION=tensorrt"
SD_ACCELERATION=tensorrt setsid $PY scripts/serve_web.py --config configs/multi_config.json --port 8000 \
  > "$OUT/multi_trt_server.log" 2>&1 < /dev/null &
t0=$(date +%s); st=""
for i in $(seq 1 240); do   # up to 20 min: the FaceID UNet engine is a new build
  st=$(curl -s -m 5 http://127.0.0.1:8000/api/state || true)
  if echo "$st" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('engines') and all(e['ready'] for e in d['engines']) else 1)" 2>/dev/null; then break; fi
  sleep 5
done
echo "$st" > "$OUT/multi_trt_state_ready.json"
say "multi_trt: engines ready after $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/multi_trt_vram.txt"
mkdir -p "$OUT/multi_trt_client"
( $PY "$CLIENT" --url http://127.0.0.1:8000 --role stage --device "$CLIP" --seconds 50 --out "$OUT/multi_trt_client" \
    > "$OUT/multi_trt_client.log" 2>&1 ) &
CPID=$!
for pair in "12:sdxl" "15:flux" "15:sd"; do
  sleep "${pair%%:*}"; eng=${pair##*:}
  curl -s -m 10 -X POST -H 'content-type: application/json' -d "{\"name\":\"$eng\"}" http://127.0.0.1:8000/api/engine \
    > "$OUT/multi_trt_switch_$eng.json"
  say "multi_trt: switched to $eng ($(head -c 120 "$OUT/multi_trt_switch_$eng.json"))"
done
wait $CPID
curl -s -m 5 http://127.0.0.1:8000/api/state > "$OUT/multi_trt_state_end.json"
pkill -INT -f "serve_web.p[y]"; sleep 8; pkill -9 -f "serve_web.p[y]|spawn_mai[n]|sd_worke[r]|sdv2_worke[r]"; sleep 3
grep -E "t=(10|20|30|40|50)s" "$OUT/multi_trt_client.log" | cut -c1-200
du -sh /workspace/engines 2>/dev/null > "$OUT/engines_size.txt"
say "=== done"
