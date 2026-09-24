# Shared helpers for the pod benchmark drivers (source it from the repo root on the pod).
# Needs: CLIP (input video), CLIENT (webrtc client script), OUT (results dir), TAG (log prefix).
# Every test writes $OUT/<name>.json (+ .log); failures are recorded and the run continues.
PY=${PY:-.venv/bin/python}
T_SECONDS=${T_SECONDS:-45}; T_WARMUP=${T_WARMUP:-10}

ts() { date +%H:%M:%S; }
say() { echo "[${TAG:-bench} $(ts)] $*"; }
kill_workers() { pkill -9 -f "spawn_mai[n]|sd_worke[r]|sdv2_worke[r]"; sleep 3; }

# T name config [extra test_backend args...]  (T_SECONDS / T_WARMUP override the durations)
T() {
  local name=$1 cfg=$2; shift 2
  say "=== $name ($cfg $*)"
  timeout 1500 $PY scripts/test_backend.py --config "$cfg" --video "$CLIP" --seconds "$T_SECONDS" --warmup "$T_WARMUP" \
    --samples 60,120,180 --json "$OUT/$name.json" --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  grep -E "^BENCH" "$OUT/$name.log" | cut -c1-300 || { say "  $name FAILED rc=$rc"; grep -E "Error|Traceback|error" "$OUT/$name.log" | tail -5; }
  kill_workers
}

# cfgvar base.json name key=value... -> configs/_bench_<name>.json, printed. Written next to the
# base so config-relative paths still resolve. Values are JSON; dotted keys nest.
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

# multi_run name config stream_seconds "delay:engine delay:engine ..."
# Starts serve_web with a multi config, waits until every engine is ready (max MULTI_WAIT_S, default
# 20 min), records
# VRAM, streams the clip over WebRTC as the stage and switches engines at the given delays.
multi_run() {
  local name=$1 cfg=$2 secs=$3 plan=$4 st="" t0 i pair eng cpid
  say "=== $name: server with $cfg"
  setsid $PY scripts/serve_web.py --config "$cfg" --port 8000 > "$OUT/${name}_server.log" 2>&1 < /dev/null &
  t0=$(date +%s)
  for i in $(seq 1 $(( ${MULTI_WAIT_S:-1200} / 5 ))); do
    st=$(curl -s -m 5 http://127.0.0.1:8000/api/state || true)
    if echo "$st" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('engines') and all(e['ready'] for e in d['engines']) else 1)" 2>/dev/null; then break; fi
    sleep 5
  done
  echo "$st" > "$OUT/${name}_state_ready.json"
  say "$name: engines ready after $(( $(date +%s) - t0 ))s"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/${name}_vram.txt"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT/${name}_vram.txt"
  mkdir -p "$OUT/${name}_client"
  ( $PY "$CLIENT" --url http://127.0.0.1:8000 --role stage --device "$CLIP" --seconds "$secs" --out "$OUT/${name}_client" \
      > "$OUT/${name}_client.log" 2>&1 ) &
  cpid=$!
  for pair in $plan; do
    sleep "${pair%%:*}"; eng=${pair##*:}
    curl -s -m 10 -X POST -H 'content-type: application/json' -d "{\"name\":\"$eng\"}" http://127.0.0.1:8000/api/engine \
      > "$OUT/${name}_switch_$eng.json"
    echo "{\"engine\":\"$eng\",\"t\":$(date +%s.%N)}" >> "$OUT/${name}_switches.jsonl"
    say "$name: switched to $eng ($(head -c 120 "$OUT/${name}_switch_$eng.json"))"
  done
  wait $cpid
  curl -s -m 5 http://127.0.0.1:8000/api/state > "$OUT/${name}_state_end.json"
  pkill -INT -f "serve_web.p[y]"; sleep 8; pkill -9 -f "serve_web.p[y]"; kill_workers
  grep -E "t=[ 0-9]*0s" "$OUT/${name}_client.log" | cut -c1-200
}
