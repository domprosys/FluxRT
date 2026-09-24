#!/usr/bin/env bash
# Pod entry point for the RunPod template (launched from /post_start.sh, detached).
#
# Env (set in the template or at deploy time):
#   BACKEND_CONFIG  config name in configs/ (default sd_controlnet_config)
#                   e.g. sd_config, sd_controlnet_config, web_config, web_bf16_config, sdv2_config
#   ACCESS_TOKEN    optional; if set, open the page once as  <proxy-url>/?token=<ACCESS_TOKEN>
#   HF_TOKEN        optional; authenticated Hugging Face downloads
#
# Installs only what BACKEND_CONFIG needs (see runpod_setup.sh), serving a live
# progress page on :8000 meanwhile, then starts the WebRTC server on :8000.
set -uo pipefail
WS=/workspace
REPO=$WS/fluxrt
LOGS=$WS/logs
mkdir -p "$LOGS"
CFG=${BACKEND_CONFIG:-sd_controlnet_config}
CFG=${CFG%.json}
export HF_HOME=$WS/hf PATH="$HOME/.local/bin:$PATH"
log() { echo "[pod_start $(date +%H:%M:%S)] $*"; }
T0=$(date +%s)
log "backend config: $CFG"

# ── progress page on :8000 until the real server takes over ───────────────────
cat > /root/status_server.py <<'PY'
import http.server, html, os, time
LOG = "/workspace/logs/setup.log"; NET = "/workspace/logs/netcheck.txt"
T0 = float(os.environ.get("T0", time.time())); CFG = os.environ.get("CFG", "?")
def netcheck():
    try: lines = open(NET).read().split("\n")
    except OSError: return "<p>Checking this host's download speed...</p>"
    info = html.escape(lines[0]) if lines else ""
    if "SLOW" in (lines[0] if lines else ""):
        return ("<div style='background:#5a1d1d;border:1px solid #c44;padding:10px;margin:8px 0'>"
                "<b>This host's network is slow.</b> Setup may take 30+ minutes. Consider terminating this pod "
                f"and deploying another one (other host or region).<br><small>{info}</small></div>")
    return f"<p style='color:#8c8'>Network check OK &mdash; {info}</p>"
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try: tail = open(LOG, errors="replace").read().splitlines()[-40:]
        except OSError: tail = ["(waiting for setup to start)"]
        body = (f"<!doctype html><meta http-equiv=refresh content=5><title>Starting...</title>"
                f"<body style='background:#111;color:#ddd;font:14px monospace;padding:16px'>"
                f"<h3>Setting up <b>{html.escape(CFG)}</b> &mdash; {int(time.time()-T0)} s elapsed</h3>"
                f"<p>This page refreshes every 5 s and turns into the app when setup finishes.</p>{netcheck()}"
                f"<pre>{html.escape(chr(10).join(tail))}</pre>").encode()
        self.send_response(503 if self.path.startswith("/api") else 200)
        self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
PY
T0=$T0 CFG=$CFG setsid python3 /root/status_server.py > "$LOGS/status_server.log" 2>&1 < /dev/null &
STATUS_PID=$!

# ── network check: large-file download speeds (setup time is dominated by these) ──
HF_BPS=$(curl -sL -r 0-209715199 -o /dev/null -m 20 -w '%{speed_download}' \
  https://huggingface.co/Lykon/dreamshaper-8/resolve/main/unet/diffusion_pytorch_model.safetensors 2>/dev/null || echo 0)
PT_BPS=$(curl -sL -o /dev/null -m 20 -w '%{speed_download}' \
  https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl 2>/dev/null || echo 0)
NET_LINE=$(python3 -c "
hf, pt = float('${HF_BPS:-0}')/1e6, float('${PT_BPS:-0}')/1e6
slow = hf < 30 or pt < 15
print(('SLOW: ' if slow else 'OK: ') + f'Hugging Face {hf:.0f} MB/s, PyTorch index {pt:.0f} MB/s (want >=30 / >=15)')")
echo "$NET_LINE" > "$LOGS/netcheck.txt"
log "network check: $NET_LINE"

# ── decide what to install from the config's backend ─────────────────────────
CFG_FILE=$REPO/configs/$CFG.json
[ -f "$CFG_FILE" ] || { log "ERROR: $CFG_FILE not found"; exit 1; }
read -r BACKEND INT8 < <(python3 -c "import json;c=json.load(open('$CFG_FILE'));print(c.get('backend','fluxrt'), int(bool(c.get('enable_int8_quantization', True))))")
export SKIP_SD=1 SKIP_SDV2=1 SKIP_FLUXRT_WEIGHTS=1 WITH_BF16=0
case "$BACKEND" in
  fluxrt) export SKIP_FLUXRT_WEIGHTS=0; [ "$INT8" = 0 ] && export WITH_BF16=1 ;;
  sd)     export SKIP_SD=0 ;;
  sdv2)   export SKIP_SDV2=0 ;;
esac
log "backend=$BACKEND  SKIP_SD=$SKIP_SD SKIP_SDV2=$SKIP_SDV2 SKIP_FLUXRT_WEIGHTS=$SKIP_FLUXRT_WEIGHTS WITH_BF16=$WITH_BF16"

if ! bash "$REPO/deploy/runpod_setup.sh" > "$LOGS/setup.log" 2>&1; then
  log "SETUP FAILED after $(( $(date +%s) - T0 ))s — see $LOGS/setup.log (progress page stays up showing the error)"
  exit 1
fi
log "setup done in $(( $(date +%s) - T0 ))s; starting server"

# ── hand port 8000 over to the real server ───────────────────────────────────
kill "$STATUS_PID" 2>/dev/null; sleep 1
cd "$REPO"
setsid .venv/bin/python scripts/serve_web.py --config "configs/$CFG.json" --port 8000 > "$LOGS/server.log" 2>&1 < /dev/null &
for i in $(seq 1 360); do
  if curl -s -m 3 http://127.0.0.1:8000/api/state 2>/dev/null | grep -q '"ready": *true'; then
    log "READY in $(( $(date +%s) - T0 ))s total"; exit 0
  fi
  sleep 5
done
log "server did not report ready within 30 min — see $LOGS/server.log"
