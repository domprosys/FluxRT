#!/bin/bash
# Written to /post_start.sh by the RunPod template's start command; RunPod's /start.sh runs it
# after SSH/nginx are up. Updates the FluxRT fork (any existing checkout on a volume is reset
# to the published branch) and launches deploy/pod_start.sh detached. Logs: /workspace/logs/.
# WS (template env, default /workspace) holds the code and model weights (venvs are always under
# /root): /workspace can be a network filesystem even without a volume (EU-RO-1: ~0.4 GB/s reads),
# so WS=/root/ws on a big container disk makes installs and model loads several times faster.
mkdir -p /workspace/logs
setsid bash -c '
export PATH=$HOME/.local/bin:$PATH
WS=${WS:-/workspace}; mkdir -p "$WS"; cd "$WS"
URL=https://github.com/domprosys/FluxRT.git
for i in $(seq 1 8); do
  if [ -d fluxrt/.git ]; then
    git -C fluxrt fetch -q "$URL" installation-controls && git -C fluxrt reset -q --hard FETCH_HEAD && break
  else
    git clone -q --branch installation-controls "$URL" fluxrt && break
  fi
  echo "git attempt $i failed; retrying in 15s"; sleep 15
done
git -C fluxrt log --oneline -1
bash "$WS/fluxrt/deploy/pod_start.sh"
' > /workspace/logs/pod_start.log 2>&1 < /dev/null &
