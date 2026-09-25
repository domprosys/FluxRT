# FluxRT real-time stylization
Browser webcam in, styled video out, over WebRTC. One visitor at a time.

**Deploy:** pick any region + GPU (RTX PRO 6000 recommended: all engines fit at once; 4090 for single engines).
No network volume needed: everything installs onto the container disk (`WS=/root/ws`) in ~5-10 min.
With one of our network volumes attached, set `WS=/workspace` to use what is on it.

**Env:**
- `BACKEND_CONFIG` = multi_all_config (default: FluxRT + SD + SDXL + StreamDiffusionV2 resident, switch on the page;
  needs ~56 GB VRAM) | multi_config (without StreamDiffusionV2, ~37 GB) | on a 24 GB card one engine:
  sd_controlnet_config | sdxl_controlnet_config | web_bf16_config | web_bf16_liveportrait_config | sdv2_config |
  sd_config | web_config.
- `ACCESS_TOKEN` = optional password; then open the page as `<Connect HTTP 8000 link>/?token=<ACCESS_TOKEN>`.
- `ENABLE_TRT=1` = TensorRT for the SD engines (1.7-1.9x faster without ControlNet), but engines build on first
  start: +10-30 min on every fresh pod.
- `WS` = install root (default `/root/ws` in this template).
- `IDLE_STOP_MIN` = stop the pod after this many minutes with no visitor (default 30; empty = never).
  `IDLE_ACTION=terminate` terminates instead. A stopped pod keeps its URL; starting it again re-runs the setup.

**Use:** Connect -> HTTP 8000. While it installs, that page shows live setup progress; it becomes the app when ready.
Logs: /workspace/logs/ (pod_start.log, setup.log, server.log).
