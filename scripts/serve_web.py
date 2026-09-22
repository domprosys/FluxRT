"""Real-time style-transfer web server: browser webcam in, styled video out, over WebRTC.

One backend (one GPU) serves a single "stage" feed. The first browser that
connects with role=stage owns the input; anyone else connects as a viewer and
receives the same output. Signaling is plain HTTP (POST /api/offer), media is
WebRTC via aiortc.

Backends (chosen by the config's "backend" key):
    fluxrt  FLUX.2-klein stream editing, in-process (fluxrt.StreamProcessor)
    sd      classic StreamDiffusion (SD-Turbo / LCM), worker in its own venv
    sdv2    StreamDiffusionV2 (Wan2.1 1.3B causal video), worker in its own venv

    python scripts/serve_web.py --config configs/web_config.json
    python scripts/serve_web.py --config configs/sd_config.json
    python scripts/serve_web.py --config configs/web_config.json --ssl   # LAN/phone (camera needs HTTPS)

ICE / TURN (for cloud hosts that block direct UDP):
    HF_TOKEN=...                     -> Cloudflare TURN via turn.fastrtc.org
    CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN -> Cloudflare TURN direct
    TURN_URL / TURN_USERNAME / TURN_CREDENTIAL             -> any TURN server
    otherwise a public STUN server only (fine on a LAN).
"""

import argparse
import asyncio
import fractions
import json
import logging
import os
import subprocess
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from fluxrt.backends.base import Backend
from fluxrt.utils import crop_maximal_rectangle

log = logging.getLogger("fluxrt.web")
REPO_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
BACKENDS_DIR = REPO_DIR / "src" / "fluxrt" / "backends"
VIDEO_CLOCK = 90000

WORKER_SCRIPTS = {
    "sd": BACKENDS_DIR / "sd_worker.py",
    "sdv2": BACKENDS_DIR / "sdv2_worker.py",
}


def build_backend(config_path: str, cfg: dict, force_int8: bool = False) -> Backend:
    kind = cfg.get("backend", "fluxrt")
    res = cfg["resolution"]
    if kind == "fluxrt":
        from fluxrt.backends.base import FluxRTBackend

        return FluxRTBackend(config_path, cfg, force_int8=force_int8)
    if kind in WORKER_SCRIPTS:
        from fluxrt.backends.worker_client import WorkerBackend

        out = cfg.get("out_resolution") or res
        python = cfg.get("python")
        if not python:
            raise ValueError(f"config for backend '{kind}' needs a \"python\" path to the worker venv")
        env = dict(os.environ, **{k: str(v) for k, v in (cfg.get("worker_env") or {}).items()})
        # "python" / "worker_cwd" may be relative to the repo root (e.g.
        # "../StreamDiffusion-daydream/.venv/bin/python"), so the same config
        # works wherever the sibling library clone lives (dev box, pod, ...).
        # normpath, not resolve(): the venv's python is a symlink to the base
        # interpreter and following it would bypass the venv's site-packages.
        python = Path(os.path.normpath(Path(python).expanduser()))
        if not python.is_absolute():
            python = Path(os.path.normpath(REPO_DIR / python))
        cwd = cfg.get("worker_cwd")
        if cwd and not Path(cwd).expanduser().is_absolute():
            cwd = os.path.normpath(REPO_DIR / cwd)
        if not python.exists():
            raise FileNotFoundError(f"worker python not found: {python}")
        return WorkerBackend(
            name=kind,
            python=str(python),
            script=str(WORKER_SCRIPTS[kind]),
            cfg=cfg,
            resolution=(res["height"], res["width"]),
            out_resolution=(out["height"], out["width"]),
            env=env,
            cwd=cwd,
        )
    raise ValueError(f"unknown backend '{kind}' (known: fluxrt, {', '.join(WORKER_SCRIPTS)})")


# ── ICE servers ─────────────────────────────────────────────────────────────
_ice_cache: dict = {"until": 0.0, "servers": None}


def fetch_ice_servers(ttl: int = 600) -> list[dict]:
    """Return ICE servers as plain dicts ({urls, username?, credential?})."""
    now = time.time()
    if _ice_cache["servers"] is not None and now < _ice_cache["until"]:
        return _ice_cache["servers"]

    servers: list[dict] | None = None
    hf_token = os.environ.get("HF_TOKEN")
    cf_id = os.environ.get("CLOUDFLARE_TURN_KEY_ID")
    cf_tok = os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
    turn_url = os.environ.get("TURN_URL")
    try:
        if hf_token:
            req = urllib.request.Request(
                f"https://turn.fastrtc.org/credentials?ttl={ttl}",
                headers={"Authorization": f"Bearer {hf_token}"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                servers = json.load(r)["iceServers"]
        elif cf_id and cf_tok:
            req = urllib.request.Request(
                f"https://rtc.live.cloudflare.com/v1/turn/keys/{cf_id}/credentials/generate-ice-servers",
                data=json.dumps({"ttl": ttl}).encode(),
                headers={"Authorization": f"Bearer {cf_tok}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                servers = json.load(r)["iceServers"]
        elif turn_url:
            servers = [
                {"urls": "stun:stun.l.google.com:19302"},
                {
                    "urls": turn_url,
                    "username": os.environ.get("TURN_USERNAME", ""),
                    "credential": os.environ.get("TURN_CREDENTIAL", ""),
                },
            ]
    except Exception as e:  # noqa: BLE001
        log.warning("TURN credential fetch failed (%s); falling back to STUN", e)

    if not servers:
        servers = [{"urls": "stun:stun.l.google.com:19302"}]
    _ice_cache.update(servers=servers, until=now + ttl * 0.8)
    return servers


def to_rtc_ice(servers: list[dict]) -> list[RTCIceServer]:
    return [
        RTCIceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential"))
        for s in servers
    ]


# ── Output track: paced reader of the backend's output slot ─────────────────
# One instance per peer connection. Do NOT fan one track out with MediaRelay:
# aiortc's VP8 encoder segfaults when two senders reformat the same VideoFrame
# object concurrently in the thread pool (reproduced with aiortc 1.15 / av 17).
# Reading the output slot per peer is cheap (one ~700 KB copy per frame).
class OutputTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, server: "Server", fps: float):
        super().__init__()
        self.server = server
        self.fps = fps
        self._start: float | None = None
        self._n = 0

    async def recv(self) -> VideoFrame:
        now = time.monotonic()
        if self._start is None or now - (self._start + self._n / self.fps) > 0.5:
            self._start = now  # first frame, or we fell far behind
            self._n = 0
        self._n += 1
        delay = self._start + self._n / self.fps - now
        if delay > 0:
            await asyncio.sleep(delay)
        arr = self.server.current_output_frame()
        frame = VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = int(self._n * VIDEO_CLOCK / self.fps)
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
        return frame


# ── Server state ────────────────────────────────────────────────────────────
class Server:
    def __init__(self, args):
        self.args = args
        with open(args.config) as f:
            self.cfg = json.load(f)
        self.backend = build_backend(args.config, self.cfg, force_int8=args.int8)
        self.res = {"height": self.backend.resolution[0], "width": self.backend.resolution[1]}
        self.out_res = {"height": self.backend.out_resolution[0], "width": self.backend.out_resolution[1]}

        self.cycle: list[str] = self.cfg.get("prompt_cycle") or []
        self.prompt_index = 0
        self.custom_prompt: str | None = None
        self.cycle_interval = float(self.cfg.get("prompt_cycle_interval_s", 0) or 0)
        self.cycle_enabled = len(self.cycle) > 1 and self.cycle_interval > 0
        self._cycle_deadline = time.monotonic() + self.cycle_interval
        self._initial_prompt_sent = False

        self.alpha = float(self.cfg.get("input_smoothing_alpha", 1.0))
        self._ema: np.ndarray | None = None

        self.pcs: dict[str, RTCPeerConnection] = {}
        self.roles: dict[str, str] = {}
        self.stage: str | None = None

        self.t0 = time.time()
        self._in_times: list[float] = []
        self._last_output: np.ndarray | None = None
        self._placeholder = np.zeros((self.out_res["height"], self.out_res["width"], 3), np.uint8)

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        log.info(
            "starting backend '%s': %dx%d -> %dx%d, cycle=%d prompts",
            self.backend.name, self.res["width"], self.res["height"],
            self.out_res["width"], self.out_res["height"], len(self.cycle),
        )
        self.backend.start()

    async def close(self):
        for pc in list(self.pcs.values()):
            await pc.close()
        self.pcs.clear()
        self.backend.stop()

    # -- frames --------------------------------------------------------------
    def push_input(self, bgr: np.ndarray):
        frame = crop_maximal_rectangle(bgr, self.res["height"], self.res["width"])
        if self.alpha < 1.0:
            f32 = frame.astype(np.float32)
            if self._ema is None or self._ema.shape != f32.shape:
                self._ema = f32
            else:
                self._ema = self.alpha * f32 + (1.0 - self.alpha) * self._ema
            frame = self._ema.astype(np.uint8)
        self.backend.push_input(frame)
        now = time.monotonic()
        self._in_times.append(now)
        if len(self._in_times) > 60:
            del self._in_times[:-60]

    def current_output_frame(self) -> np.ndarray:
        if self.backend.is_ready():
            out = self.backend.current_output_frame()
            if out is not None:
                self._last_output = out
                return out
            if self._last_output is not None:
                return self._last_output
        img = self._placeholder.copy()
        if not self.backend.alive():
            msg = "Backend died - check the server log"
        elif self.backend.is_ready():
            msg = "Ready - waiting for a camera"
        else:
            msg = f"Loading {self.backend.name} model... {int(time.time() - self.t0)}s"
        cv2.putText(img, msg, (20, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2, cv2.LINE_AA)
        return img

    def input_fps(self) -> float:
        t = self._in_times
        if len(t) < 2 or time.monotonic() - t[-1] > 1.0:
            return 0.0
        return (len(t) - 1) / max(t[-1] - t[0], 1e-6)

    # -- prompts -------------------------------------------------------------
    def apply_index(self, idx: int):
        if not self.cycle:
            return
        self.prompt_index = idx % len(self.cycle)
        self.custom_prompt = None
        self.backend.set_prompt_index(self.prompt_index, self.cycle[self.prompt_index])
        self._cycle_deadline = time.monotonic() + self.cycle_interval

    def apply_custom(self, text: str):
        self.custom_prompt = text
        self.cycle_enabled = False
        self.backend.set_prompt(text)

    async def cycle_loop(self):
        while True:
            await asyncio.sleep(0.25)
            if not self.backend.is_ready():
                continue
            if not self._initial_prompt_sent:
                # Worker backends start with no prompt; FluxRT pre-encodes its own.
                self._initial_prompt_sent = True
                if self.backend.name != "fluxrt":
                    text = self.cycle[0] if self.cycle else self.cfg.get("default_prompt", "")
                    if text:
                        self.backend.set_prompt_index(0, text)
            if self.cycle_enabled and time.monotonic() >= self._cycle_deadline:
                self.apply_index(self.prompt_index + 1)

    # -- webrtc --------------------------------------------------------------
    async def consume_input(self, track: MediaStreamTrack, pc_id: str):
        log.info("stage %s: receiving camera", pc_id[:8])
        while self.stage == pc_id:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            self.push_input(frame.to_ndarray(format="bgr24"))
        log.info("stage %s: camera ended", pc_id[:8])

    async def drop_pc(self, pc_id: str):
        pc = self.pcs.pop(pc_id, None)
        self.roles.pop(pc_id, None)
        if self.stage == pc_id:
            self.stage = None
            self._ema = None
        if pc is not None:
            await pc.close()

    async def handle_offer(self, sdp: str, typ: str, role: str, force: bool):
        if role not in ("stage", "view"):
            raise HTTPException(400, "role must be 'stage' or 'view'")
        pc_id = uuid.uuid4().hex
        if role == "stage":
            if self.stage is not None and not force:
                raise HTTPException(409, "stage is busy; connect as viewer or force")
            if self.stage is not None:
                await self.drop_pc(self.stage)
            self.stage = pc_id

        pc = RTCPeerConnection(RTCConfiguration(iceServers=to_rtc_ice(fetch_ice_servers())))
        self.pcs[pc_id] = pc
        self.roles[pc_id] = role

        # The page adds its transceivers in a fixed order: output (recvonly)
        # first, then the camera (sendonly). aiortc pairs our unassociated
        # transceiver with the first video m-line, so add our track now.
        sender = pc.addTrack(OutputTrack(self, self.args.out_fps))
        if self.args.codec:
            from aiortc import RTCRtpSender

            want = self.args.codec.upper()
            prefs = [c for c in RTCRtpSender.getCapabilities("video").codecs if want in c.mimeType.upper()]
            if prefs:
                for t in pc.getTransceivers():
                    if t.sender is sender:
                        t.setCodecPreferences(prefs)

        @pc.on("track")
        def on_track(track):
            if track.kind == "video" and role == "stage":
                asyncio.ensure_future(self.consume_input(track, pc_id))

        @pc.on("connectionstatechange")
        async def on_state():
            log.info("pc %s (%s): %s", pc_id[:8], role, pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.drop_pc(pc_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=typ))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"id": pc_id, "role": role, "sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    def state(self) -> dict:
        st = self.backend.stats()
        proc = float(st.get("proc_time_s") or 0.0)
        return {
            "backend": self.backend.name,
            "ready": self.backend.is_ready(),
            "alive": self.backend.alive(),
            "uptime_s": round(time.time() - self.t0, 1),
            "resolution": self.res,
            "out_resolution": self.out_res,
            "stage_busy": self.stage is not None,
            "viewers": sum(1 for r in self.roles.values() if r == "view"),
            "prompt_index": self.prompt_index,
            "custom_prompt": self.custom_prompt,
            "cycle": self.cycle,
            "cycle_enabled": self.cycle_enabled,
            "cycle_interval_s": self.cycle_interval,
            "cycle_remaining_s": max(0.0, round(self._cycle_deadline - time.monotonic(), 1))
            if self.cycle_enabled else None,
            "base_fps": round(1.0 / proc, 1) if proc > 0 else 0.0,
            "input_fps": round(self.input_fps(), 1),
            "out_fps": self.args.out_fps,
            "smoothing_alpha": self.alpha,
            **st,
        }


# ── FastAPI app ─────────────────────────────────────────────────────────────
def build_app(server: Server) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server.start()
        task = asyncio.ensure_future(server.cycle_loop())
        try:
            yield
        finally:
            task.cancel()
            await server.close()

    app = FastAPI(title="Real-time style transfer", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/ice-servers")
    async def ice():
        return {"iceServers": fetch_ice_servers()}

    @app.get("/api/state")
    async def state():
        return server.state()

    @app.post("/api/offer")
    async def offer(body: dict):
        return await server.handle_offer(
            body["sdp"], body["type"], body.get("role", "stage"), bool(body.get("force"))
        )

    @app.post("/api/hangup")
    async def hangup(body: dict):
        await server.drop_pc(body.get("id", ""))
        return {"ok": True}

    @app.post("/api/prompt_index")
    async def prompt_index(body: dict):
        server.apply_index(int(body["index"]))
        return server.state()

    @app.post("/api/prompt")
    async def prompt(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "empty prompt")
        server.apply_custom(text)
        return server.state()

    @app.post("/api/cycle")
    async def cycle(body: dict):
        if "enabled" in body:
            server.cycle_enabled = bool(body["enabled"]) and len(server.cycle) > 1
            server._cycle_deadline = time.monotonic() + server.cycle_interval
        if "interval_s" in body:
            server.cycle_interval = max(1.0, float(body["interval_s"]))
            server._cycle_deadline = time.monotonic() + server.cycle_interval
        return server.state()

    @app.post("/api/smoothing")
    async def smoothing(body: dict):
        server.alpha = min(1.0, max(0.05, float(body["alpha"])))
        server._ema = None
        return server.state()

    @app.post("/api/gen")
    async def gen(body: dict):
        # Backend-specific live parameter. fluxrt: steps, seed, shift,
        # use_dynamic_shifting, stochastic_sampling, time_shift_type, sigmas,
        # rife_scale, interpolation_exp. Workers: whatever their on_command accepts.
        server.backend.set_param(str(body["name"]), body["value"])
        return {"ok": True}

    return app


def ensure_dev_cert(cert_dir: Path) -> tuple[Path, Path]:
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if not cert.exists():
        cert_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
             "-subj", "/CN=fluxrt-dev", "-keyout", str(key), "-out", str(cert)],
            check=True, capture_output=True,
        )
        log.info("generated self-signed dev cert in %s", cert_dir)
    return cert, key


def main():
    ap = argparse.ArgumentParser(description="Serve real-time style transfer over WebRTC.")
    ap.add_argument("--config", default="configs/web_config.json")
    ap.add_argument("--int8", action="store_true", help="fluxrt: force int8 (also honoured from config)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-fps", type=float, default=30.0, help="WebRTC output frame rate")
    ap.add_argument("--codec", default=None, help="prefer 'h264' or 'vp8' for the output track")
    ap.add_argument("--ssl", action="store_true", help="serve HTTPS with a self-signed dev cert")
    ap.add_argument("--debug-rtc", action="store_true", help="aiortc debug logging (minus per-packet lines)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.debug_rtc:
        class _NoPackets(logging.Filter):
            def filter(self, rec):
                m = rec.getMessage()
                return not (" > " in m or " < " in m)

        for h in logging.getLogger().handlers:
            h.addFilter(_NoPackets())
        logging.getLogger("aiortc").setLevel(logging.DEBUG)

    server = Server(args)
    app = build_app(server)

    ssl_kw = {}
    if args.ssl:
        cert, key = ensure_dev_cert(Path(".cache") / "dev-cert")
        ssl_kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **ssl_kw)


if __name__ == "__main__":
    main()
