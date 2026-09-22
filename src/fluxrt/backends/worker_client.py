"""Parent-side backend that runs a worker script in another Python environment.

    backend = WorkerBackend(
        python="/path/to/other/.venv/bin/python",
        script="/path/to/sd_worker.py",
        cfg={...},                # written to a temp JSON and passed as --config
        resolution=(512, 512),
    )
Frames go through shared memory, commands through the worker's stdin, events
(ready / log / error) come back on its stdout. See shm_protocol.py.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from fluxrt.backends.base import Backend
from fluxrt.backends.shm_protocol import FrameSlot, Meta

log = logging.getLogger("fluxrt.backend")


class WorkerBackend(Backend):
    def __init__(
        self,
        name: str,
        python: str,
        script: str,
        cfg: dict,
        resolution: tuple[int, int],
        out_resolution: tuple[int, int] | None = None,
        env: dict | None = None,
        cwd: str | None = None,
    ):
        self.name = name
        self.python, self.script, self.cfg = python, script, cfg
        self.resolution = tuple(resolution)
        self.out_resolution = tuple(out_resolution or resolution)
        self.env, self.cwd = env, cwd
        self.proc: subprocess.Popen | None = None
        self._ready = False
        self._last_error: str | None = None
        self.info: dict = {}
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        h, w = self.resolution
        oh, ow = self.out_resolution
        self.in_slot = FrameSlot((h, w, 3), create=True)
        self.out_slot = FrameSlot((oh, ow, 3), create=True)
        self.meta = Meta(create=True)
        self._cfg_file = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="fluxrt-worker-", delete=False)
        json.dump(self.cfg, self._cfg_file)
        self._cfg_file.close()
        cmd = [
            self.python, self.script,
            "--in-shm", self.in_slot.name, "--out-shm", self.out_slot.name, "--meta-shm", self.meta.name,
            "--height", str(h), "--width", str(w), "--out-height", str(oh), "--out-width", str(ow),
            "--config", self._cfg_file.name,
        ]
        log.info("[%s] spawning worker: %s", self.name, " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=self.env, cwd=self.cwd,
        )
        threading.Thread(target=self._event_reader, daemon=True).start()

    def _event_reader(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                log.info("[%s] %s", self.name, line)
                continue
            kind = ev.get("event")
            if kind == "ready":
                self._ready = True
                log.info("[%s] worker ready", self.name)
            elif kind == "log":
                log.info("[%s] %s", self.name, ev.get("msg"))
            elif kind == "error":
                self._last_error = ev.get("msg")
                log.error("[%s] %s", self.name, ev.get("msg"))
            elif kind == "info":
                self.info.update({k: v for k, v in ev.items() if k != "event"})
        log.warning("[%s] worker stdout closed (exit=%s)", self.name, self.proc.poll())

    def _send(self, obj: dict) -> None:
        if not self.proc or not self.proc.stdin or self.proc.poll() is not None:
            return
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self._send({"cmd": "stop"})
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for s in (getattr(self, "in_slot", None), getattr(self, "out_slot", None), getattr(self, "meta", None)):
            if s is not None:
                s.close()
        try:
            Path(self._cfg_file.name).unlink()
        except Exception:  # noqa: BLE001
            pass

    # -- state ---------------------------------------------------------------
    def is_ready(self) -> bool:
        return self._ready and self.alive()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def push_input(self, bgr: np.ndarray) -> None:
        self.in_slot.write(np.ascontiguousarray(bgr))
        self.meta.inc("input_seq")

    def current_output_frame(self) -> np.ndarray | None:
        if self.meta.get("output_seq") <= 0:
            return None
        return self.out_slot.read()

    def set_prompt(self, text: str) -> None:
        self._send({"cmd": "set_prompt", "text": text})

    def set_param(self, name: str, value) -> None:
        self._send({"cmd": "set_param", "name": name, "value": value})

    def stats(self) -> dict:
        hb = self.meta.get("heartbeat")
        return {
            "proc_time_s": round(self.meta.get("proc_time_ms") / 1000.0, 4),
            "gpu_reserved_mb": int(self.meta.get("gpu_mb")),
            "worker_alive": self.alive(),
            "worker_stalled_s": round(time.time() - hb, 1) if hb else None,
            "last_error": self._last_error,
            **self.info,
        }
