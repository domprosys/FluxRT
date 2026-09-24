"""MultiBackend: several engines resident at once, one active, crossfaded switches.

Config (configs/multi_config.json):
    {
      "backend": "multi",
      "resolution":     {"height": 576, "width": 1024},   # input canvas the server crops the camera to
      "out_resolution": {"height": 368, "width": 640},    # every engine's output is fitted to this
      "default_engine": "sd",
      "crossfade_s": 1.5,
      "engines": {
        "flux": {"config": "web_bf16_config"},
        "sd":   {"config": "sd_controlnet_config", "resolution": {"height": 384, "width": 640}},
        ...
      }
    }
Per-engine "resolution" overrides the engine config's own, so square engines can run
16:9 and fit the shared canvas without distortion (inputs and outputs are fitted with an
aspect-preserving centre crop).

Only the active engine is fed frames (plus the outgoing one during a crossfade), so idle
worker engines use no GPU time. FluxRT's loop runs regardless of input, so idle FluxRT
engines are paused explicitly (set_param "paused").
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from fluxrt.backends.base import Backend
from fluxrt.utils import crop_maximal_rectangle

log = logging.getLogger("fluxrt.multi")


def _fit(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if frame.shape[:2] == (h, w):
        return frame
    return crop_maximal_rectangle(frame, h, w)


class _Engine:
    def __init__(self, name: str, cfg: dict, config_path: str, backend: Backend):
        self.name, self.cfg, self.config_path, self.backend = name, cfg, config_path, backend
        self.cycle: list[str] = cfg.get("prompt_cycle") or ([cfg["default_prompt"]] if cfg.get("default_prompt") else [])
        self.cycle_interval = float(cfg.get("prompt_cycle_interval_s", 0) or 0)
        self.prompt_index = 0
        self.custom_prompt: str | None = None
        self.prompted = False          # initial prompt sent (worker engines start without one)
        self.last_out: np.ndarray | None = None

    @property
    def is_fluxrt(self) -> bool:
        return self.backend.name == "fluxrt"


class MultiBackend(Backend):
    name = "multi"

    def __init__(self, cfg: dict, config_path: str, build_fn):
        self.cfg = cfg
        cfg_dir = Path(config_path).resolve().parent
        res, out = cfg["resolution"], cfg.get("out_resolution") or cfg["resolution"]
        self.resolution = (res["height"], res["width"])
        self.out_resolution = (out["height"], out["width"])
        self.crossfade_s = float(cfg.get("crossfade_s", 1.5))
        self.engines: dict[str, _Engine] = {}
        for name, spec in cfg["engines"].items():
            spec = {"config": spec} if isinstance(spec, str) else dict(spec)
            path = cfg_dir / (spec["config"] if spec["config"].endswith(".json") else spec["config"] + ".json")
            sub = json.loads(path.read_text())
            if "resolution" in spec:
                sub["resolution"] = spec["resolution"]
            for k, v in (spec.get("overrides") or {}).items():
                sub[k] = v
            self.engines[name] = _Engine(name, sub, str(path), build_fn(str(path), sub))
        self.active = cfg.get("default_engine") or next(iter(self.engines))
        if self.active not in self.engines:
            raise ValueError(f"default_engine {self.active!r} not in engines {list(self.engines)}")
        self.prev: str | None = None
        self.fade_t0: float | None = None
        self._paused: set[str] = set()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        # Start everything; engines load concurrently in their own processes.
        for e in self.engines.values():
            log.info("starting engine %s (%s, %s)", e.name, e.backend.name, Path(e.config_path).name)
            e.backend.start()
        for e in self.engines.values():
            if e.name != self.active:
                self._pause(e)

    def stop(self) -> None:
        for e in self.engines.values():
            try:
                e.backend.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("stopping %s: %s", e.name, exc)

    def _pause(self, e: _Engine) -> None:
        if e.is_fluxrt:
            e.backend.set_param("paused", True)
        self._paused.add(e.name)

    def _resume(self, e: _Engine) -> None:
        if e.is_fluxrt:
            e.backend.set_param("paused", False)
        self._paused.discard(e.name)

    # -- readiness / prompts ---------------------------------------------------
    def _send_initial_prompts(self) -> None:
        for e in self.engines.values():
            if not e.prompted and not e.is_fluxrt and e.backend.is_ready() and e.cycle:
                e.backend.set_prompt_index(0, e.cycle[0])
                e.prompted = True
            elif e.is_fluxrt:
                e.prompted = True  # FluxRT pre-encodes its own cycle

    def is_ready(self) -> bool:
        self._send_initial_prompts()
        return self.engines[self.active].backend.is_ready()

    def alive(self) -> bool:
        return self.engines[self.active].backend.alive()

    def all_ready(self) -> bool:
        return all(e.backend.is_ready() for e in self.engines.values())

    # -- switching -------------------------------------------------------------
    def switch(self, name: str) -> None:
        if name not in self.engines:
            raise KeyError(f"unknown engine {name!r}; have {list(self.engines)}")
        if name == self.active:
            return
        new = self.engines[name]
        if not new.backend.is_ready():
            raise RuntimeError(f"engine {name!r} is still loading")
        if self.prev is not None and self.prev != name:
            self._pause(self.engines[self.prev])   # a switch during a fade drops the oldest engine
        self._resume(new)
        self.prev, self.active = self.active, name
        # let the new engine produce fresh frames before blending towards it
        self.fade_t0 = time.monotonic() + 0.3
        log.info("switch %s -> %s (crossfade %.1fs)", self.prev, name, self.crossfade_s)

    def _fade_alpha(self) -> float | None:
        if self.prev is None or self.fade_t0 is None:
            return None
        a = (time.monotonic() - self.fade_t0) / max(self.crossfade_s, 1e-3)
        if a >= 1.0:
            self._pause(self.engines[self.prev])
            self.prev, self.fade_t0 = None, None
            return None
        return max(0.0, a)

    # -- frames ----------------------------------------------------------------
    def push_input(self, bgr: np.ndarray) -> None:
        feed = [self.active] + ([self.prev] if self.prev else [])
        for name in feed:
            e = self.engines[name]
            e.backend.push_input(np.ascontiguousarray(_fit(bgr, e.backend.resolution)))

    def _out(self, e: _Engine) -> np.ndarray | None:
        o = e.backend.current_output_frame()
        if o is not None:
            e.last_out = _fit(o, self.out_resolution)
        return e.last_out

    def current_output_frame(self) -> np.ndarray | None:
        self._send_initial_prompts()
        alpha = self._fade_alpha()
        cur = self._out(self.engines[self.active])
        if alpha is None or self.prev is None:
            return cur
        old = self._out(self.engines[self.prev])
        if cur is None:
            return old
        if old is None:
            return cur
        return cv2.addWeighted(cur, alpha, old, 1.0 - alpha, 0.0)

    # -- controls (go to the active engine) ------------------------------------
    def set_prompt(self, text: str) -> None:
        e = self.engines[self.active]
        e.custom_prompt = text
        e.backend.set_prompt(text)

    def set_prompt_index(self, idx: int, text: str) -> None:
        e = self.engines[self.active]
        e.prompt_index, e.custom_prompt = idx, None
        e.backend.set_prompt_index(idx, text)

    def set_param(self, name: str, value) -> None:
        self.engines[self.active].backend.set_param(name, value)

    def stats(self) -> dict:
        st = dict(self.engines[self.active].backend.stats())
        st["active_engine"] = self.active
        st["fading_from"] = self.prev
        st["engines"] = [
            {"name": e.name, "backend": e.backend.name, "config": Path(e.config_path).stem,
             "ready": e.backend.is_ready(), "alive": e.backend.alive(), "paused": e.name in self._paused,
             "resolution": f"{e.backend.resolution[1]}x{e.backend.resolution[0]}"}
            for e in self.engines.values()
        ]
        # total reserved GPU memory across engines (each reports its own process)
        st["gpu_reserved_mb_total"] = int(sum((e.backend.stats() or {}).get("gpu_reserved_mb") or 0 for e in self.engines.values()))
        return st
