"""Backend interface for the web server, plus the in-process FluxRT backend.

A backend turns camera frames (BGR uint8) into styled frames. The server owns
prompt cycling and WebRTC; the backend only needs latest-wins frame I/O plus a
few live controls.
"""

from __future__ import annotations

import numpy as np


class Backend:
    name = "base"

    # (height, width) of the frames the backend consumes / produces
    resolution: tuple[int, int]
    out_resolution: tuple[int, int]

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def is_ready(self) -> bool:
        return False

    def alive(self) -> bool:
        return True

    def push_input(self, bgr: np.ndarray) -> None: ...

    def current_output_frame(self) -> np.ndarray | None:
        """Latest styled frame, or None if nothing has been produced yet."""
        return None

    def set_prompt(self, text: str) -> None: ...

    def set_prompt_index(self, idx: int, text: str) -> None:
        """Switch to cycle prompt `idx` (text given for backends without a cache)."""
        self.set_prompt(text)

    def set_param(self, name: str, value) -> None: ...

    def stats(self) -> dict:
        return {}


class FluxRTBackend(Backend):
    """Wraps fluxrt.StreamProcessor (FLUX.2-klein stream editing)."""

    name = "fluxrt"

    def __init__(self, config_path: str, cfg: dict, force_int8: bool = False):
        from fluxrt import StreamProcessor

        self.sp = StreamProcessor(config_path)
        self._lip_available = bool((cfg.get("lip_transfer") or {}).get("enable"))
        self._lip_active = self._lip_available and bool((cfg.get("lip_transfer") or {}).get("start_active"))
        if force_int8 or cfg.get("enable_int8_quantization", False):
            self.sp.enable_quantization()
        self.in_t = self.sp.get_input_tensor()
        self.out_t = self.sp.get_output_tensor()
        r, o = self.sp.get_resolution(), self.sp.get_out_resolution()
        self.resolution = (r["height"], r["width"])
        self.out_resolution = (o["height"], o["width"])

    def start(self) -> None:
        self.sp.start()

    def stop(self) -> None:
        self.sp.stop()

    def is_ready(self) -> bool:
        return self.sp.is_ready()

    def alive(self) -> bool:
        p = getattr(self.sp.model_inference_subprocess, "process", None)
        return True if p is None else p.is_alive()

    def push_input(self, bgr: np.ndarray) -> None:
        self.in_t.copy_from(np.ascontiguousarray(bgr))

    def current_output_frame(self) -> np.ndarray | None:
        return self.out_t.to_numpy() if self.sp.is_ready() else None

    def set_prompt(self, text: str) -> None:
        self.sp.set_prompt(text)

    def set_prompt_index(self, idx: int, text: str) -> None:
        self.sp.set_prompt_index(idx)  # pre-encoded, instant

    def set_param(self, name: str, value) -> None:
        if name == "lip_transfer":          # LivePortrait expression/lip transfer on/off
            self._lip_active = bool(value) and self._lip_available
            self.sp.set_lip_transfer(bool(value))
        else:
            self.sp.set_gen_param(name, value)

    def stats(self) -> dict:
        proc = self.sp.get_last_processing_time()
        return {
            "proc_time_s": round(proc, 4),
            "gpu_reserved_mb": self.sp.get_reserved_memory(),
            "interpolation_exp": self.sp.interpolation_exp_value.value,
            **({"lip_transfer": {"available": True, "active": self._lip_active}} if self._lip_available else {}),
        }
