"""Shared-memory + JSON-lines protocol between the web server and a backend
worker that runs in a DIFFERENT Python environment (its own venv, torch, etc.).

Keep this module dependency-light (stdlib + numpy only) and free of project
imports: workers import it by file path from whatever venv they run in.

Data path (latest-wins, no queues):
    parent  --(input FrameSlot, BGR uint8 HxWx3)-->  worker
    worker  --(output FrameSlot, BGR uint8 HxWx3)--> parent
    Meta: small float64 array of counters/stats shared both ways.

Control path:
    parent -> worker : JSON lines on the worker's stdin
        {"cmd": "set_prompt", "text": "..."}
        {"cmd": "set_param", "name": "...", "value": ...}
        {"cmd": "stop"}
    worker -> parent : JSON lines on the worker's ORIGINAL stdout (fd 1)
        {"event": "ready"} | {"event": "log", "msg": "..."} | {"event": "error", "msg": "..."}
        {"event": "info", ...}  (free-form, e.g. model name, resolution)
    The worker redirects Python's sys.stdout to stderr at startup so stray
    prints from libraries never corrupt the event stream.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
import traceback
from multiprocessing import resource_tracker, shared_memory

# Cap the CPU thread pools (OpenMP/MKL size them to the host's cores: 224 on a RunPod PRO 6000 host,
# 469 threads in one worker). Pods get a CPU quota (cpu.max ~24 CPUs); spinning pools exhausted it and
# the kernel paused the process for the rest of each 100 ms period: 30-40% of frames took +50 ms
# (2026-09-25). FLUXRT_THREADS overrides the default 4. Must run before numpy/torch are imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("FLUXRT_THREADS", "4"))


def cap_onnxruntime_threads() -> None:
    """Default onnxruntime sessions to FLUXRT_THREADS non-spinning intra-op threads. insightface (FaceID,
    LivePortrait's face detector) builds its sessions with default options: one thread per physical core,
    spinning while idle, which eats a pod's CPU quota like the OpenMP pools did. Explicit options win."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if getattr(ort.InferenceSession, "_fluxrt_capped", False):
        return
    n = int(os.environ.get("FLUXRT_THREADS", "4"))
    init0 = ort.InferenceSession.__init__

    def __init__(self, path_or_bytes, sess_options=None, *args, **kw):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        init0(self, path_or_bytes, sess_options, *args, **kw)

    ort.InferenceSession.__init__ = __init__
    ort.InferenceSession._fluxrt_capped = True

import numpy as np


def _attach(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment without the resource tracker later trying
    to unlink it at exit (the creating process owns the lifetime)."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass
    return shm

META_FIELDS = (
    "input_seq",      # parent increments after writing a new input frame
    "output_seq",     # worker increments after writing a new output frame
    "ready",          # worker sets 1 when the model is loaded
    "proc_time_ms",   # worker: last per-frame processing time
    "gpu_mb",         # worker: reserved GPU memory (best effort)
    "heartbeat",      # worker: time.time() of last loop iteration
)


class FrameSlot:
    """One uint8 HxWx3 frame in shared memory. Parent creates, worker attaches."""

    def __init__(self, shape: tuple[int, int, int], name: str | None = None, create: bool = False):
        self.shape = tuple(int(x) for x in shape)
        size = int(np.prod(self.shape))
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
        else:
            self.shm = _attach(name)
        self.array = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        self._owner = create

    @property
    def name(self) -> str:
        return self.shm.name

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape:
            raise ValueError(f"frame shape {frame.shape} != slot shape {self.shape}")
        np.copyto(self.array, frame)

    def read(self) -> np.ndarray:
        return self.array.copy()

    def close(self) -> None:
        try:
            self.shm.close()
            if self._owner:
                self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass


class Meta:
    """Small float64 vector of counters/stats, addressed by META_FIELDS name."""

    def __init__(self, name: str | None = None, create: bool = False):
        n = len(META_FIELDS)
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=n * 8)
        else:
            self.shm = _attach(name)
        self.array = np.ndarray((n,), dtype=np.float64, buffer=self.shm.buf)
        if create:
            self.array[:] = 0.0
        self._owner = create

    @property
    def name(self) -> str:
        return self.shm.name

    def get(self, field: str) -> float:
        return float(self.array[META_FIELDS.index(field)])

    def set(self, field: str, value: float) -> None:
        self.array[META_FIELDS.index(field)] = float(value)

    def inc(self, field: str) -> None:
        self.array[META_FIELDS.index(field)] += 1.0

    def close(self) -> None:
        try:
            self.shm.close()
            if self._owner:
                self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass


def worker_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-shm", required=True)
    ap.add_argument("--out-shm", required=True)
    ap.add_argument("--meta-shm", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out-height", type=int, default=None)
    ap.add_argument("--out-width", type=int, default=None)
    ap.add_argument("--config", required=True, help="path to a JSON file with the backend config")
    return ap


class WorkerBase:
    """Subclass this in a worker script and implement setup / process / on_command.

    Run loop: wait for a new input frame (input_seq changed), call process(),
    write the result, bump output_seq. Commands from stdin are applied between
    frames. Any exception in process() is reported and the loop continues, so a
    bad live parameter never kills the stream.
    """

    idle_sleep_s = 0.002

    def __init__(self, argv: list[str] | None = None):
        args = worker_arg_parser().parse_args(argv)
        self.args = args
        with open(args.config) as f:
            self.cfg: dict = json.load(f)
        self.height, self.width = args.height, args.width
        self.out_height = args.out_height or args.height
        self.out_width = args.out_width or args.width

        # Reserve the original stdout for events; send everything else to stderr.
        self._events = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr

        self.in_slot = FrameSlot((self.height, self.width, 3), name=args.in_shm)
        self.out_slot = FrameSlot((self.out_height, self.out_width, 3), name=args.out_shm)
        self.meta = Meta(name=args.meta_shm)
        self._cmds: queue.Queue = queue.Queue()
        self._stop = False

    # -- events ---------------------------------------------------------------
    def emit(self, event: str, **kw) -> None:
        self._events.write(json.dumps({"event": event, **kw}) + "\n")
        self._events.flush()

    def log(self, msg: str) -> None:
        self.emit("log", msg=str(msg))

    # -- to implement -----------------------------------------------------------
    def setup(self) -> None:
        """Load models. Called once. May call self.log()."""
        raise NotImplementedError

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return the styled frame (BGR uint8, out_height x out_width x 3).

        Return None to emit nothing for this input (e.g. a block/chunk model that
        is still accumulating frames). Such workers publish results themselves
        with self.write_output(frame), possibly several per input or from a
        background thread."""
        raise NotImplementedError

    def write_output(self, out: np.ndarray, proc_time_s: float | None = None) -> None:
        out = np.ascontiguousarray(out, dtype=np.uint8)
        self.out_slot.write(out)
        self.meta.inc("output_seq")
        if proc_time_s is not None:
            self.meta.set("proc_time_ms", proc_time_s * 1000.0)
        if int(self.meta.get("output_seq")) % 30 == 0:
            self.meta.set("gpu_mb", self.gpu_mb())

    def on_command(self, name: str, payload: dict) -> None:
        """Handle set_prompt / set_param. Unknown commands may be ignored."""
        raise NotImplementedError

    def gpu_mb(self) -> float:
        try:
            import torch

            return torch.cuda.memory_reserved() / 2**20
        except Exception:  # noqa: BLE001
            return 0.0

    # -- loop -------------------------------------------------------------------
    def _stdin_reader(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                self._cmds.put(json.loads(line))
            except json.JSONDecodeError:
                self.emit("error", msg=f"bad command line: {line[:80]}")
        self._cmds.put({"cmd": "stop"})  # parent closed stdin

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return
            name = cmd.get("cmd")
            if name == "stop":
                self._stop = True
                return
            try:
                self.on_command(name, cmd)
            except Exception:  # noqa: BLE001
                self.emit("error", msg=f"command {name} failed: {traceback.format_exc()}")

    def run(self) -> int:
        threading.Thread(target=self._stdin_reader, daemon=True).start()
        try:
            self.setup()
        except Exception:  # noqa: BLE001
            self.emit("error", msg=f"setup failed: {traceback.format_exc()}")
            return 1
        # Move everything setup created (models, modules: millions of objects) out of the cyclic GC's
        # reach, so its periodic full collections don't pause the frame loop for tens of ms.
        # FLUXRT_GC_FREEZE=0 disables this (A/B for the latency spikes seen on 2026-09-25).
        if os.environ.get("FLUXRT_GC_FREEZE", "1") != "0":
            gc.collect()
            gc.freeze()
        self.meta.set("ready", 1)
        self.meta.set("gpu_mb", self.gpu_mb())
        self.emit("ready")

        last_in = -1.0
        while not self._stop:
            self._drain_commands()
            if self._stop:
                break
            self.meta.set("heartbeat", time.time())
            seq = self.meta.get("input_seq")
            if seq == last_in:
                time.sleep(self.idle_sleep_s)
                continue
            last_in = seq
            frame = self.in_slot.read()
            t0 = time.perf_counter()
            try:
                out = self.process(frame)
            except Exception:  # noqa: BLE001
                self.emit("error", msg=f"process failed: {traceback.format_exc()}")
                time.sleep(0.05)
                continue
            if out is not None:
                self.write_output(out, proc_time_s=time.perf_counter() - t0)
        self.emit("log", msg="worker stopping")
        return 0
