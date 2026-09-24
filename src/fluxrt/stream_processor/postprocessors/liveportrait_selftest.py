"""Self-test / micro-benchmark of the LivePortrait post-processor, without the FLUX model.

    .venv/bin/python -m fluxrt.stream_processor.postprocessors.liveportrait_selftest \
        [--config configs/web_bf16_liveportrait_config.json] [--models-dir LivePortrait/liveportrait] \
        [--source IMG] [--driving IMG] [--size 640x368] [--iters 30] [--device cuda] \
        [--region lip] [--relative 0|1] [--source-crop detect|driving] [--out FILE.jpg]

Builds the processor from the config's "lip_transfer" block (like the FluxRT worker), runs it on a
still "generated" face (--source, default LivePortrait's Mona Lisa example) driven by a still
"webcam" face (--driving, default an open-mouth example), letterboxed to --size, and prints load
time, the onnxruntime providers, per-frame ms (with a detect / net / paste breakdown) and the
no-face paths. Writes driving | source | result side by side to --out. Exit code 1 if the face
was not re-animated.
"""
import argparse
import json
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch

from fluxrt.stream_processor.postprocessors.liveportrait import _REPO_ROOT, LivePortraitPostProcessor

_EX = osp.join(_REPO_ROOT, "LivePortrait-code", "assets", "examples")
_FALLBACK = osp.join(_REPO_ROOT, "LivePortrait-code", "src", "utils", "dependencies", "insightface", "data", "images", "t1.jpg")


def letterbox(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    s = min(w / img_bgr.shape[1], h / img_bgr.shape[0])
    nw, nh = int(round(img_bgr.shape[1] * s)), int(round(img_bgr.shape[0] * s))
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def load_rgb(path: str, w: int, h: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(letterbox(img, w, h), cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="FluxRT config with a lip_transfer block")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--driving", default=None)
    ap.add_argument("--size", default="640x368")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--relative", type=int, default=None, help="1 = original additive formula, 0 = replace")
    ap.add_argument("--source-crop", default=None)
    ap.add_argument("--out", default=".cache/liveportrait_selftest.jpg")
    a = ap.parse_args()

    cfg = {"models_dir": "LivePortrait/liveportrait"}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg.update(json.load(f).get("lip_transfer", {}))
    for key, val in (("models_dir", a.models_dir), ("device", a.device), ("region", a.region),
                     ("source_crop", a.source_crop)):
        if val is not None:
            cfg[key] = val
    if a.relative is not None:
        cfg["relative"] = bool(a.relative)
    cfg["log_interval_s"] = 0
    w, h = (int(v) for v in a.size.lower().split("x"))

    drv_path = a.driving or (osp.join(_EX, "driving", "d19.jpg") if osp.isfile(osp.join(_EX, "driving", "d19.jpg")) else _FALLBACK)
    src_path = a.source or (osp.join(_EX, "source", "s9.jpg") if osp.isfile(osp.join(_EX, "source", "s9.jpg")) else _FALLBACK)
    if cfg.get("source_crop") == "driving" and not a.source:
        src_path = drv_path  # this mode assumes the generated frame is aligned with the webcam frame
    source, driving = load_rgb(src_path, w, h), load_rgb(drv_path, w, h)
    print(f"source {src_path}\ndriving {drv_path}\nsize {w}x{h}  options {cfg}", flush=True)

    proc = LivePortraitPostProcessor.from_config(cfg)
    print(f"load {proc.load_s:.1f}s  torch device {proc.device}  detector providers {proc.detector_providers}", flush=True)

    # no-face paths must hand the generated frame back untouched
    blank = np.zeros_like(source)
    tm = {}
    out, st = proc._process(source, blank, tm)
    ok_nodrv = st == "no_driving_face" and out is source
    out, st2 = proc._process(blank, driving, tm) if proc.source_crop == "detect" else (blank, "no_source_face")
    ok_nosrc = st2 == "no_source_face" and out is blank
    print(f"no driving face -> {st} ({'ok' if ok_nodrv else 'FAIL'}); no source face -> {st2} ({'ok' if ok_nosrc else 'FAIL'})", flush=True)

    result, status = source, None
    ms, parts = [], {}
    for i in range(max(1, a.iters)):
        tm = {}
        t0 = time.perf_counter()
        result, status = proc._process(source, driving, tm)
        ms.append(1000 * (time.perf_counter() - t0))
        if i >= min(3, a.iters - 1):  # skip the first iterations in the breakdown
            for k, v in tm.items():
                parts.setdefault(k, []).append(1000 * v)
        if status != "applied":
            break
    public = proc.process(source, driving)  # the never-raising entry point the worker calls
    steady = sorted(ms[min(3, len(ms) - 1):])
    diff = float(np.abs(result.astype(np.int16) - source.astype(np.int16)).mean())
    print(f"status {status}  public process() -> {proc.last_status}  mean |result-source| {diff:.2f}", flush=True)
    if steady:
        print(
            f"per frame: first {ms[0]:.1f} ms, steady p50 {steady[len(steady) // 2]:.1f} ms, "
            f"p95 {steady[min(len(steady) - 1, int(0.95 * len(steady)))]:.1f} ms | "
            + " ".join(f"{k} {np.median(v):.1f}" for k, v in sorted(parts.items())),
            flush=True,
        )
    if torch.cuda.is_available():
        print(f"GPU max allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)
    if a.out:
        import os
        os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
        sheet = np.hstack([driving, source, public])
        cv2.imwrite(a.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"wrote {a.out} (driving | source | result)", flush=True)
    good = status == "applied" and diff > 0.1 and ok_nodrv and ok_nosrc
    print("SELFTEST " + ("PASS" if good else "FAIL"), flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
