"""Exercise a backend without the web server, optionally as a benchmark.

    .venv/bin/python scripts/test_backend.py --config configs/sd_config.json [--device 0|-1] [--seconds 20] [--out DIR]
    .venv/bin/python scripts/test_backend.py --config configs/sd_config.json --video clip.mp4 --seconds 40 \
        --warmup 8 --json result.json --samples 50,100,150,200

Input: webcam (--device N), a synthetic moving pattern (--device -1), or a video file
(--video, looped) fed at --fps. Prints backend stats once a second and saves the last
input/output pair. Benchmark extras: per-sample generation time after --warmup seconds
(median / p95), output frame rate (distinct output frames per second, which includes
interpolated frames for fluxrt), GPU memory, model load time, output samples at fixed
input-frame indices, all written to --json.
"""
import argparse, json, logging, statistics, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_web import build_backend  # noqa: E402
from fluxrt.utils import crop_maximal_rectangle  # noqa: E402


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", type=int, default=0, help="-1 = synthetic")
    ap.add_argument("--video", default=None, help="video file to loop as input (overrides --device)")
    ap.add_argument("--static", action="store_true", help="feed only the video's first frame: any output change is flicker")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--warmup", type=float, default=0, help="seconds excluded from benchmark stats")
    ap.add_argument("--samples", default="", help="comma-separated input frame indices to save outputs for")
    ap.add_argument("--json", default=None, help="write benchmark results here")
    ap.add_argument("--out", default=".cache/test_backend")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="FRAME:NAME=VALUE",
                    help="send set_param NAME=VALUE (JSON value) when input frame FRAME is pushed, e.g. 100:faceid_capture=true")
    a = ap.parse_args()
    # worker log events (load/warm-up/reset timings) reach the backend's logger at INFO
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = json.load(open(a.config))
    be = build_backend(a.config, cfg)
    h, w = be.resolution
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    be.start()
    t0 = time.time()
    while not be.is_ready():
        if not be.alive():
            print("worker died during setup", flush=True)
            be.stop()  # also stops helper processes that would otherwise keep the interpreter alive
            return 1
        time.sleep(0.5)
    ready_s = time.time() - t0
    print(f"ready in {ready_s:.1f}s, resolution {w}x{h}, out {be.out_resolution[1]}x{be.out_resolution[0]}", flush=True)
    prompt = a.prompt or (cfg.get("prompt_cycle") or [cfg.get("default_prompt", "a painting")])[0]
    be.set_prompt(prompt)

    video = None
    if a.video:
        video = cv2.VideoCapture(a.video)
        if not video.isOpened():
            print(f"cannot open {a.video}"); be.stop(); return 1
    cap = cv2.VideoCapture(a.device) if (video is None and a.device >= 0) else None
    sample_at = {int(x) for x in a.samples.split(",") if x.strip()}
    timed = {}
    for spec in a.set:
        f, kv = spec.split(":", 1); k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except json.JSONDecodeError: pass
        timed.setdefault(int(f), []).append((k, v))

    n = 0; last_stat = time.time(); t_start = time.time(); last_out = None; frame = None
    gen_samples, gpu_samples = [], []
    out_changes = 0; last_sig = None; bench_t0 = None
    diffs = []; prev_small = None; static_frame = None
    pending_samples = {}
    while time.time() - t_start < a.seconds:
        if video is not None and static_frame is not None:
            frame = static_frame
        elif video is not None:
            ok, frame = video.read()
            if not ok:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, frame = video.read()
            if a.static:
                static_frame = frame
        else:
            ok, frame = (cap.read() if cap is not None else (False, None))
            if not ok:
                frame = np.zeros((720, 1280, 3), np.uint8)
                cv2.circle(frame, (640 + int(300 * np.sin(n / 15)), 360), 120, (40, 200, 255), -1)
                cv2.putText(frame, f"synthetic {n}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        inp = crop_maximal_rectangle(frame, h, w)
        be.push_input(inp)
        for k, v in timed.get(n, []):
            print(f"frame {n}: set_param {k}={v!r}", flush=True); be.set_param(k, v)
        if n in sample_at:
            # the matching output appears ~one generation later; save it a few frames on
            cv2.imwrite(str(out_dir / f"in_{n:04d}.png"), inp)
            pending_samples[n + 6] = n
        n += 1
        out = be.current_output_frame()
        in_bench = time.time() - t_start >= a.warmup
        if out is not None:
            last_out = out
            sig = int(out[::16, ::16].astype(np.int64).sum())
            if in_bench:
                if bench_t0 is None:
                    bench_t0 = time.time()
                if last_sig is not None and sig != last_sig:
                    out_changes += 1
                    small = cv2.cvtColor(cv2.resize(out, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if prev_small is not None:
                        diffs.append(float(np.abs(small - prev_small).mean()))
                    prev_small = small
            last_sig = sig
            # first output at or after the target frame (some backends return None between new frames)
            for k in [k for k in pending_samples if n >= k]:
                cv2.imwrite(str(out_dir / f"out_{pending_samples.pop(k):04d}.png"), out)
        if in_bench:  # every input frame (~25 Hz): short latency spikes show up in p99 / max
            st = be.stats()
            if st.get("proc_time_s"):
                gen_samples.append(float(st["proc_time_s"]))
            if st.get("gpu_reserved_mb"):
                gpu_samples.append(int(st["gpu_reserved_mb"]))
        if time.time() - last_stat >= 1.0:
            st = be.stats()
            print(f"t={time.time()-t_start:5.1f}s in={n} proc={st.get('proc_time_s')}s gpu={st.get('gpu_reserved_mb')}MB alive={be.alive()} err={st.get('last_error')}", flush=True)
            last_stat = time.time()
        time.sleep(1.0 / a.fps)

    if frame is not None: cv2.imwrite(str(out_dir / "in_last.png"), crop_maximal_rectangle(frame, h, w))
    if last_out is not None: cv2.imwrite(str(out_dir / "out_last.png"), last_out)
    stats = be.stats()
    print("saved", a.out, "| stats:", stats, flush=True)
    if a.json:
        p50, p95 = pct(gen_samples, 0.5), pct(gen_samples, 0.95)
        bench_secs = (time.time() - bench_t0) if bench_t0 else 0
        res = {
            "config": Path(a.config).stem, "backend": cfg.get("backend", "fluxrt"),
            "resolution": f"{w}x{h}", "ready_s": round(ready_s, 1), "prompt": prompt[:80],
            "gen_ms_p50": round(1000 * p50, 1) if p50 else None, "gen_ms_p95": round(1000 * p95, 1) if p95 else None,
            "gen_ms_p99": round(1000 * pct(gen_samples, 0.99), 1) if gen_samples else None,
            "gen_ms_max": round(1000 * max(gen_samples), 1) if gen_samples else None,
            # share of samples slower than 1.5x the median (a spike lasts until the next frame is done)
            "spike_frac": round(sum(v > 1.5 * p50 for v in gen_samples) / len(gen_samples), 4) if p50 else None,
            "gen_fps": round(1 / p50, 1) if p50 else None,
            "out_fps": round(out_changes / bench_secs, 1) if bench_secs else None,
            "gpu_mb": max(gpu_samples) if gpu_samples else stats.get("gpu_reserved_mb"),
            "samples": len(gen_samples), "bench_s": round(bench_secs, 1),
            # mean abs change between consecutive distinct output frames (0-255 scale, 160x90 gray);
            # with --static input this is pure flicker, with moving input it is motion + flicker
            "temporal_diff": round(sum(diffs) / len(diffs), 2) if diffs else (0.0 if last_out is not None else None),
            "static_input": bool(a.static),
            "interpolation_exp": stats.get("interpolation_exp"),
            "stats_last": {k: v for k, v in stats.items() if k in ("proc_time_s", "model", "warmup_fps", "chunk_size", "native_fps", "vae_type",
                                                                  "acceleration", "unet_runtime", "ipadapter", "lip_transfer", "attention", "model_size",
                                                                  "use_cached_attn", "cache_maxframes", "text_encoder_device")},
        }
        Path(a.json).write_text(json.dumps(res, indent=2))
        print("BENCH " + json.dumps(res), flush=True)
    be.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
