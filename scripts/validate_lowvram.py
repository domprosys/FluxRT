"""Headless validation for the low-VRAM (CPU text-encoder offload) path.

Drives StreamProcessor with a synthetic frame, measures peak GPU memory of the
inference subprocess, steady-state cadence, and prompt-change re-encode latency.
Pass --fraction 0.5 to hard-cap the subprocess to ~12GB on a 24GB card.

Robust: detects inference-subprocess death, always reaps children (no orphans),
and enables faulthandler so hard crashes surface a C-level traceback.
"""
import argparse
import faulthandler
import subprocess
import threading
import time

import numpy as np

faulthandler.enable()

from fluxrt import StreamProcessor


def gpu_python_mem_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return 0
    total = 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        name, mem = [x.strip() for x in line.split(",")]
        if "python" in name.lower():
            total += int(mem)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lowvram_config.json")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--ready-timeout", type=float, default=420.0)
    args = ap.parse_args()

    sp = StreamProcessor(args.config)
    if args.fraction is not None:
        sp.config["cuda_memory_fraction"] = args.fraction
    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()

    sp.enable_quantization()
    sp.start()

    # Handle to the inference subprocess so we can detect a crash.
    infer_proc = sp.model_inference_subprocess.process

    peak = {"mib": 0, "reserved": 0, "reserved_last": 0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["mib"] = max(peak["mib"], gpu_python_mem_mib())
            try:
                r = sp.get_reserved_memory()
                peak["reserved"] = max(peak["reserved"], r)
                if r:
                    peak["reserved_last"] = r
            except Exception:
                pass
            time.sleep(0.25)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    try:
        sp.set_prompt("Turn this into oil on canvas art in style of Wassily Kandinsky.")
        res = sp.get_resolution()
        h, w = res["height"], res["width"]
        print(f"[validate] resolution = {w}x{h}", flush=True)
        base = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        print("[validate] waiting for model load + first compiled frame…", flush=True)
        t0 = time.time()
        while not sp.is_ready():
            if infer_proc is not None and not infer_proc.is_alive():
                raise RuntimeError(
                    f"inference subprocess died during init/compile "
                    f"(exitcode={infer_proc.exitcode})")
            if time.time() - t0 > args.ready_timeout:
                raise TimeoutError("model never became ready")
            time.sleep(0.2)
        print(f"[validate] ready after {time.time()-t0:.1f}s; "
              f"peak so far {peak['mib']}MiB", flush=True)

        times = []
        for i in range(args.iters):
            if infer_proc is not None and not infer_proc.is_alive():
                raise RuntimeError("inference subprocess died during streaming")
            input_tensor.copy_from(np.ascontiguousarray(np.roll(base, i, axis=1)))
            ts = time.time()
            _ = output_tensor.to_numpy()
            dt = time.time() - ts
            if i >= args.iters - 40:
                times.append(dt)
            if i % 20 == 0:
                print(f"[validate] iter {i:3d}  read_dt={dt*1000:6.1f}ms  "
                      f"peak={peak['mib']}MiB", flush=True)

        steady = sorted(times)[len(times)//2] if times else float("nan")
        print(f"[validate] steady read ~{steady*1000:.1f}ms", flush=True)

        print("[validate] changing prompt (CPU re-encode) …", flush=True)
        tp = time.time()
        sp.set_prompt("Make it a neon cyberpunk city at night, heavy rain, cinematic.")
        for i in range(80):
            input_tensor.copy_from(np.ascontiguousarray(np.roll(base, i, axis=1)))
            _ = output_tensor.to_numpy()
            time.sleep(0.02)
        print(f"[validate] prompt-change window {time.time()-tp:.2f}s "
              f"(see re-encode log)", flush=True)

        print(f"[validate] PEAK GPU (sum python procs) = {peak['mib']} MiB "
              f"({peak['mib']/1024:.2f} GiB)", flush=True)
        print(f"[validate] FluxRT torch.cuda.memory_reserved peak = "
              f"{peak['reserved']} MiB ({peak['reserved']/1024:.2f} GiB), "
              f"steady = {peak['reserved_last']} MiB "
              f"({peak['reserved_last']/1024:.2f} GiB)", flush=True)
        print(f"[validate] est. total card VRAM = reserved_steady + ~0.6GB "
              f"CUDA ctx ≈ {(peak['reserved_last']+600)/1024:.2f} GiB", flush=True)
        print("[validate] RESULT OK", flush=True)
    finally:
        stop.set()
        try:
            sp.stop()
        except Exception as e:
            print(f"[validate] stop() error: {e}", flush=True)
        print("[validate] cleaned up.", flush=True)


if __name__ == "__main__":
    main()
