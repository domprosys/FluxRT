"""Exercise a worker backend without the web server.

    .venv/bin/python scripts/test_backend.py --config configs/sd_config.json [--device 0|-1] [--seconds 20] [--out DIR]

Feeds webcam (or a synthetic moving pattern with --device -1) frames at ~25 fps,
prints backend stats once a second, saves the last input/output pair as PNGs.
Works for any config whose "backend" is a worker (sd, sdv2); "fluxrt" runs in-process.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_web import build_backend  # noqa: E402
from fluxrt.utils import crop_maximal_rectangle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", type=int, default=0, help="-1 = synthetic")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--out", default=".cache/test_backend")
    ap.add_argument("--prompt", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    be = build_backend(a.config, cfg)
    h, w = be.resolution
    be.start()
    t0 = time.time()
    while not be.is_ready():
        if not be.alive():
            print("worker died during setup"); return 1
        time.sleep(0.5)
    print(f"ready in {time.time()-t0:.1f}s, resolution {w}x{h}, out {be.out_resolution[1]}x{be.out_resolution[0]}", flush=True)
    prompt = a.prompt or (cfg.get("prompt_cycle") or [cfg.get("default_prompt", "a painting")])[0]
    be.set_prompt(prompt)
    cap = cv2.VideoCapture(a.device) if a.device >= 0 else None
    n = 0; last_stat = time.time(); t_start = time.time(); last_seq = None; out_count = 0; last_out = None; frame = None
    while time.time() - t_start < a.seconds:
        ok, frame = (cap.read() if cap is not None else (False, None))
        if not ok:
            frame = np.zeros((720, 1280, 3), np.uint8)
            cv2.circle(frame, (640 + int(300 * np.sin(n / 15)), 360), 120, (40, 200, 255), -1)
            cv2.putText(frame, f"synthetic {n}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        n += 1
        be.push_input(crop_maximal_rectangle(frame, h, w))
        out = be.current_output_frame()
        if out is not None:
            last_out = out
            s = be.stats().get("output_seq")
        if time.time() - last_stat >= 1.0:
            st = be.stats()
            print(f"t={time.time()-t_start:5.1f}s in={n} proc={st.get('proc_time_s')}s gpu={st.get('gpu_reserved_mb')}MB alive={be.alive()} err={st.get('last_error')}", flush=True)
            last_stat = time.time()
        time.sleep(1.0 / a.fps)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if frame is not None: cv2.imwrite(f"{a.out}/in_last.png", crop_maximal_rectangle(frame, h, w))
    if last_out is not None: cv2.imwrite(f"{a.out}/out_last.png", last_out)
    print("saved", a.out, "| stats:", be.stats(), flush=True)
    be.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
