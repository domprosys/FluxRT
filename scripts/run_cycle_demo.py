"""Live webcam demo that cycles through pre-encoded prompts.

All prompts in config['prompt_cycle'] are text-encoded once at startup (on the
CPU-offloaded encoder), so switching between them mid-stream is instant — no
per-switch re-encode, no freeze. Shows input | output side-by-side with the
current prompt overlaid; auto-advances every --interval seconds.
"""
import argparse
import json
import time

import cv2
import numpy as np

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cycle_demo_config.json")
    ap.add_argument("--interval", type=float, default=6.0,
                    help="seconds between prompt switches")
    ap.add_argument("--int8", action="store_true", default=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    prompts = cfg["prompt_cycle"]

    sp = StreamProcessor(args.config)
    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()
    sp.enable_quantization()
    sp.start()

    res = sp.get_resolution()
    h, w = res["height"], res["width"]
    cap = cv2.VideoCapture(0)

    print("Initializing (loading model + pre-encoding all prompts)…", flush=True)
    while not sp.is_ready():
        time.sleep(0.1)
    print("Ready. Cycling prompts; press 'q' in the window to quit.", flush=True)

    idx = 0
    sp.set_prompt_index(0)
    last_switch = time.time()
    win = "FluxRT — prompt cycle (input | output)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = crop_maximal_rectangle(frame, h, w)
        input_tensor.copy_from(cropped)
        out_bgr = output_tensor.to_numpy()

        # advance prompt on the interval (instant swap of pre-encoded embeds)
        if time.time() - last_switch >= args.interval:
            idx = (idx + 1) % len(prompts)
            sp.set_prompt_index(idx)
            last_switch = time.time()
            print(f"[cycle] -> [{idx}] {prompts[idx]}", flush=True)

        combo = np.hstack([cropped, out_bgr])
        label = f"[{idx+1}/{len(prompts)}] {prompts[idx]}"
        cv2.rectangle(combo, (0, 0), (combo.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(combo, label[:80], (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, combo)
        if cv2.waitKey(1000 // 25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    sp.stop()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
