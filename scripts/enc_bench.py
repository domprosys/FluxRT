"""Micro-benchmark: Qwen3-4B text-encoder forward on CPU.
Finds the fastest dtype / sequence-length / thread config for prompt re-encode.
No GPU, no compile — runs in ~1-2 min.
"""
import time
import torch
from transformers import Qwen3ForCausalLM

MODEL = "FLUX.2-klein-4B/text_encoder"
LAYERS = (9, 18, 27)


def run(dtype, seq, threads):
    torch.set_num_threads(threads)
    enc = Qwen3ForCausalLM.from_pretrained(MODEL, local_files_only=True)
    enc.eval().to("cpu", dtype)
    ids = torch.randint(0, 150000, (1, seq))
    mask = torch.ones((1, seq), dtype=torch.long)
    with torch.no_grad():
        # warmup
        _ = enc(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                use_cache=False)
        t0 = time.time()
        out = enc(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                  use_cache=False)
        _ = torch.stack([out.hidden_states[k] for k in LAYERS], dim=1)
        dt = time.time() - t0
    del enc
    return dt


def main():
    ncpu = torch.get_num_threads()
    import os
    cores = os.cpu_count()
    print(f"cpu cores={cores}, default torch threads={ncpu}", flush=True)
    for dtype in (torch.float32, torch.bfloat16):
        for seq in (512, 256, 128):
            dt = run(dtype, seq, cores)
            print(f"dtype={str(dtype).split('.')[-1]:8s} seq={seq:4d} "
                  f"threads={cores}  forward={dt*1000:7.0f}ms", flush=True)


if __name__ == "__main__":
    main()
