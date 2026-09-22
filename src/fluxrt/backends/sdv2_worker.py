"""StreamDiffusionV2 backend worker (Wan2.1-1.3B causal DMD, video-to-video).

Runs in the StreamDiffusionV2 clone's own venv (python 3.10 / torch 2.6 / flash-attn),
spawned by fluxrt.backends.worker_client.WorkerBackend with cwd = the library clone:

    <clone>/.venv/bin/python sdv2_worker.py --in-shm ... --config cfg.json

Streaming model (mirrors demo/vid2vid.py + causvid CausalStreamInferencePipeline):
  * The Wan VAE compresses 4 pixel frames -> 1 latent frame, so the model consumes
    "chunks" of 4 input frames (5 for the very first chunk after a reset).
  * Each chunk is one DiT call. The pipeline is a *stream batch*: the DiT batch holds
    one latent per denoising step (row 0 = newest chunk at the highest noise level,
    row -1 = chunk from (steps-1) calls ago, now fully denoised). So every call
    finishes exactly one chunk; output latency is `steps` chunk periods.
  * Self-attention KV cache (sink + rolling window, num_kv_cache latent frames) and
    the causal VAE encoder/decoder feature caches are carried across chunks.

Mapping onto the latest-wins shm protocol:
  * process(frame) only appends the frame to a small ring buffer (2 chunks deep) and
    returns None -> old frames are dropped automatically when the model is behind.
  * A background inference thread takes 4 frames (evenly spaced over the ring
    buffer, newest included), runs one step, and hands the 4 decoded frames to a
    publisher thread.
  * The publisher writes them with write_output() spaced by (measured chunk period / 4),
    i.e. at the model's actual throughput (the model's native rate is 16 fps; with a
    25-30 fps camera and a fast GPU the throughput can be higher). A fixed rate can
    be forced with worker.output_fps.

stdout is reserved for JSON events (WorkerBase redirects prints to stderr).
"""

from __future__ import annotations

import collections
import hashlib
import math
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shm_protocol import WorkerBase  # noqa: E402

import numpy as np  # noqa: E402

STEP_LISTS = {  # same mapping as streamv2v/inference.py and demo/vid2vid.py
    1: [700, 0],
    2: [700, 500, 0],
    3: [700, 600, 400, 0],
    4: [700, 600, 500, 400, 0],
}
CHUNK = 4          # pixel frames per latent frame (Wan VAE temporal stride)
FIRST_CHUNK = 5    # first chunk after a reset: 1 + 4 frames -> 2 latent frames
NATIVE_FPS = 16    # Wan2.1 training / demo frame rate
TEXT_LEN = 512


def _err(msg: str) -> None:
    print(f"[sdv2] {msg}", file=sys.stderr, flush=True)


class SDV2Worker(WorkerBase):
    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        t_setup = time.perf_counter()
        w = dict(self.cfg.get("worker") or {})
        self.wcfg = w
        self.lib_root = os.path.abspath(w.get("lib_root") or os.getcwd())
        if self.lib_root not in sys.path:
            sys.path.insert(0, self.lib_root)
        os.chdir(self.lib_root)
        self._start_parent_watchdog()

        if self.height % 16 or self.width % 16:
            raise ValueError(f"resolution {self.width}x{self.height} must be multiples of 16 for Wan")

        import torch

        self.torch = torch
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = bool(w.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        # live params
        self.steps = int(w.get("steps", 2))
        self.seed = int(w.get("seed", self.cfg.get("default_seed", 42)))
        self.strength = float(w.get("noise_scale", 0.8))
        self.adaptive_noise = bool(w.get("adaptive_noise", True))
        self.prompt_reset = str(w.get("prompt_reset", "hard"))  # hard (demo) | soft (keep KV)
        self.output_fps = float(w.get("output_fps", 0) or 0)     # 0 = auto (chunk throughput)
        self.te_device = str(w.get("text_encoder_device", "cpu"))
        self.prompt = self.cfg.get("default_prompt") or "a cinematic portrait"

        # --- text encoder: load in the background (it is only needed for prompts
        # that are not in the on-disk embedding cache)
        self.embed_cache_dir = os.path.join(self.lib_root, ".cache", "sdv2_prompt_embeds")
        os.makedirs(self.embed_cache_dir, exist_ok=True)
        self._embeds: dict[str, "torch.Tensor"] = {}
        self._te = None
        self._te_ready = threading.Event()
        self._te_error: str | None = None
        self._init_text_encoder()
        threading.Thread(target=self._load_text_encoder, daemon=True, name="t5-load").start()

        # --- diffusion pipeline
        t0 = time.perf_counter()
        self.log("loading Wan2.1-1.3B causal DMD v2v pipeline ...")
        self._build_pipeline()
        self.log(f"pipeline loaded in {time.perf_counter() - t0:.1f}s, gpu {self.gpu_mb():.0f}MB")

        # --- prompt embeddings for default + cycle prompts
        wanted = [self.prompt] + list(self.cfg.get("prompt_cycle") or [])
        for p in wanted:
            self._get_embeds(p, allow_encode=True)
        self._cur_embeds = self._embeds[self.prompt]

        # --- streaming state + threads
        self._in_lock = threading.Lock()
        self._in_cv = threading.Condition(self._in_lock)
        self._in_buf: collections.deque = collections.deque(maxlen=2 * CHUNK)
        self._out_lock = threading.Lock()
        self._out_cv = threading.Condition(self._out_lock)
        self._out_q: collections.deque = collections.deque()
        self._frame_interval = 1.0 / NATIVE_FPS
        self._ctl_lock = threading.Lock()
        self._pending: dict = {}
        self._need_reset = True
        self._halt = threading.Event()
        self._enc_q: list[str] = []
        self._enc_cv = threading.Condition()
        self.stats = {"chunks": 0, "chunk_ms": 0.0, "latency_ms": 0.0, "dropped_in": 0, "out_frames": 0}

        # --- warmup (cudnn autotune, flash-attn, allocator) on synthetic frames
        t0 = time.perf_counter()
        self._warmup()
        self.log(f"warmup done in {time.perf_counter() - t0:.1f}s, chunk {self.stats['chunk_ms']:.0f}ms")

        threading.Thread(target=self._infer_loop, daemon=True, name="sdv2-infer").start()
        threading.Thread(target=self._publish_loop, daemon=True, name="sdv2-publish").start()
        threading.Thread(target=self._encode_loop, daemon=True, name="sdv2-encode").start()

        self.emit(
            "info",
            model="StreamDiffusionV2 (Wan2.1-T2V-1.3B causal DMD v2v)",
            resolution=[self.width, self.height],
            chunk_size=CHUNK,
            first_chunk=FIRST_CHUNK,
            native_fps=NATIVE_FPS,
            steps=self.steps,
            latency_chunks=self.steps,
            warm_chunk_ms=round(self.stats["chunk_ms"], 1),
            text_encoder_device=self.te_device,
            prompt_reset=self.prompt_reset,
            vae_type=self.vae_type,
            setup_s=round(time.perf_counter() - t_setup, 1),
        )

    def _start_parent_watchdog(self) -> None:
        ppid = os.getppid()

        def watch():
            while True:
                time.sleep(1.0)
                if os.getppid() != ppid:
                    _err("parent died, exiting")
                    os._exit(0)

        threading.Thread(target=watch, daemon=True, name="ppid-watch").start()

    def _make_config(self):
        from omegaconf import OmegaConf

        w = self.wcfg
        cfg = OmegaConf.load(w.get("config_path", "configs/wan_causal_dmd_v2v.yaml"))
        cfg.height, cfg.width = self.height, self.width
        cfg.denoising_step_list = STEP_LISTS[max(1, min(4, self.steps))]
        cfg.warp_denoising_step = False
        if "num_kv_cache" in w:
            cfg.num_kv_cache = int(w["num_kv_cache"])
        return cfg

    def _build_pipeline(self) -> None:
        torch = self.torch
        import causvid.models as cm
        import causvid.models.wan.wan_wrapper as ww
        from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline

        worker = self

        class _EmbedProvider(torch.nn.Module):
            """Stands in for the T5 wrapper inside the pipeline: returns the
            current (pre-computed) prompt embedding, so prepare() never runs T5."""

            def forward(self, text_prompts):
                return {"prompt_embeds": worker._cur_embeds.clone()}

        class _SkipWanModel:  # CausalWanDiffusionWrapper's parent would load a 2nd WanModel
            @staticmethod
            def from_pretrained(*a, **k):
                return torch.nn.Identity()

        self.pcfg = self._make_config()
        orig_te, orig_wm = cm.TEXTENCODER_NAME_TO_CLASS["wan"], ww.WanModel
        cm.TEXTENCODER_NAME_TO_CLASS["wan"] = _EmbedProvider
        ww.WanModel = _SkipWanModel
        try:
            pipe = CausalStreamInferencePipeline(self.pcfg, device="cpu")
        finally:
            cm.TEXTENCODER_NAME_TO_CLASS["wan"], ww.WanModel = orig_te, orig_wm
        pipe.to(device=self.device, dtype=self.dtype)
        pipe.device = self.device
        ckpt = self.wcfg.get("checkpoint", "ckpts/wan_causal_dmd_v2v/model.pt")
        sd = torch.load(ckpt, map_location="cpu", mmap=True)["generator"]
        pipe.generator.load_state_dict(sd, strict=True)
        del sd
        if self.wcfg.get("fp8", False):  # torchao fp8 dynamic quant (as Daydream Scope does)
            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_)

            quantize_(pipe.generator, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
        pipe._init_denoising_step_list(self.pcfg, self.device)
        self.pipe = pipe
        self.fsl = pipe.frame_seq_length

        # VAE selection. "wan" = full Wan2.1 VAE (reference quality, ~250ms/chunk at
        # 512^2); "lightvae" = lightx2v's 75%-channel-pruned Wan VAE (same latent space,
        # same streaming code, several x faster); "lightvae_decode" = full encoder
        # (faithful latents for the DiT) + light decoder.
        self.vae_type = str(self.wcfg.get("vae_type", "wan"))
        self._enc_model = self._dec_model = pipe.vae.model
        if self.vae_type in ("lightvae", "lightvae_decode"):
            from causvid.models.wan.wan_base.modules.vae import _video_vae

            path = self.wcfg.get("lightvae_path", "wan_models/Autoencoders/lightvaew2_1.pth")
            light = _video_vae(pretrained_path=path, z_dim=16, dim=24).eval().requires_grad_(False)
            light = light.to(self.device, self.dtype)
            self._dec_model = light
            if self.vae_type == "lightvae":
                self._enc_model = light
                pipe.vae.model = light  # drop the full VAE
        elif self.vae_type != "wan":
            raise ValueError(f"unknown vae_type {self.vae_type!r} (wan | lightvae | lightvae_decode)")
        self._vae_scale = [pipe.vae.mean.to(self.device, self.dtype), (1.0 / pipe.vae.std).to(self.device, self.dtype)]
        torch.cuda.empty_cache()

    # ------------------------------------------------------------ text encoder
    def _init_text_encoder(self) -> None:
        """Build UMT5-XXL on the meta device and assign the mmap'd bf16 weights (fast,
        no random init). Must run in the main thread *before* diffusers'
        from_pretrained, whose init_empty_weights patch is process-global."""
        torch = self.torch
        from causvid.models.wan.wan_base.modules.t5 import umt5_xxl
        from causvid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer

        base = os.path.join(self.lib_root, "wan_models/Wan2.1-T2V-1.3B")
        te = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device=torch.device("meta"))
        sd = torch.load(os.path.join(base, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", mmap=True)
        te.load_state_dict(sd, assign=True)
        te = te.eval().requires_grad_(False)
        tok = HuggingfaceTokenizer(name=os.path.join(base, "google/umt5-xxl/"), seq_len=TEXT_LEN, clean="whitespace")
        self._te_raw = (te, tok)

    def _load_text_encoder(self) -> None:
        """Background: move/convert T5 as configured and page the weights in."""
        try:
            torch = self.torch
            t0 = time.perf_counter()
            te, tok = self._te_raw
            if self.wcfg.get("text_encoder_fp32", False):
                te = te.float()
            if self.te_device == "cuda":
                te = te.to("cuda")
            self._te = (te, tok)
            with torch.no_grad():  # page in the mmap'd weights / warm kernels
                ids, mask = tok(["warmup"], return_mask=True, add_special_tokens=True)
                n = int(mask.sum())
                dev = next(te.parameters()).device
                te(ids[:, :n].to(dev), mask[:, :n].to(dev))
            _err(f"text encoder ready on {self.te_device} in {time.perf_counter() - t0:.1f}s")
        except Exception:  # noqa: BLE001
            self._te = None
            self._te_error = traceback.format_exc()
            _err(f"text encoder failed to load: {self._te_error}")
        finally:
            self._te_ready.set()

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.embed_cache_dir, hashlib.sha1(text.encode()).hexdigest() + ".pt")

    def _encode_text(self, text: str):
        """UMT5-XXL on the configured device. Encodes only the real tokens (padding is
        masked out in T5 attention, so this is exact) and zero-pads to 512."""
        torch = self.torch
        self._te_ready.wait()
        if self._te is None:
            raise RuntimeError(f"text encoder unavailable: {self._te_error}")
        te, tok = self._te
        ids, mask = tok([text], return_mask=True, add_special_tokens=True)
        n = max(1, int(mask.sum()))
        dev = next(te.parameters()).device
        t0 = time.perf_counter()
        with torch.no_grad():
            ctx = te(ids[:, :n].to(dev), mask[:, :n].to(dev))
        _err(f"encoded prompt ({n} tokens) on {dev} in {time.perf_counter() - t0:.2f}s")
        return ctx[0].to("cpu", torch.bfloat16).contiguous()  # [n, 4096]

    def _get_embeds(self, text: str, allow_encode: bool):
        """Return [1, 512, 4096] bf16 cuda embeds, from memory / disk cache / T5."""
        torch = self.torch
        if text in self._embeds:
            return self._embeds[text]
        path = self._cache_path(text)
        trimmed = None
        if os.path.exists(path):
            try:
                trimmed = torch.load(path, map_location="cpu")
            except Exception:  # noqa: BLE001
                trimmed = None
        if trimmed is None:
            if not allow_encode:
                return None
            trimmed = self._encode_text(text)
            try:
                torch.save(trimmed, path)
            except Exception:  # noqa: BLE001
                pass
        full = torch.zeros(1, TEXT_LEN, trimmed.shape[-1], dtype=self.dtype, device=self.device)
        full[0, : trimmed.shape[0]] = trimmed.to(self.device, self.dtype)
        self._embeds[text] = full
        return full

    def _encode_loop(self) -> None:
        """Background prompt encoding (latest-wins) so the stream keeps running."""
        while not self._halt.is_set():
            with self._enc_cv:
                while not self._enc_q and not self._halt.is_set():
                    self._enc_cv.wait(0.5)
                if self._halt.is_set():
                    return
                text = self._enc_q[-1]
                self._enc_q.clear()
            try:
                self._get_embeds(text, allow_encode=True)
                with self._ctl_lock:
                    if self._pending.get("want_prompt") == text:
                        self._pending["prompt"] = text
            except Exception:  # noqa: BLE001
                self.emit("error", msg=f"prompt encode failed: {traceback.format_exc()}")

    # ---------------------------------------------------------- core stepping
    def _frames_to_tensor(self, frames: list[np.ndarray]):
        torch = self.torch
        arr = np.stack(frames, 0)  # T,H,W,3 BGR uint8
        t = torch.from_numpy(arr).to(self.device, non_blocking=True)
        t = t.flip(-1).permute(3, 0, 1, 2).unsqueeze(0)  # 1,3,T,H,W RGB
        return t.to(self.dtype).div_(127.5).sub_(1.0)

    def _decode(self, latents) -> np.ndarray:
        torch = self.torch
        zs = latents.permute(0, 2, 1, 3, 4).to(self.dtype)
        video = self._dec_model.stream_decode(zs, self._vae_scale)  # 1,3,T,H,W in ~[-1,1]
        video = video.permute(0, 2, 1, 3, 4).float()
        video = ((video[0] * 0.5 + 0.5).clamp_(0, 1) * 255.0).round_().to(torch.uint8)
        video = video.flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # T,H,W,3 BGR
        if (self.out_height, self.out_width) != (self.height, self.width):
            import cv2

            video = np.stack([cv2.resize(f, (self.out_width, self.out_height), interpolation=cv2.INTER_LINEAR)
                              for f in video])
        return video

    def _noise_step(self, images):
        """Motion-adaptive noise (streamv2v.inference.compute_noise_scale_and_step),
        with the 0.8 base replaced by the live `strength` param."""
        if self.adaptive_noise:
            d = (images[:, :, -CHUNK:] - images[:, :, -CHUNK - 1:-1]) ** 2
            d = (self.torch.sqrt(d.float().mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1).item()
            ns = (self.strength - 0.1 * d) * 0.9 + self._noise_scale * 0.1
        else:
            ns = self.strength
        ns = float(min(0.99, max(0.05, ns)))
        self._noise_scale = ns
        step = int(1000 * ns) - 100
        lst = self.pipe.denoising_step_list
        floor = int(lst[1].item()) + 20 if len(lst) > 1 else 50
        return ns, max(step, floor)

    def _encode_latents(self, images, ns):
        torch = self.torch
        lat = self._enc_model.stream_encode(images)
        lat = lat.transpose(2, 1).contiguous().to(self.dtype)
        noise = torch.randn_like(lat)
        return noise * ns + lat * (1 - ns)

    def _reset_stream(self, frames: list[np.ndarray]) -> np.ndarray:
        """Hard reset (as demo/vid2vid.py does on start / prompt change): clear KV,
        cross-attn and VAE caches, run the first 5-frame chunk through all steps."""
        torch, p = self.torch, self.pipe
        torch.manual_seed(self.seed)
        self._enc_model.first_encode = True
        self._dec_model.first_decode = True
        p.kv_cache1 = None
        p.crossattn_cache = None
        p.block_x = None
        p.hidden_states = None
        torch.cuda.empty_cache()  # batch size may have changed (steps); release old caches
        images = self._frames_to_tensor(frames)
        self._noise_scale = self.strength
        ns, _ = self._noise_step(images)
        noisy = self._encode_latents(images, ns)
        self.cur_start, self.cur_end = 0, 2 * self.fsl
        den = p.prepare(text_prompts=[self.prompt], device=self.device, dtype=self.dtype,
                        noise=noisy, current_start=self.cur_start, current_end=self.cur_end)
        out = self._decode(den)
        self.cur_start = self.cur_end
        self.cur_end += self.fsl
        self._last_image = images[:, :, [-1]]
        self._processed = 0
        self._chunk_ts = collections.deque(maxlen=len(p.denoising_step_list))
        return out

    def _stream_step(self, frames: list[np.ndarray]) -> np.ndarray | None:
        torch, p = self.torch, self.pipe
        if self.cur_start // self.fsl >= 50:  # keep RoPE positions bounded (demo logic)
            self.cur_start = p.kv_cache_length - self.fsl
            self.cur_end = self.cur_start + self.fsl
        images = self._frames_to_tensor(frames)
        ns, step = self._noise_step(torch.cat([self._last_image, images], dim=2))
        noisy = self._encode_latents(images, ns)
        den = p.inference_stream(noise=noisy, current_start=self.cur_start,
                                 current_end=self.cur_end, current_step=step)
        self._processed += 1
        self.cur_start = self.cur_end
        self.cur_end += self.fsl
        self._last_image = images[:, :, [-1]]
        if self._processed >= len(p.denoising_step_list):
            return self._decode(den[[-1]])
        return None

    def _soft_prompt_switch(self) -> None:
        """Swap the text conditioning but keep KV / VAE caches (Scope-style)."""
        p = self.pipe
        if p.crossattn_cache is None or p.conditional_dict is None:
            self._need_reset = True
            return
        b = len(p.denoising_step_list)
        p.conditional_dict["prompt_embeds"] = self._cur_embeds.repeat(b, 1, 1)
        for c in p.crossattn_cache:
            c["is_init"] = False

    def _set_steps(self, steps: int) -> None:
        self.steps = max(1, min(4, int(steps)))
        self.pcfg.denoising_step_list = STEP_LISTS[self.steps]
        self.pipe._init_denoising_step_list(self.pcfg, self.device)
        self._need_reset = True

    def _warmup(self) -> None:
        rng = np.random.default_rng(0)
        base = (rng.random((self.height, self.width, 3)) * 255).astype(np.uint8)
        frames = [np.roll(base, 8 * i, axis=1) for i in range(64)]
        self._reset_stream(frames[:FIRST_CHUNK])
        n = len(self.pipe.denoising_step_list) + 3
        for k in range(n):
            self.torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._stream_step(frames[FIRST_CHUNK + 4 * k: FIRST_CHUNK + 4 * k + 4])
            self.torch.cuda.synchronize()
            self.stats["chunk_ms"] = (time.perf_counter() - t0) * 1000
        self._need_reset = True

    # ------------------------------------------------------------ the threads
    def _take_frames(self, n: int, latest: bool):
        """Wait for >= n buffered frames; return n of them (newest included)."""
        with self._in_cv:
            while len(self._in_buf) < n:
                if self._halt.is_set() or self._pending_reset_request():
                    return None
                self._in_cv.wait(0.05)
            buf = list(self._in_buf)
            self._in_buf.clear()
        if latest or len(buf) == n:
            sel = buf[-n:]
        else:  # evenly spaced over the buffer, like demo util.select_images
            idx = np.linspace(0, len(buf) - 1, n).round().astype(int)
            sel = [buf[i] for i in idx]
            self.stats["dropped_in"] += len(buf) - n
        return [f for _, f in sel], sel[-1][0]

    def _pending_reset_request(self) -> bool:
        with self._ctl_lock:
            return bool(self._pending) and ("prompt" in self._pending or "reset" in self._pending
                                            or "steps" in self._pending)

    def _apply_pending(self) -> None:
        with self._ctl_lock:
            pend, keep = dict(self._pending), {}
            if "want_prompt" in pend and "prompt" not in pend:
                keep["want_prompt"] = pend["want_prompt"]
            self._pending = keep
        if "steps" in pend:
            self._set_steps(pend["steps"])
        if "reset" in pend:
            self._need_reset = True
        if "prompt" in pend:
            text = pend["prompt"]
            self.prompt = text
            self._cur_embeds = self._embeds[text]
            if self.prompt_reset == "soft" and not self._need_reset:
                self._soft_prompt_switch()
            else:
                self._need_reset = True
            self.log(f"prompt -> {text[:80]!r} ({'soft' if not self._need_reset else 'hard reset'})")

    def _infer_loop(self) -> None:
        torch = self.torch
        with torch.no_grad():
            last_done = None
            last_report = time.perf_counter()
            while not self._halt.is_set():
                try:
                    self._apply_pending()
                    need = FIRST_CHUNK if self._need_reset else CHUNK
                    got = self._take_frames(need, latest=self._need_reset)
                    if got is None:
                        continue
                    frames, cap_t = got
                    t0 = time.perf_counter()
                    if self._need_reset:
                        out = self._reset_stream(frames)
                        self._need_reset = False
                        last_done = None
                        with self._out_lock:
                            self._out_q.clear()
                        self._chunk_ts.append(cap_t)
                    else:
                        self._chunk_ts.append(cap_t)
                        out = self._stream_step(frames)
                    t1 = time.perf_counter()
                    dt = t1 - t0
                    self.stats["chunks"] += 1
                    self.stats["chunk_ms"] = 0.9 * self.stats["chunk_ms"] + 0.1 * dt * 1000
                    if out is None:
                        continue
                    lat_ts = self._chunk_ts[0]
                    self.stats["latency_ms"] = (t1 - lat_ts) * 1000
                    if last_done is not None:
                        period = min(1.0, t1 - last_done)
                        iv = period / len(out)
                        self._frame_interval = 0.8 * self._frame_interval + 0.2 * iv
                    last_done = t1
                    with self._out_cv:
                        for i, f in enumerate(out):
                            self._out_q.append((f, dt if i == 0 else None))
                        while len(self._out_q) > 3 * CHUNK:  # latest-wins on the output side too
                            self._out_q.popleft()
                        self._out_cv.notify()
                    if t1 - last_report > 10:
                        last_report = t1
                        _err(f"chunk {self.stats['chunk_ms']:.0f}ms  e2e latency {self.stats['latency_ms']:.0f}ms  "
                             f"out {1.0 / max(1e-3, self._frame_interval):.1f}fps  dropped_in {self.stats['dropped_in']}  "
                             f"gpu {self.gpu_mb():.0f}MB")
                except Exception:  # noqa: BLE001
                    self.emit("error", msg=f"inference step failed: {traceback.format_exc()}")
                    self._need_reset = True
                    time.sleep(0.2)

    def _publish_loop(self) -> None:
        next_t = time.perf_counter()
        while not self._halt.is_set():
            with self._out_cv:
                while not self._out_q and not self._halt.is_set():
                    self._out_cv.wait(0.1)
                if self._halt.is_set():
                    return
                frame, proc = self._out_q.popleft()
                backlog = len(self._out_q)
            iv = (1.0 / self.output_fps) if self.output_fps > 0 else self._frame_interval
            if backlog > CHUNK + 1:
                iv *= 0.75  # drain a growing backlog a bit faster
            now = time.perf_counter()
            if next_t < now - iv:
                next_t = now
            if next_t > now:
                time.sleep(next_t - now)
            self.write_output(frame, proc_time_s=proc)
            self.stats["out_frames"] += 1
            next_t += iv

    # ------------------------------------------------------- protocol hooks
    def process(self, frame_bgr: np.ndarray):
        with self._in_cv:
            self._in_buf.append((time.perf_counter(), frame_bgr))
            self._in_cv.notify()
        return None

    def on_command(self, name: str, payload: dict) -> None:
        if name == "set_prompt":
            text = str(payload.get("text") or "").strip()
            if not text or text == self.prompt:
                return
            if self._get_embeds(text, allow_encode=False) is not None:
                with self._ctl_lock:
                    self._pending["want_prompt"] = text
                    self._pending["prompt"] = text
            else:
                with self._ctl_lock:
                    self._pending["want_prompt"] = text
                    self._pending.pop("prompt", None)
                with self._enc_cv:
                    self._enc_q.append(text)
                    self._enc_cv.notify()
                self.log("encoding new prompt in background (stream continues with the old one)")
            return
        if name == "set_param":
            key, val = payload.get("name"), payload.get("value")
            if key in ("steps", "num_steps", "denoising_steps"):
                with self._ctl_lock:
                    self._pending["steps"] = int(val)
            elif key == "seed":
                self.seed = int(val)
                with self._ctl_lock:
                    self._pending["reset"] = True
            elif key in ("strength", "noise_scale"):
                self.strength = float(val)
            elif key == "adaptive_noise":
                self.adaptive_noise = bool(val)
            elif key == "prompt_reset":
                self.prompt_reset = "soft" if str(val) == "soft" else "hard"
            elif key == "output_fps":
                self.output_fps = float(val or 0)
            elif key in ("reset", "restart"):
                with self._ctl_lock:
                    self._pending["reset"] = True
            else:
                self.log(f"unknown param {key!r} ignored")
                return
            self.log(f"param {key} = {val}")
            return
        self.log(f"unknown command {name!r} ignored")


def main() -> int:
    w = SDV2Worker()
    try:
        rc = w.run()
    finally:
        halt = getattr(w, "_halt", None)
        if halt is not None:
            halt.set()
        _err("exiting")
        sys.stderr.flush()
    # daemon threads may hold CUDA work; exit hard so the process never lingers
    os._exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
