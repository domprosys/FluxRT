"""StreamDiffusionV2 backend worker: Wan2.1 causal-DMD video-to-video, 1.3B or 14B.

Targets upstream StreamDiffusionV2 @ 6961a5c (2026-09-14; pip-package layout with the
top-level packages `models`, `streamv2v`, `streamdiffusionv2`), installed into its own venv
by deploy/setup_sdv2_env.sh (python 3.10, torch 2.11 cu128 -> Ada sm_89 and Blackwell
sm_120). Spawned by fluxrt.backends.worker_client.WorkerBackend with cwd = the
StreamDiffusionV2 clone, under which the weights live:

    <clone>/.venv/bin/python sdv2_worker.py --in-shm ... --config cfg.json

Streaming model (mirrors streamv2v/inference.py SingleGPUInferencePipeline
.start_stream_session / .run_stream_batch, which demo/vid2vid.py drives):
  * The Wan VAE compresses 4 pixel frames -> 1 latent frame, so the model consumes
    "chunks" of 4 input frames (5 for the very first chunk after a reset).
  * Each chunk is one DiT call. The pipeline is a *stream batch*: the DiT batch holds one
    latent per denoising step (row 0 = newest chunk at the highest noise level, row -1 =
    the chunk from (steps-1) calls ago, now fully denoised). Every call finishes exactly
    one chunk; output latency is `steps` chunk periods.
  * Self-attention KV cache: ring buffer of num_kv_cache latent frames with sink frames
    (adaptively refreshed), carried across chunks, plus the causal VAE feature caches. Every
    t_refresh chunks the RoPE position is rewound; upstream re-rotates the cached keys
    (the 2026-09 "position refresh" fix; PyPI 0.1.1 lacks it -> ghosting after rewinds).

Mapping onto the latest-wins shm protocol (unchanged from the previous worker):
  * process(frame) only appends to a small ring buffer (2 chunks) and returns None, so
    stale frames are dropped automatically when the model is behind.
  * An inference thread takes 4 frames (evenly spaced over the buffer, newest included),
    runs one step and hands the 4 decoded frames to a publisher thread, which writes them
    with write_output() spaced by (measured chunk period / 4) or at worker.output_fps.

Differences from upstream's loaders (why this file does not just call the upstream API):
  * The generator is built from the base model's config.json on the meta device and filled
    straight from the DMD checkpoint, which holds the complete generator. Upstream calls
    diffusers from_pretrained() on the base Wan weights first (and, a quirk of
    CausalWanDiffusionWrapper, always also loads the full bidirectional Wan2.1-1.3B), only to
    overwrite them: the 14B path would need the 57 GB base shards for nothing. Needed here:
    config.json + UMT5 + tokenizer + VAE (all shared by 1.3B and 14B) + the DMD checkpoint.
  * UMT5 is never loaded by the pipeline (upstream: fp32 on CPU, 22 GB RAM). Prompts are
    encoded by our own bf16 UMT5 (device configurable) with an on-disk cache, in a
    background thread, so a new prompt never stalls the stream.
  * VAE: full Wan, lightx2v LightVAE or TAEHV decoder, mixable (worker.vae_type).
  * A hard reset also clears the per-layer ring-buffer eviction queues (upstream keeps them
    across sessions) and restarts the TAEHV decode context.
  * Without flash-attn (none is built for torch 2.11), cross-attention goes through plain
    SDPA instead of upstream's fallback that passes an all-true mask (which rules out the
    flash SDPA kernel). Self-attention uses upstream's SDPA KV-cache fallback as is.

stdout is reserved for JSON events (WorkerBase redirects prints to stderr).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shm_protocol import WorkerBase  # noqa: E402

import numpy as np  # noqa: E402

CHUNK = 4          # pixel frames per latent frame (Wan VAE temporal stride)
FIRST_CHUNK = 5    # first chunk after a reset: 1 + 4 frames -> 2 latent frames
NATIVE_FPS = 16    # Wan2.1 training / demo frame rate
TEXT_LEN = 512
# Canonical v2v schedule (configs/wan_causal_dmd_v2v*.yaml). Like upstream merge_cli_config,
# `steps` N uses its first N entries + the terminal 0: 1 -> [700], 2 -> [700, 500], ...
BASE_STEPS = (700, 500, 400, 200)
MODEL_PRESETS = {
    "1.3b": {"model_type": "T2V-1.3B", "base_dir": "wan_models/Wan2.1-T2V-1.3B",
             "checkpoint": "ckpts/wan_causal_dmd_v2v/model.pt", "steps": 2},
    "14b": {"model_type": "T2V-14B", "base_dir": "wan_models/Wan2.1-T2V-14B",
            "checkpoint": "ckpts/wan_causal_dmd_v2v_14b/model.pt", "steps": 1},
}
SHARED_DIR = "wan_models/Wan2.1-T2V-1.3B"  # UMT5 / tokenizer / VAE are byte-identical for 14B
# Inference-relevant keys of upstream configs/wan_causal_dmd_v2v.yaml (== ..._14b.yaml).
# The _fast.yaml variant uses num_kv_cache 5, num_sink_tokens 2, adapt_sink_threshold -1.
PIPE_DEFAULTS = {
    "model_name": "wan", "generator_name": "causal_wan", "num_frame_per_block": 1,
    "num_kv_cache": 6, "num_sink_tokens": 3, "adapt_sink_threshold": 0.2,
}
# vae_type -> (encoder, decoder). "wan" = full Wan2.1 VAE (reference quality, slow);
# "light" = lightx2v LightVAE (75% channel-pruned Wan VAE, same latent space, several x
# faster); "taehv" = madebyollin TAEHV decoder (upstream --use_taehv; fastest decode).
VAE_TYPES = {
    "wan": ("wan", "wan"),
    "lightvae": ("light", "light"),
    "lightvae_decode": ("wan", "light"),
    "taehv": ("wan", "taehv"),
    "lightvae_taehv": ("light", "taehv"),
}
# Wan2.1 VAE latent statistics (models/wan/wan_wrapper.py WanVAEWrapper)
VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
TAEHV_CONTEXT = 3  # latents of decode context kept between chunks (upstream TAEHVWanVAEWrapper)


def _err(msg: str) -> None:
    print(f"[sdv2] {msg}", file=sys.stderr, flush=True)


def _torch_load(torch, path: str, mmap: bool = True):
    """mmap'd weights-only load; falls back for legacy/non-zip or pickled-object files
    (all checkpoints used here come from pinned HF/GitHub revisions)."""
    try:
        return torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        _err(f"torch.load(mmap={mmap}, weights_only) failed for {path}: {str(exc)[:160]}; retrying plain")
        return torch.load(path, map_location="cpu", weights_only=False)


class SDV2Worker(WorkerBase):
    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        t_setup = time.perf_counter()
        w = dict(self.cfg.get("worker") or {})
        self.wcfg = w
        self.lib_root = os.path.abspath(w.get("lib_root") or os.getcwd())
        os.chdir(self.lib_root)
        self._start_parent_watchdog()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        size = str(w.get("model_size", "1.3b")).lower().replace("t2v-", "")
        if size not in MODEL_PRESETS:
            raise ValueError(f"worker.model_size {size!r} unknown (1.3b | 14b)")
        self.model_size, self.preset = size, MODEL_PRESETS[size]
        if self.height % 16 or self.width % 16:
            raise ValueError(f"resolution {self.width}x{self.height} must be multiples of 16 for Wan")

        import torch

        self.torch = torch
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = bool(w.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(w.get("device", "cuda"))
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[str(w.get("dtype", "bf16"))]
        self._check_cuda()
        self._import_upstream()

        # --- paths (relative to the clone = cwd); UMT5 / tokenizer / VAE fall back to the
        # 1.3B folder, which the 14B shares
        self.base_dir = w.get("base_dir") or self.preset["base_dir"]
        self.config_json = self._path("base_dir", os.path.join(self.base_dir, "config.json"))
        self.ckpt_path = self._path("checkpoint", w.get("checkpoint") or self.preset["checkpoint"])
        if os.path.isdir(self.ckpt_path):  # upstream-style checkpoint_folder
            self.ckpt_path = self._path("checkpoint", os.path.join(self.ckpt_path, "model.pt"))
        self.vae_type = str(w.get("vae_type", "lightvae"))
        if self.vae_type not in VAE_TYPES:
            raise ValueError(f"unknown vae_type {self.vae_type!r} ({' | '.join(VAE_TYPES)})")

        # --- live params
        self.steps = max(1, min(4, int(w.get("steps", self.preset["steps"]))))
        self.seed = int(w.get("seed", self.cfg.get("default_seed", 42)))
        self.strength = float(w.get("noise_scale", 0.8))
        self.adaptive_noise = bool(w.get("adaptive_noise", True))
        self.normalize_latents = bool(w.get("normalize_latents", False))
        self.prompt_reset = "soft" if str(w.get("prompt_reset", "hard")) == "soft" else "hard"
        self.output_fps = float(w.get("output_fps", 0) or 0)     # 0 = auto (chunk throughput)
        self.gap_reset_s = float(w.get("reset_after_gap_s", 2.0) or 0)
        self.te_device_cfg = str(w.get("text_encoder_device", "auto"))
        self.te_device = self.te_device_cfg
        self.prompt = self.cfg.get("default_prompt") or "a cinematic portrait"
        self.pcfg = self._make_pipe_config()
        self.t_refresh = int(w.get("t_refresh", 50))
        lo = 3 * int(self.pcfg.num_kv_cache) + 2  # upstream re-aligns cached keys only after a big rewind
        if not lo <= self.t_refresh <= 900:       # RoPE table has 1024 positions
            _err(f"t_refresh {self.t_refresh} out of range, clamped to [{lo}, 900]")
            self.t_refresh = max(lo, min(900, self.t_refresh))

        # --- text encoder: UMT5 skeleton now (main thread), weights paged in / moved in the
        # background; only needed for prompts missing from the on-disk embedding cache
        self.embed_cache_dir = os.path.join(self.lib_root, ".cache", "sdv2_prompt_embeds")
        os.makedirs(self.embed_cache_dir, exist_ok=True)
        self._embeds: dict[str, "torch.Tensor"] = {}
        self._te = None
        self._te_raw = None
        self._te_ready = threading.Event()
        self._te_error: str | None = None
        self._skeleton_done = threading.Event()  # accelerate's init_empty_weights is process-global
        self._gen_ready = threading.Event()
        if self.te_device_cfg == "none":
            self._te_error = "text_encoder_device is 'none' (embedding cache only)"
            self._te_ready.set()
        else:
            self._init_text_encoder()
            threading.Thread(target=self._load_text_encoder, daemon=True, name="t5-load").start()

        # --- diffusion pipeline + VAEs
        t0 = time.perf_counter()
        self.log(f"loading Wan2.1-{self.preset['model_type']} causal DMD v2v from {os.path.relpath(self.ckpt_path)} ...")
        self._build_pipeline()
        self._load_vaes()
        self.attn_backend = self._setup_attention()
        self._gen_ready.set()
        self._skeleton_done.set()
        self.log(f"pipeline loaded in {time.perf_counter() - t0:.1f}s, gpu {self.gpu_mb():.0f}MB, "
                 f"attention {self.attn_backend}, vae {self.vae_type}")

        # --- prompt embeddings for default + cycle prompts
        self._get_embeds(self.prompt, allow_encode=True)
        for p in self.cfg.get("prompt_cycle") or []:
            try:
                self._get_embeds(p, allow_encode=True)
            except Exception as exc:  # noqa: BLE001 (cache-only mode: skip, encode later)
                _err(f"cycle prompt not available yet ({exc})")
        self._cur_embeds = self._embeds[self.prompt]

        # --- streaming state + threads
        self._in_lock = threading.Lock()
        self._in_cv = threading.Condition(self._in_lock)
        self._in_buf: collections.deque = collections.deque(maxlen=2 * CHUNK)
        self._last_in_t = 0.0
        self._out_lock = threading.Lock()
        self._out_cv = threading.Condition(self._out_lock)
        self._out_q: collections.deque = collections.deque()
        self._frame_interval = 1.0 / NATIVE_FPS
        self._ctl_lock = threading.Lock()
        self._pending: dict = {}
        self._need_reset = True
        self._paused = False
        self._halt = threading.Event()
        self._enc_q: list[str] = []
        self._enc_cv = threading.Condition()
        self.stats = {"chunks": 0, "chunk_ms": 0.0, "latency_ms": 0.0, "dropped_in": 0, "out_frames": 0, "resets": 0}

        # --- warmup (cudnn autotune, SDPA kernels, allocator) on synthetic frames
        t0 = time.perf_counter()
        self._warmup()
        self.log(f"warmup done in {time.perf_counter() - t0:.1f}s, chunk {self.stats['chunk_ms']:.0f}ms, "
                 f"gpu {self.gpu_mb():.0f}MB")

        threading.Thread(target=self._infer_loop, daemon=True, name="sdv2-infer").start()
        threading.Thread(target=self._publish_loop, daemon=True, name="sdv2-publish").start()
        threading.Thread(target=self._encode_loop, daemon=True, name="sdv2-encode").start()

        self.emit(
            "info",
            model=f"StreamDiffusionV2 (Wan2.1-{self.preset['model_type']} causal DMD v2v)",
            model_size=self.model_size,
            resolution=[self.width, self.height],
            chunk_size=CHUNK,
            first_chunk=FIRST_CHUNK,
            native_fps=NATIVE_FPS,
            steps=self.steps,
            denoising_steps=self._step_list(self.steps)[:-1],
            latency_chunks=self.steps,
            warm_chunk_ms=round(self.stats["chunk_ms"], 1),
            text_encoder_device=self.te_device,
            prompt_reset=self.prompt_reset,
            vae_type=self.vae_type,
            attention=self.attn_backend,
            kv_cache=[int(self.pcfg.num_kv_cache), int(self.pcfg.num_sink_tokens), float(self.pcfg.adapt_sink_threshold)],
            rope_realign=self._has_realign,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
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

    def _check_cuda(self) -> None:
        torch = self.torch
        if self.device.type != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available (torch {torch.__version__}, CUDA {torch.version.cuda})")
        major, minor = torch.cuda.get_device_capability(self.device)
        archs = torch.cuda.get_arch_list()
        name = torch.cuda.get_device_name(self.device)
        parsed = [(m.group(1), int(m.group(2))) for m in (re.fullmatch(r"(sm|compute)_(\d+)[a-z]?", a) for a in archs) if m]
        # a cubin for sm_XY runs on sm_XZ (Z >= Y), PTX compute_XY JIT-compiles for newer GPUs
        sass = [c for k, c in parsed if k == "sm" and c // 10 == major and c % 10 <= minor]
        ptx = [c for k, c in parsed if k == "compute" and c <= major * 10 + minor]
        if not sass and not ptx:
            raise RuntimeError(
                f"torch {torch.__version__} (CUDA {torch.version.cuda}) has no kernels for {name} "
                f"(sm_{major}{minor}); arch list {archs}. Rebuild the venv: bash deploy/setup_sdv2_env.sh")
        if not sass:
            _err(f"WARNING: no sm_{major}{minor} kernels in torch ({archs}); relying on PTX JIT")
        _err(f"{name} sm_{major}{minor}, torch {torch.__version__} CUDA {torch.version.cuda}")

    def _import_upstream(self) -> None:
        """Import the installed upstream package; fall back to the clone (source checkout)."""
        try:
            import models.wan.causal_stream_inference  # noqa: F401
        except ImportError:
            if self.lib_root not in sys.path:
                sys.path.insert(1, self.lib_root)
            import models.wan.causal_stream_inference  # noqa: F401
        import models.wan.causal_model as cmod

        self._has_realign = hasattr(cmod, "KV_POS_EMPTY")
        if not self._has_realign:
            _err("WARNING: this StreamDiffusionV2 build predates the KV RoPE re-alignment fix "
                 "(upstream 2026-09); expect brief ghosting every t_refresh chunks. "
                 "Install upstream @ 6961a5c (deploy/setup_sdv2_env.sh).")

    def _path(self, key: str, default: str, fallbacks: tuple = ()) -> str:
        explicit = self.wcfg.get(key) if key != "base_dir" else None
        cands = [c for c in (explicit, default, *fallbacks) if c]
        for c in cands:
            if os.path.exists(c):
                if explicit and c != explicit:
                    _err(f"worker.{key} {explicit} not found, using {c}")
                return os.path.abspath(c)
        raise FileNotFoundError(f"worker.{key}: none of {cands} exists (cwd {self.lib_root}); "
                                f"run deploy/setup_sdv2_env.sh (SDV2_14B=1 for the 14B)")

    def _shared_path(self, key: str, rel: str) -> str:
        return self._path(key, os.path.join(self.base_dir, rel), (os.path.join(SHARED_DIR, rel),))

    def _step_list(self, steps: int) -> list[int]:
        return [int(s) for s in self._base_steps[: max(1, min(4, steps))]] + [0]

    def _make_pipe_config(self):
        """Namespace with the attributes CausalStreamInferencePipeline reads. Defaults are
        upstream's v2v YAML; worker.config_path may point at a YAML to take them from."""
        w, cfg = self.wcfg, dict(PIPE_DEFAULTS)
        base_steps = list(BASE_STEPS)
        cp = w.get("config_path")
        if cp and os.path.exists(cp):
            from omegaconf import OmegaConf

            y = OmegaConf.to_container(OmegaConf.load(cp), resolve=True)
            cfg.update({k: y[k] for k in PIPE_DEFAULTS if k in y})
            if y.get("denoising_step_list"):
                base_steps = [int(s) for s in y["denoising_step_list"] if int(s) != 0] or base_steps
        elif cp:
            _err(f"config_path {cp} not found, using built-in upstream defaults")
        for k in ("num_kv_cache", "num_sink_tokens", "adapt_sink_threshold"):
            if k in w:
                cfg[k] = type(PIPE_DEFAULTS[k])(w[k])
        self._base_steps = base_steps
        cfg.update(
            model_type=self.preset["model_type"], height=self.height, width=self.width,
            num_frame_per_block=1, t2v=False, warp_denoising_step=False,
            use_taehv=False, use_tensorrt=False,  # VAEs are handled by this worker
            denoising_step_list=self._step_list(self.steps),
        )
        return types.SimpleNamespace(**cfg)

    # ------------------------------------------------------------ model loading
    def _load_generator_state_dict(self) -> dict:
        """Generator weights from the DMD checkpoint, keys relative to CausalWanModel
        (same key search as upstream streamv2v.inference_common.load_generator_state_dict)."""
        torch = self.torch
        ck = _torch_load(torch, self.ckpt_path)
        sd = None
        if isinstance(ck, dict):
            for key in ("generator", "generator_ema", "state_dict", "model"):
                if isinstance(ck.get(key), dict):
                    sd = ck[key]
                    break
        sd = ck if sd is None else sd
        out = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            for junk in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.", "module."):
                k = k.replace(junk, "")
            out[k[6:] if k.startswith("model.") else k] = v
        return out

    def _load_generator_model(self):
        from accelerate import init_empty_weights
        from models.wan.causal_model import CausalWanModel

        with open(self.config_json) as f:
            mcfg = json.load(f)
        kwargs = {k: v for k, v in mcfg.items() if not k.startswith("_")}
        with init_empty_weights():  # parameters on meta; RoPE freqs (plain tensor) stay on CPU
            model = CausalWanModel(**kwargs)
        self._skeleton_done.set()
        t0 = time.perf_counter()
        sd = self._load_generator_state_dict()
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)  # size mismatch raises
        if missing:
            raise RuntimeError(
                f"{os.path.basename(self.ckpt_path)} lacks {len(missing)} generator weights for "
                f"{os.path.relpath(self.config_json)} (e.g. {missing[:4]}); model_size / checkpoint mismatch?")
        if unexpected:
            _err(f"ignoring {len(unexpected)} unexpected checkpoint keys (e.g. {unexpected[:3]})")
        del sd
        model = model.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        n = sum(p.numel() for p in model.parameters())
        _err(f"generator {n / 1e9:.2f}B params on {self.device} in {time.perf_counter() - t0:.1f}s")
        if self.wcfg.get("fp8", False):  # torchao fp8 dynamic quant (as Daydream Scope does)
            try:  # needs torchao (SDV2_TORCHAO=1 in deploy/setup_sdv2_env.sh)
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_
            except ImportError:
                from torchao.quantization.quant_api import (
                    Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_)

            quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
            _err("generator quantized to fp8 (torchao)")
        return model

    def _build_pipeline(self) -> None:
        """Upstream CausalStreamInferencePipeline with our generator / text / VAE stand-ins
        swapped into the model registry for the duration of its constructor."""
        torch = self.torch
        import models as registry
        from models.wan.causal_stream_inference import CausalStreamInferencePipeline
        from models.wan.flow_match import FlowMatchScheduler
        from models.wan.wan_wrapper import CausalWanDiffusionWrapper

        worker = self

        class _Generator(CausalWanDiffusionWrapper):
            def __init__(self, model_type=None):  # noqa: ARG002 - no from_pretrained loads
                torch.nn.Module.__init__(self)
                self.model = worker._load_generator_model()
                self.uniform_timestep = False
                self.scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
                self.scheduler.set_timesteps(1000, training=True)
                self.seq_len = 32760
                self.post_init()

        class _EmbedProvider(torch.nn.Module):
            """Stands in for the UMT5 wrapper: prepare() gets the current pre-computed
            prompt embedding instead of running the text encoder."""

            def __init__(self, model_type=None):  # noqa: ARG002
                super().__init__()

            def forward(self, text_prompts):  # noqa: ARG002
                return {"prompt_embeds": worker._cur_embeds.clone()}

        class _NoVAE(torch.nn.Module):
            def __init__(self, model_type=None):  # noqa: ARG002
                super().__init__()

        tables = (registry.DIFFUSION_NAME_TO_CLASS, registry.TEXT_ENCODER_NAME_TO_CLASS, registry.VAE_NAME_TO_CLASS)
        saved = [dict(t) for t in tables]
        registry.DIFFUSION_NAME_TO_CLASS[self.pcfg.generator_name] = _Generator
        registry.TEXT_ENCODER_NAME_TO_CLASS[self.pcfg.model_name] = _EmbedProvider
        registry.VAE_NAME_TO_CLASS[self.pcfg.model_name] = _NoVAE
        try:
            pipe = CausalStreamInferencePipeline(self.pcfg, device=self.device)
        finally:
            for t, s in zip(tables, saved, strict=True):
                t.clear()
                t.update(s)
        pipe.device = self.device
        self.pipe = pipe
        self.fsl = pipe.frame_seq_length

    def _load_vaes(self) -> None:
        torch = self.torch
        from models.wan.wan_base.modules.vae import _video_vae

        enc_kind, dec_kind = VAE_TYPES[self.vae_type]
        mods: dict = {}

        def get(kind):
            if kind not in mods:
                if kind == "wan":
                    m = _video_vae(pretrained_path=self._shared_path("vae_path", "Wan2.1_VAE.pth"), z_dim=16)
                else:
                    path = self._path("lightvae_path", "wan_models/Autoencoders/lightvaew2_1.pth")
                    m = _video_vae(pretrained_path=path, z_dim=16, dim=24)
                mods[kind] = m.eval().requires_grad_(False).to(self.device, self.dtype)
            return mods[kind]

        self._enc_model = get(enc_kind)
        self._dec_model = self._taehv = None
        if dec_kind == "taehv":
            from models.wan.taehv_wrapper import TAEHV

            path = self._path("taehv_path", "ckpts/taew2_1.pth", ("wan_models/Autoencoders/taew2_1.pth",))
            tdtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self._taehv = TAEHV(checkpoint_path=path).eval().requires_grad_(False).to(self.device, tdtype)
            self._taehv_dtype = tdtype
            self._taehv_parallel = bool(self.wcfg.get("taehv_parallel", True))
        else:
            self._dec_model = get(dec_kind)
        self._taehv_ctx = None
        self._taehv_first = True
        mean = torch.tensor(VAE_MEAN, device=self.device, dtype=self.dtype)
        self._vae_scale = [mean, 1.0 / torch.tensor(VAE_STD, device=self.device, dtype=self.dtype)]
        self._empty_cache()

    def _setup_attention(self) -> str:
        import models.wan.causal_model as cmod
        import models.wan.wan_base.modules.attention as am

        if getattr(cmod, "FLASH_ATTN_AVAILABLE", False):
            return "flash_attn"
        warnings.filterwarnings("ignore", message=r"flash_attn(_with_kvcache)? (is not installed|failed)")
        if not self.wcfg.get("fast_sdpa", True):
            return "sdpa (upstream fallback)"
        torch = self.torch
        F = torch.nn.functional
        orig = am._sdpa_attention_fallback

        def sdpa_fallback(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                          causal=False, window_size=(-1, -1), dtype=torch.bfloat16, attn_mask=None):
            # the cross-attention case: no lengths, no mask -> the all-true mask upstream would
            # build only blocks the flash SDPA kernel. Anything else: upstream's code path.
            if (q_lens is None and k_lens is None and attn_mask is None and softmax_scale is None
                    and q_scale is None and not causal and tuple(window_size) == (-1, -1)
                    and not dropout_p and q.device.type == "cuda"):
                out = F.scaled_dot_product_attention(q.transpose(1, 2).to(dtype), k.transpose(1, 2).to(dtype),
                                                     v.transpose(1, 2).to(dtype))
                return out.transpose(1, 2).contiguous().to(q.dtype)
            return orig(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
                        q_scale=q_scale, causal=causal, window_size=window_size, dtype=dtype, attn_mask=attn_mask)

        am._sdpa_attention_fallback = sdpa_fallback
        return "sdpa"

    # ------------------------------------------------------------ text encoder
    def _init_text_encoder(self) -> None:
        """UMT5-XXL built on the meta device with the mmap'd bf16 weights assigned (fast,
        no random init). Runs in the main thread before the generator skeleton."""
        torch = self.torch
        from models.wan.wan_base.modules.t5 import umt5_xxl
        from models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer

        t5_path = self._shared_path("t5_path", "models_t5_umt5-xxl-enc-bf16.pth")
        tok_path = self._shared_path("tokenizer_path", "google/umt5-xxl")
        self._te_bytes = os.path.getsize(t5_path)
        te = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device=torch.device("meta"))
        te.load_state_dict(_torch_load(torch, t5_path), assign=True)
        te = te.eval().requires_grad_(False)
        tok = HuggingfaceTokenizer(name=tok_path.rstrip("/") + "/", seq_len=TEXT_LEN, clean="whitespace")
        self._te_raw = (te, tok)

    def _resolve_te_device(self) -> str:
        """auto: GPU when it has room for UMT5 plus a margin once the generator is loaded
        (a 96 GB card always does; a 24 GB card with other engines resident may not)."""
        d = self.te_device_cfg
        if d != "auto":
            return d
        if self.device.type != "cuda":
            return "cpu"
        self._gen_ready.wait()
        free, _ = self.torch.cuda.mem_get_info(self.device)
        fp32 = 2 if self.wcfg.get("text_encoder_fp32", False) else 1
        need = self._te_bytes * fp32 + float(self.wcfg.get("text_encoder_auto_margin_gb", 8)) * 2**30
        return "cuda" if free >= need else "cpu"

    def _load_text_encoder(self) -> None:
        """Background: move/convert UMT5 as configured and page the weights in."""
        try:
            torch = self.torch
            self._skeleton_done.wait()
            dev = self._resolve_te_device()
            self.te_device = dev
            t0 = time.perf_counter()
            te, tok = self._te_raw
            if self.wcfg.get("text_encoder_fp32", False):
                te = te.float()
            if dev != "cpu":
                te = te.to(self.device if dev == "cuda" else dev)
            self._te = (te, tok)
            with torch.no_grad():  # page in the mmap'd weights / warm kernels
                ids, mask = tok(["warmup"], return_mask=True, add_special_tokens=True)
                n = int(mask.sum())
                d = next(te.parameters()).device
                te(ids[:, :n].to(d), mask[:, :n].to(d))
            _err(f"text encoder ready on {dev} in {time.perf_counter() - t0:.1f}s")
        except Exception:  # noqa: BLE001
            self._te = None
            self._te_error = traceback.format_exc()
            _err(f"text encoder failed to load: {self._te_error}")
        finally:
            self._te_raw = None
            self._te_ready.set()

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.embed_cache_dir, hashlib.sha1(text.encode()).hexdigest() + ".pt")

    def _encode_text(self, text: str):
        """UMT5-XXL on the configured device. Encodes only the real tokens (padding is masked
        out in T5 attention, so this is exact) and zero-pads to 512 like upstream."""
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
        """Return [1, 512, 4096] embeds on the model device, from memory / disk cache / UMT5."""
        torch = self.torch
        if text in self._embeds:
            return self._embeds[text]
        path = self._cache_path(text)
        trimmed = None
        if os.path.exists(path):
            try:
                trimmed = torch.load(path, map_location="cpu", weights_only=True)
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
    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _empty_cache(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _frames_to_tensor(self, frames: list[np.ndarray]):
        torch = self.torch
        arr = np.stack(frames, 0)  # T,H,W,3 BGR uint8
        t = torch.from_numpy(arr).to(self.device, non_blocking=True)
        t = t.flip(-1).permute(3, 0, 1, 2).unsqueeze(0)  # 1,3,T,H,W RGB
        return t.to(self.dtype).div_(127.5).sub_(1.0)

    def _decode(self, latents) -> np.ndarray:
        """[1, T, 16, h, w] denoised latents -> T (or 4T) BGR uint8 frames at the output size."""
        torch = self.torch
        if self._taehv is not None:
            video = self._taehv_decode(latents)  # T,3,H,W in [0,1]
        else:
            zs = latents.permute(0, 2, 1, 3, 4).to(self.dtype)
            video = self._dec_model.stream_decode(zs, self._vae_scale)  # 1,3,T,H,W in ~[-1,1]
            video = (video[0].permute(1, 0, 2, 3).float() * 0.5 + 0.5)
        oh, ow = self.out_height, self.out_width
        if (oh, ow) != (self.height, self.width):
            video = torch.nn.functional.interpolate(video.float(), size=(oh, ow), mode="bilinear",
                                                    align_corners=False,
                                                    antialias=oh < self.height or ow < self.width)
        video = (video.clamp_(0, 1) * 255.0).round_().to(torch.uint8)
        return video.flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # T,H,W,3 BGR

    def _taehv_decode(self, latents):
        """Streaming TAEHV decode as upstream TAEHVWanVAEWrapper.stream_decode_to_pixel: keep
        a short latent prefix as temporal context, emit only the new frames."""
        torch = self.torch
        lat = latents.to(self._taehv_dtype)
        if self._taehv_first:
            self._taehv_first = False
            dec, emit = lat, max(lat.shape[1] - 1, 0)
        else:
            ctx = self._taehv_ctx
            dec = lat if ctx is None else torch.cat([ctx, lat], dim=1)
            emit = lat.shape[1]
        self._taehv_ctx = dec[:, -min(TAEHV_CONTEXT, dec.shape[1]):].clone()
        video = self._taehv.decode_video(dec, parallel=self._taehv_parallel, show_progress_bar=False)
        video = video[0, -emit * CHUNK:] if emit else video[0, :0]  # T,3,H,W in [0,1]
        return video.float()

    def _noise_step(self, images):
        """Motion-adaptive noise (streamv2v.inference.compute_noise_scale_and_step), with the
        0.8 base replaced by the live `strength` param."""
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
        lat = self._enc_model.stream_encode(images, self._vae_scale if self.normalize_latents else None)
        lat = lat.transpose(2, 1).contiguous().to(self.dtype)
        noise = torch.randn_like(lat)
        return noise * ns + lat * (1 - ns)

    def _reset_stream(self, frames: list[np.ndarray]) -> np.ndarray:
        """Hard reset (as upstream start_stream_session on start / prompt change): clear KV,
        cross-attn and VAE caches, run the first 5-frame chunk through all steps."""
        torch, p = self.torch, self.pipe
        torch.manual_seed(self.seed)
        self._enc_model.first_encode = True
        if self._dec_model is not None:
            self._dec_model.first_decode = True
        self._taehv_first, self._taehv_ctx = True, None
        p.kv_cache1 = None
        p.crossattn_cache = None
        p.block_x = None
        p.hidden_states = None
        for blk in p.generator.model.blocks:  # ring-buffer eviction queues (upstream keeps them)
            blk.self_attn.evict_idx = None
        # upstream's pipe.timestep aliases denoising_step_list, so the adaptive current_step
        # overwrote entry 0: restore the canonical schedule for the first (all-steps) chunk
        p._init_denoising_step_list(self.pcfg, self.device)
        tm = [time.perf_counter()]
        self._empty_cache()  # batch size may have changed (steps); release old caches
        images = self._frames_to_tensor(frames)
        self._noise_scale = self.strength
        ns, _ = self._noise_step(images)
        noisy = self._encode_latents(images, ns)
        self._sync(); tm.append(time.perf_counter())
        self.cur_start, self.cur_end = 0, 2 * self.fsl
        den = p.prepare(text_prompts=[self.prompt], device=self.device, dtype=self.dtype,
                        noise=noisy, current_start=self.cur_start, current_end=self.cur_end)
        self._sync(); tm.append(time.perf_counter())
        out = self._decode(den)
        self._sync(); tm.append(time.perf_counter())
        ms = [round((b - a) * 1000) for a, b in zip(tm, tm[1:])]
        self.log(f"reset: encode {ms[0]} ms, prepare {ms[1]} ms, decode {ms[2]} ms")
        self.cur_start = self.cur_end
        self.cur_end += self.fsl
        self._last_image = images[:, :, [-1]]
        self._processed = 0
        self._chunk_ts = collections.deque(maxlen=len(p.denoising_step_list))
        self.stats["resets"] = self.stats.get("resets", 0) + 1
        return out

    def _stream_step(self, frames: list[np.ndarray]) -> np.ndarray | None:
        torch, p = self.torch, self.pipe
        if self.cur_start // self.fsl >= self.t_refresh:  # keep RoPE positions bounded (upstream)
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
            c["is_init"] = False  # recomputed from the new context on the next call

    def _set_steps(self, steps: int) -> None:
        self.steps = max(1, min(4, int(steps)))
        self.pcfg.denoising_step_list = self._step_list(self.steps)
        self.pipe._init_denoising_step_list(self.pcfg, self.device)
        self._need_reset = True

    def _warmup(self) -> None:
        rng = np.random.default_rng(0)
        base = (rng.random((self.height, self.width, 3)) * 255).astype(np.uint8)
        n = len(self.pipe.denoising_step_list) + 3
        frames = [np.roll(base, 8 * i, axis=1) for i in range(FIRST_CHUNK + CHUNK * n)]
        self._reset_stream(frames[:FIRST_CHUNK])
        for k in range(n):
            self._sync()
            t0 = time.perf_counter()
            self._stream_step(frames[FIRST_CHUNK + CHUNK * k: FIRST_CHUNK + CHUNK * (k + 1)])
            self._sync()
            self.stats["chunk_ms"] = (time.perf_counter() - t0) * 1000
        self.stats["resets"] = 0
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
            return bool(self._pending) and any(k in self._pending for k in ("prompt", "reset", "steps", "pause"))

    def _apply_pending(self) -> None:
        with self._ctl_lock:
            pend, keep = dict(self._pending), {}
            if "want_prompt" in pend and "prompt" not in pend:
                keep["want_prompt"] = pend["want_prompt"]
            self._pending = keep
        if "pause" in pend:
            self._paused = bool(pend["pause"])
            if not self._paused:
                self._need_reset = True  # stale context after a pause
            with self._in_cv:
                self._in_buf.clear()
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
                    if self._paused:
                        time.sleep(0.05)
                        continue
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
                    if last_done is not None and len(out):
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
                             f"resets {self.stats['resets']}  gpu {self.gpu_mb():.0f}MB")
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
        now = time.perf_counter()
        with self._in_cv:
            if self.gap_reset_s and self._last_in_t and now - self._last_in_t > self.gap_reset_s:
                # input stalled (engine idle in multi mode, client reconnect): start fresh
                self._in_buf.clear()
                with self._ctl_lock:
                    self._pending["reset"] = True
            self._last_in_t = now
            self._in_buf.append((now, frame_bgr))
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
            elif key == "normalize_latents":
                self.normalize_latents = bool(val)
                with self._ctl_lock:
                    self._pending["reset"] = True  # different latent statistics: start fresh
            elif key == "prompt_reset":
                self.prompt_reset = "soft" if str(val) == "soft" else "hard"
            elif key == "output_fps":
                self.output_fps = float(val or 0)
            elif key in ("reset", "restart"):
                with self._ctl_lock:
                    self._pending["reset"] = True
            elif key == "paused":
                with self._ctl_lock:
                    self._pending["pause"] = bool(val)
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
