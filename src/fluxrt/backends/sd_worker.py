"""Classic StreamDiffusion worker (SD-Turbo / SD1.5+LCM-LoRA / SDXL-Turbo img2img).

Runs in the StreamDiffusion venv (daydreamlive fork), NOT the FluxRT venv:
    <sd-venv>/bin/python sd_worker.py --in-shm ... --config cfg.json
Only imports shm_protocol (by path) from this repo; never the fluxrt package.
Importing this module is CPU-only; everything that touches CUDA runs in setup().

Config (the "worker" dict of configs/sd*_config.json), all optional:
    model_id            "stabilityai/sd-turbo" | "Lykon/dreamshaper-8" | "stabilityai/sdxl-turbo" | local path
    sdxl                force SDXL handling on/off (default: "xl" in model_id, the fork's own heuristic)
    variant             "fp16": load the fp16 weight files of the pipeline and the ControlNets (the fork
                        loads the fp32 files: 2x download/RAM, SDXL-Turbo 13.9 GB instead of 6.9 GB).
                        Falls back to the default files for repos without that variant.
    t_index_list        [35, 45]   (indices into num_inference_steps; LCM schedule: t ~ 999 - 20*index at 50)
    num_inference_steps 50
    acceleration        "none" (PyTorch SDPA, default) | "tensorrt" | "xformers" (broken in the fork ->
                        "none" unless force_xformers). Env SD_ACCELERATION overrides it for any SD/SDXL
                        config (pod: ENABLE_TRT=1). "tensorrt" without a working TensorRT install falls
                        back to "none" with a log line (install: deploy/setup_trt.sh).
    engine_dir          TensorRT engine root. Default (also for the legacy value "engines"): $WS/engines
                        (WS default /workspace, if that exists) so engines persist on a network volume,
                        else ./engines. Env SD_ENGINE_DIR overrides. Engines are GPU-arch and TensorRT-
                        version specific, so they go into a subdir "sm<cc>-trt<version>"
                        (engine_dir_per_gpu: false to disable).
    max_batch_size      4: TensorRT UNet/ControlNet engines accept batch 1..max (batch = len(t_index_list),
                        2x for cfg_type "full", +1 for "initialize")
    trt_controlnet_engines "enabled" (default: engines only for the nets enabled at start; the others
                        run in PyTorch if switched on live) | "all" (build every listed net: ~2-8 min each)
    sdxl_trt_added_cond true: SDXL engines get real pooled-text/size inputs (see _patch_sdxl_trt_added_cond)
    use_tiny_vae        true (TAESD / TAESDXL); vae_id overrides the tiny VAE repo
    cfg_type            "none" | "self" | "full" | "initialize"
    guidance_scale      1.0 (turbo) / 1.2+ with cfg_type "self"
    delta, seed, negative_prompt, use_denoising_batch, similar_image_filter(_threshold), warmup_frames
    lora_dict           {"latent-consistency/lcm-lora-sdv1-5": 1.0}  (SD1.5 models)
    use_lcm_lora        true -> adds the right LCM LoRA automatically (ignored for turbo models)
    controlnets         list of {name, model_id, preprocessor, conditioning_scale, enabled,
                        preprocessor_params?, conditioning_channels?, async?}. Must match the base model
                        (SD1.5: lllyasviel/control_v11*; SDXL: diffusers/controlnet-*-sdxl-1.0).
                        All listed nets are loaded at start; enable/scale switch live. "enabled": false
                        costs VRAM but no time (the preprocessor and the net are skipped).
    use_controlnet      default true when "controlnets" is non-empty
    use_cached_attn     false. StreamV2V cached attention: every UNet self-attention also attends to
                        keys/values cached from earlier frames (less flicker, some ghosting). Needs
                        cfg_type "none"/"self". Works with ControlNet (only the UNet caches) and with
                        TensorRT (the cache is an engine input).
    cache_maxframes     1: cached frames (live, within [min_cache_maxframes, max_cache_maxframes])
    cache_interval      1: refresh the cache every N frames (live)
    min_cache_maxframes 1 / max_cache_maxframes max(4, cache_maxframes): TensorRT bakes this range into
                        the UNet engine (outside it: rebuild); in PyTorch mode it is a memory guard.
                        Memory: K/V per cached frame ~46 MB x batch (SD1.5/SD2.1 at 512x512, fp16),
                        ~105 MB x batch (SDXL at 512x512); self-attention cost grows ~(1+maxframes)x in
                        the high-resolution layers. 16 frames on SD1.5 2-step: ~1.5 GB, noticeably slower.
    use_ipadapter       false
    ipadapter           {type: "faceid" | "regular" | "plus",
                         ipadapter_model_path ("h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"),
                         image_encoder_path ("h94/IP-Adapter/models/image_encoder"),
                         insightface_model_name ("buffalo_l"), scale (0.8), num_image_tokens (4),
                         reference_image (optional start identity, file path), insightface_bgr (true)}
                        FaceID needs deploy/setup_faceid.sh (pod: derived from use_ipadapter). Without
                        its packages the adapter is skipped with a log line (ipadapter_required: true
                        makes that fatal).

Live commands (set_param), besides set_prompt:
    guidance_scale, delta, seed, negative_prompt, t_index_list, num_inference_steps, strength
    ControlNet:
    {"name": "controlnet_scale",   "value": {"name": "depth", "scale": 0.6}}
    {"name": "controlnet_scales",  "value": {"depth": 0.6, "canny": 0.2}}
    {"name": "controlnet_enabled", "value": {"name": "openpose", "enabled": true}}
    {"name": "controlnet_params",  "value": {"name": "canny", "params": {"low_threshold": 50}}}
    {"name": "controlnet_scale:depth", "value": 0.6}     (flat form, handy for sliders)
    {"name": "controlnet_enabled:depth", "value": false}
    Cached attention:
    {"name": "cache_maxframes", "value": 4}   {"name": "cache_interval", "value": 2}
    {"name": "use_cached_attn", "value": false}   (PyTorch mode only; TensorRT bakes it into the engine)
    IP-Adapter / FaceID:
    {"name": "faceid_capture", "value": true}     identity from the NEXT input frame (the visitor's face)
    {"name": "ipadapter_image", "value": "/path/face.jpg"}
    {"name": "faceid_clear", "value": true}       drop the identity (zero image tokens)
    {"name": "ipadapter_scale", "value": 0.8}     {"name": "ipadapter_enabled", "value": false}
    "No face found" keeps the previous identity; results are reported as info events (ipadapter.faceid).

TensorRT limits: t_index_list length (engine batch) is fixed; resolution is fixed (never changes live);
cache_maxframes stays within the engine's [min, max]; use_cached_attn cannot be toggled.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shm_protocol import WorkerBase  # noqa: E402

import threading  # noqa: E402

import numpy as np  # noqa: E402

# Preprocessors too slow to run inline every frame: they run on a background thread
# on the newest frame and the last result is reused meanwhile (override per net with "async").
ASYNC_PREPROCESSORS = {"openpose", "mediapipe_pose", "hed", "lineart", "standard_lineart", "mediapipe_segmentation"}

# Modules the fork's TensorRT path imports (diffusers_ipadapter: its export-wrapper package imports
# the IP-Adapter wrapper unconditionally, so TensorRT needs it even without IP-Adapter).
TRT_MODULES = ("tensorrt", "polygraphy", "cuda.cudart", "onnx", "onnx_graphsurgeon", "diffusers_ipadapter")


def _truthy(v) -> bool:
    return v.strip().lower() in ("1", "true", "on", "yes") if isinstance(v, str) else bool(v)


def _missing_modules(names) -> list[str]:
    out = []
    for n in names:
        try:
            importlib.import_module(n)
        except Exception as e:  # noqa: BLE001  (ImportError, but also e.g. protobuf/onnx mismatches)
            out.append(f"{n} ({type(e).__name__}: {str(e)[:80]})")
    return out


def resolve_acceleration(w: dict, env=None, log=print) -> tuple[str, str | None]:
    """-> (acceleration, note). Env SD_ACCELERATION beats the config; unusable TensorRT -> "none"."""
    env = os.environ if env is None else env
    src = "env SD_ACCELERATION" if env.get("SD_ACCELERATION") else "config"
    accel = str(env.get("SD_ACCELERATION") or w.get("acceleration") or "none").strip().lower()
    if accel == "trt":
        accel = "tensorrt"
    note = None
    if accel == "xformers" and not w.get("force_xformers", False):
        # The fork's patched diffusers expects attention processors to return
        # (hidden_states, kvo_cache); diffusers' XFormersAttnProcessor returns only
        # hidden_states, which is then unpacked along the BATCH dim: silently wrong
        # with 2 denoising steps, a crash with 1 or 3+. PyTorch SDPA ("none") uses the
        # fork's AttnProcessor2_0 and is just as fast on torch 2.7.
        log("acceleration 'xformers' is broken in this fork; using 'none' (PyTorch SDPA)")
        accel = "none"
    elif accel == "tensorrt":
        missing = _missing_modules(TRT_MODULES)
        if missing:
            note = (f"tensorrt requested ({src}) but not usable: {'; '.join(missing)}. "
                    "Install with deploy/setup_trt.sh (pod: ENABLE_TRT=1). Falling back to PyTorch.")
            log(note)
            accel = "none"
    elif accel not in ("none", "xformers"):
        note = f"unknown acceleration {accel!r} ({src}); using 'none'"
        log(note)
        accel = "none"
    return accel, note


def resolve_engine_dir(w: dict, env=None, gpu_tag: str | None = None) -> str:
    """TensorRT engine dir: $SD_ENGINE_DIR | worker.engine_dir | $WS/engines | ./engines, + gpu subdir."""
    env = os.environ if env is None else env
    root = env.get("SD_ENGINE_DIR") or w.get("engine_dir") or ""
    if not root or root == "engines":  # "engines" was the old relative default: prefer the volume
        ws = env.get("WS") or ("/workspace" if os.path.isdir("/workspace") else "")
        root = os.path.join(ws, "engines") if ws else "engines"
    root = os.path.expanduser(str(root))
    if gpu_tag and w.get("engine_dir_per_gpu", True):
        root = os.path.join(root, gpu_tag)
    return os.path.abspath(root)


def _resolve_hf_dir(spec: str) -> str:
    """"org/repo/sub/dir" -> local dir with only config + safetensors (the fork's resolver downloads the
    whole subdir, e.g. h94/IP-Adapter/models/image_encoder = 2.5 GB safetensors + 2.5 GB .bin)."""
    if os.path.exists(spec):
        return spec
    parts = spec.split("/")
    if len(parts) < 3 or "." in parts[-1]:
        return spec  # a repo id or a file: leave it to the fork
    from huggingface_hub import snapshot_download

    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    root = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/*.json", f"{sub}/*.safetensors"])
    path = os.path.join(root, sub)
    if not any(f.endswith(".safetensors") for f in os.listdir(path)):
        root = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/*"])
    return path


def _engine_lock(engine_dir: str, log):
    """Serialise TensorRT engine builds/loads between workers sharing a GPU and an engine dir (a build
    takes all free VRAM as workspace, and two builders of the same engine would corrupt it)."""
    try:
        import fcntl

        f = open(os.path.join(engine_dir, ".build.lock"), "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"waiting for another worker using {engine_dir} ...")
            fcntl.flock(f, fcntl.LOCK_EX)
        return f
    except Exception as e:  # noqa: BLE001  (e.g. a network fs without flock)
        log(f"engine lock unavailable ({e}); continuing without")
        return None


# ── fork workarounds applied before the wrapper is built (process-wide monkeypatches) ────────────
def _patch_variant_loading(variant: str, log) -> None:
    """Make the fork's plain from_pretrained() calls (pipeline + ControlNets) load `variant` files."""
    import torch
    from diffusers import AutoPipelineForText2Image
    from diffusers.models import ControlNetModel

    def wrap(cls, extra: dict) -> None:
        if getattr(cls, "_fluxrt_variant", None) == variant:
            return
        func = cls.from_pretrained.__func__

        def from_pretrained(klass, path, *args, **kw):
            if "variant" not in kw and not os.path.isfile(str(path)):
                try:
                    return func(klass, path, *args, **{**extra, **kw, "variant": variant})
                except (OSError, ValueError) as e:
                    log(f"{klass.__name__}({path}): no {variant} weights ({type(e).__name__}); using the default files")
            return func(klass, path, *args, **kw)

        cls.from_pretrained = classmethod(from_pretrained)
        cls._fluxrt_variant = variant

    wrap(AutoPipelineForText2Image, {"torch_dtype": torch.float16})
    wrap(ControlNetModel, {})


def _patch_unet_engine_path(suffix: str) -> None:
    """Append `suffix` to the UNet engine dir name. The fork's name has no resolution / cache range /
    input-set, but cached-attention engines are resolution-specific and our SDXL engines have extra
    inputs, so those must not collide with (or be mistaken for) plain engines."""
    from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType

    if not hasattr(EngineManager, "_fluxrt_get_engine_path"):
        EngineManager._fluxrt_get_engine_path = EngineManager.get_engine_path
    orig = EngineManager._fluxrt_get_engine_path

    def get_engine_path(self, engine_type, *args, **kw):
        path = orig(self, engine_type, *args, **kw)
        if suffix and engine_type == EngineType.UNET:
            path = path.parent.with_name(path.parent.name + suffix) / path.name
        return path

    EngineManager.get_engine_path = get_engine_path


def _patch_controlnet_engines(allowed: set | None, log) -> None:
    """Build ControlNet engines only for `allowed` model ids (None = all) and load each engine once
    (the fork loads a second copy when two nets share a model, e.g. tile + color)."""
    from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager

    if not hasattr(EngineManager, "_fluxrt_get_cn"):
        EngineManager._fluxrt_get_cn = EngineManager.get_or_load_controlnet_engine
    orig = EngineManager._fluxrt_get_cn
    loaded: dict = {}

    def get_or_load_controlnet_engine(self, model_id, *args, **kw):
        if allowed is not None and model_id not in allowed:
            log(f"TensorRT: no engine for {model_id} (disabled at start; runs in PyTorch if enabled)")
            raise RuntimeError("skipped: ControlNet disabled at start")  # the wrapper logs and moves on
        key = (str(self.engine_dir), model_id)
        if key not in loaded:
            eng = orig(self, model_id, *args, **kw)
            if eng is None:
                return None
            loaded[key] = eng
        return loaded[key]

    EngineManager.get_or_load_controlnet_engine = get_or_load_controlnet_engine


SDXL_ADDED_COND = (("text_embeds", 1280), ("time_ids", 6))


def _patch_sdxl_trt_added_cond() -> None:
    """SDXL + TensorRT: give the UNet engine real `text_embeds` / `time_ids` inputs.

    Fork bug: the UNet ONNX export never passes SDXL's added conditioning, so SDXLExportWrapper
    auto-generates zeros for it and those zeros are baked into the engine; at runtime the real
    tensors are dropped (Engine.infer filters unknown inputs). SDXL(-Turbo) under TensorRT then
    ignores the pooled prompt embedding and sees "original/target size 0x0" micro-conditioning.
    Fix: two extra engine inputs appended after all existing ones (control, kvo cache), an export
    wrapper that routes them into added_cond_kwargs, and a runtime shim that feeds them. Only UNet
    models with embedding_dim 2048 (SDXL) are affected. Engines go to a "--sdxlcond" UNet dir."""
    import torch
    from streamdiffusion.acceleration.tensorrt.export_wrappers import unet_sdxl_export
    from streamdiffusion.acceleration.tensorrt.models import models as trt_models
    from streamdiffusion.acceleration.tensorrt.runtime_engines import unet_engine

    U = trt_models.UNet
    if getattr(U, "_fluxrt_sdxl_cond", False):
        return
    names0, axes0, prof0, sample0 = U.get_input_names, U.get_dynamic_axes, U.get_input_profile, U.get_sample_input

    def xl(m) -> bool:
        return getattr(m, "embedding_dim", None) == 2048

    def get_input_names(self):
        names = names0(self)
        return names + [n for n, _ in SDXL_ADDED_COND] if xl(self) else names

    def get_dynamic_axes(self):
        axes = axes0(self)
        if xl(self):
            axes.update({n: {0: "2B"} for n, _ in SDXL_ADDED_COND})
        return axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        prof = prof0(self, batch_size, image_height, image_width, static_batch, static_shape)
        if xl(self):
            (bmin, *_), (bopt, *_), (bmax, *_) = prof["sample"]
            for n, d in SDXL_ADDED_COND:
                prof[n] = [(bmin, d), (bopt, d), (bmax, d)]
        return prof

    def get_sample_input(self, batch_size, image_height, image_width):
        inputs = tuple(sample0(self, batch_size, image_height, image_width))
        if not xl(self):
            return inputs
        n, dt = inputs[0].shape[0], (torch.float16 if self.fp16 else torch.float32)
        te = torch.zeros(n, 1280, dtype=dt, device=self.device)
        ti = torch.tensor([[image_height, image_width, 0, 0, image_height, image_width]] * n, dtype=dt, device=self.device)
        return inputs + (te, ti)

    class SDXLAddedCondExportWrapper(torch.nn.Module):
        """Replaces the fork's SDXLExportWrapper: the last two positional inputs are the added cond."""

        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, *args):
            *rest, text_embeds, time_ids = args
            return self.unet(*rest, added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids})

    U.get_input_names, U.get_dynamic_axes = get_input_names, get_dynamic_axes
    U.get_input_profile, U.get_sample_input = get_input_profile, get_sample_input
    unet_sdxl_export.SDXLExportWrapper = SDXLAddedCondExportWrapper  # export_onnx imports it at call time

    E = unet_engine.UNet2DConditionModelEngine
    call0 = E.__call__

    def wrap_io(eng) -> None:
        names = {eng.engine.get_tensor_name(i) for i in range(eng.engine.num_io_tensors)}
        eng._fluxrt_accepts = all(n in names for n, _ in SDXL_ADDED_COND)  # False: an old engine
        eng._fluxrt_extra = None
        if not eng._fluxrt_accepts:
            return
        alloc0, infer0 = eng.allocate_buffers, eng.infer

        def allocate_buffers(shape_dict=None, device="cuda"):
            ex = eng._fluxrt_extra
            if ex and shape_dict is not None:
                shape_dict = {**shape_dict, **{k: tuple(v.shape) for k, v in ex.items()}}
            return alloc0(shape_dict=shape_dict, device=device)

        def infer(feed_dict, stream, use_cuda_graph=False):
            ex = eng._fluxrt_extra
            return infer0({**feed_dict, **ex} if ex else feed_dict, stream, use_cuda_graph=use_cuda_graph)

        eng.allocate_buffers, eng.infer = allocate_buffers, infer

    def __call__(self, latent_model_input, timestep, encoder_hidden_states, *args, **kwargs):
        extra = {n: kwargs.pop(n) for n, _ in SDXL_ADDED_COND if n in kwargs}
        eng = self.engine
        if not hasattr(eng, "_fluxrt_accepts"):
            wrap_io(eng)
        if extra and eng._fluxrt_accepts:
            eng._fluxrt_extra = extra
        try:
            return call0(self, latent_model_input, timestep, encoder_hidden_states, *args, **kwargs)
        finally:
            eng._fluxrt_extra = None

    E.__call__ = __call__
    U._fluxrt_sdxl_cond = True


def _trt_torch_helpers():
    """-> (models.utils, models.attention_processors) of the fork's TensorRT package.

    Both are plain torch/diffusers code, but importing them runs streamdiffusion.acceleration.tensorrt
    /__init__, which imports tensorrt/polygraphy/onnx: absent (or, for onnx, broken by the protobuf
    pin) in the base SD venv. The fork itself imports models.utils for use_cached_attn even without
    TensorRT, so cached attention in PyTorch mode would die at import. Fallback: register the two
    parent packages as bare namespace modules (no __init__ run) and import the files directly."""
    try:
        from streamdiffusion.acceleration.tensorrt.models import attention_processors, utils
        return utils, attention_processors
    except Exception:  # noqa: BLE001
        pass
    import types

    import streamdiffusion.acceleration as acc

    base = Path(acc.__file__).resolve().parent / "tensorrt"
    for name, path in (("streamdiffusion.acceleration.tensorrt", base),
                       ("streamdiffusion.acceleration.tensorrt.models", base / "models")):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod.__package__ = name
        sys.modules[name] = mod
    from streamdiffusion.acceleration.tensorrt.models import attention_processors, utils

    return utils, attention_processors


def _set_self_attn_processors(unet, cached: bool, only_incompatible: bool = False) -> int:
    """Install the fork-compatible processor on every self-attention (attn1) layer.

    The fork's patched diffusers calls self-attention processors with kvo_cache=... and unpacks
    (hidden_states, kvo_cache) from them. IP-Adapter (Diffusers_IPAdapter) installs its own
    AttnProcessor2_0 there, which takes no kvo_cache and returns one tensor -> TypeError in PyTorch
    mode (its TensorRT export swaps them back itself). cached=True installs the StreamV2V processor."""
    import inspect

    from diffusers.models.attention_processor import AttnProcessor2_0

    CachedSTAttnProcessor2_0 = _trt_torch_helpers()[1].CachedSTAttnProcessor2_0
    procs = dict(unet.attn_processors)
    n = 0
    for name, p in procs.items():
        if not name.endswith("attn1.processor"):
            continue
        if only_incompatible and "kvo_cache" in inspect.signature(p.__call__).parameters:
            continue
        procs[name] = CachedSTAttnProcessor2_0() if cached else AttnProcessor2_0()
        n += 1
    if n:
        unet.set_attn_processor(procs)
    return n


class KVOCacheShim:
    """StreamV2V cached attention for the PyTorch UNet (the fork wires it for TensorRT only).

    In PyTorch mode the fork passes its FLAT K/V cache list to a UNet that indexes it per block, and
    keeps the default self-attention processors (which ignore it): use_cached_attn crashes on SD1.5
    and does nothing on SDXL (whose PyTorch path passes no cache). This installs the fork's
    CachedSTAttnProcessor2_0 on the self-attention layers and converts flat <-> nested around
    UNet.forward; for SDXL it also passes the cache and updates it. `enabled` flips live."""

    def __init__(self, stream, height: int, width: int):
        u = _trt_torch_helpers()[0]
        self.stream = stream
        unet = stream.unet
        _, self.structure, count = u.get_kvo_cache_info(unet, height, width)
        if count != len(stream.kvo_cache):
            raise RuntimeError(f"kvo cache has {len(stream.kvo_cache)} tensors, UNet has {count} self-attentions")
        _set_self_attn_processors(unet, cached=True)
        self.enabled = True
        nest, flat, orig = u.convert_list_to_structure, u.convert_structure_to_list, unet.forward
        missing = object()

        def forward(*args, **kw):
            kvo = kw.pop("kvo_cache", missing)
            if kvo is missing:  # SDXL path: the pipeline passes no cache and uses out[0]
                if not (self.enabled and stream.kvo_cache):
                    return orig(*args, **kw)
                out = orig(*args, kvo_cache=nest(stream.kvo_cache, self.structure), **kw)
                stream.update_kvo_cache(flat(out[1]))
                return (out[0],)
            if self.enabled and kvo:  # SD1.5/2.1 path: pipeline expects (pred, flat cache out)
                out = orig(*args, kvo_cache=nest(kvo, self.structure), **kw)
                return out[0], flat(out[1])
            return orig(*args, **kw)[0], []

        unet.forward = forward

    def reset(self) -> None:
        for t in self.stream.kvo_cache:
            t.zero_()
        self.stream.frame_idx = 0


class FastDepth:
    """Depth-Anything straight from a CUDA tensor in fp16 (~6 ms on a 4090).

    Replaces the fork's "depth" preprocessor, which runs a transformers pipeline via PIL
    in fp32 (~18 ms with Depth-Anything-V2-Small, much more with the default dpt-large)."""

    def __init__(self, model_name: str, detect_resolution: int, height: int, width: int):
        import torch
        from transformers import AutoModelForDepthEstimation

        self.torch = torch
        self.params = {"model_name": model_name, "detect_resolution": detect_resolution}
        self.height, self.width = height, width
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, dtype=torch.float16).to("cuda").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)

    def process_tensor(self, x):
        torch, F = self.torch, self.torch.nn.functional
        with torch.inference_mode():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            r = max(14, int(round(int(self.params.get("detect_resolution", 378)) / 14.0)) * 14)
            h, w = x.shape[-2:]
            rh, rw = (r, max(14, round(r * w / h / 14) * 14)) if h <= w else (max(14, round(r * h / w / 14) * 14), r)
            y = F.interpolate(x.to(torch.float16), size=(rh, rw), mode="bilinear", align_corners=False)
            d = self.model(pixel_values=(y - self.mean) / self.std).predicted_depth[:, None]
            d = F.interpolate(d, size=(self.height, self.width), mode="bilinear", align_corners=False)
            lo, hi = d.amin(), d.amax()
            return ((d - lo) / (hi - lo + 1e-6)).repeat(1, 3, 1, 1)


class SDWorker(WorkerBase):
    # Class defaults keep on_command / *_state usable on a partially initialised worker (and in tests).
    cn = None
    ipa = None
    kvo_shim = None
    cached = None
    is_sdxl = False
    acceleration = "none"
    accel_note = None
    engine_dir = None
    ip_disabled_reason = None
    _faceid_capture = False

    def setup(self) -> None:
        import torch
        from streamdiffusion import StreamDiffusionWrapper

        self.torch = torch
        w = dict(self.cfg.get("worker") or {})
        self.wcfg = w
        self.model_id = w.get("model_id", "stabilityai/sd-turbo")
        is_turbo = "turbo" in self.model_id.lower()
        self.is_sdxl = bool(w.get("sdxl", "xl" in self.model_id.lower()))
        self.t_index_list = list(w.get("t_index_list", [35, 45]))
        self.acceleration, self.accel_note = resolve_acceleration(w, os.environ, self.log)
        self.cfg_type = w.get("cfg_type", "none" if is_turbo else "self")
        self.guidance_scale = float(w.get("guidance_scale", 1.0 if is_turbo else 1.2))
        self.delta = float(w.get("delta", 1.0))
        self.seed = int(w.get("seed", 2))
        self.negative_prompt = w.get("negative_prompt", "")
        self.prompt = self.cfg.get("default_prompt") or w.get("prompt", "")
        self.max_batch_size = int(w.get("max_batch_size", 4))

        lora_dict = w.get("lora_dict")
        use_lcm_lora = w.get("use_lcm_lora")
        if use_lcm_lora is None and not is_turbo and not lora_dict:
            use_lcm_lora = True  # SD1.5 checkpoints need the LCM LoRA for few-step sampling

        # ControlNets. Names are our handles for live commands.
        cn_list = [dict(c) for c in (w.get("controlnets") or [])]
        self.use_controlnet = bool(w.get("use_controlnet", bool(cn_list))) and bool(cn_list)
        self.cn_names: list[str] = []
        self.cn_meta: list[dict] = []
        cn_config = []
        if self.use_controlnet:
            for i, c in enumerate(cn_list):
                name = str(c.get("name") or c.get("type") or f"cn{i}")
                self.cn_names.append(name)
                self.cn_meta.append({"name": name, "model_id": c["model_id"], "preprocessor": c.get("preprocessor")})
                cn_config.append({
                    "model_id": c["model_id"],
                    "preprocessor": c.get("preprocessor"),
                    "conditioning_scale": float(c.get("conditioning_scale", 0.5)),
                    "enabled": bool(c.get("enabled", True)),
                    "preprocessor_params": c.get("preprocessor_params"),
                    "conditioning_channels": c.get("conditioning_channels"),
                })

        self.cached = self._cached_attn_config(w)
        if self.cached and self.acceleration != "tensorrt":
            _trt_torch_helpers()  # lets the wrapper's create_kvo_cache import work without TensorRT installed
        ip_cfg = self._ipadapter_config(w)
        if w.get("variant"):
            _patch_variant_loading(str(w["variant"]), self.log)

        engine_dir, lock = "engines", None
        if self.acceleration == "tensorrt":
            import tensorrt

            cc = torch.cuda.get_device_capability()
            engine_dir = resolve_engine_dir(w, os.environ, f"sm{cc[0]}{cc[1]}-trt{tensorrt.__version__}")
            os.makedirs(engine_dir, exist_ok=True)
            suffix = ""
            if self.cached:  # the K/V cache inputs have the build resolution baked in
                suffix += f"--kvo{self.width}x{self.height}-f{self.cached['min']}-{self.cached['max']}"
            if self.is_sdxl and w.get("sdxl_trt_added_cond", True):
                _patch_sdxl_trt_added_cond()
                suffix += "--sdxlcond"
            _patch_unet_engine_path(suffix)
            if self.use_controlnet:
                mode = str(w.get("trt_controlnet_engines", "enabled")).lower()
                _patch_controlnet_engines(None if mode == "all" else {c["model_id"] for c in cn_config if c["enabled"]}, self.log)
            n_eng = sum(1 for _ in Path(engine_dir).rglob("*.engine"))
            self.log(f"TensorRT {tensorrt.__version__}: engines in {engine_dir} ({n_eng} present). Missing ones are "
                     "built now, once: SD1.5/SD2.1 UNet ~3-6 min, SDXL UNet ~10-20 min, ControlNet ~2-4 min "
                     "(SDXL ~5-8), VAE ~1 min. Later starts load them in seconds.")
            lock = _engine_lock(engine_dir, self.log)
        self.engine_dir = engine_dir

        self.log(
            f"building StreamDiffusionWrapper model={self.model_id} sdxl={self.is_sdxl} t={self.t_index_list} "
            f"{self.width}x{self.height} accel={self.acceleration} cfg={self.cfg_type} "
            f"cached_attn={bool(self.cached)} ipadapter={ip_cfg['type'] if ip_cfg else None}"
        )
        t0 = time.time()
        try:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=self.model_id,
                t_index_list=self.t_index_list,
                max_batch_size=self.max_batch_size,
                lora_dict=lora_dict,
                use_lcm_lora=use_lcm_lora if not is_turbo else None,
                mode="img2img",
                output_type="pt",
                vae_id=w.get("vae_id"),
                device="cuda",
                dtype=torch.float16,
                frame_buffer_size=1,
                width=self.width,
                height=self.height,
                acceleration=self.acceleration,
                use_tiny_vae=bool(w.get("use_tiny_vae", True)),
                use_denoising_batch=bool(w.get("use_denoising_batch", True)),
                cfg_type=self.cfg_type,
                seed=self.seed,
                enable_similar_image_filter=bool(w.get("similar_image_filter", False)),
                similar_image_filter_threshold=float(w.get("similar_image_filter_threshold", 0.98)),
                engine_dir=engine_dir,
                use_safety_checker=False,
                use_controlnet=self.use_controlnet,
                controlnet_config=cn_config or None,
                use_ipadapter=bool(ip_cfg),
                ipadapter_config=[ip_cfg] if ip_cfg else None,
                use_cached_attn=bool(self.cached),
                cache_maxframes=self.cached["maxframes"] if self.cached else 1,
                cache_interval=self.cached["interval"] if self.cached else 1,
                min_cache_maxframes=self.cached["min"] if self.cached else 1,
                max_cache_maxframes=self.cached["max"] if self.cached else 4,
            )
        finally:
            if lock is not None:
                lock.close()  # releases the flock
        s = self.stream.stream
        self.is_sdxl = bool(getattr(s, "is_sdxl", self.is_sdxl))
        if self.acceleration != "tensorrt":
            if self.cached:
                self.kvo_shim = KVOCacheShim(s, self.height, self.width)  # also covers IP-Adapter's attn1
            elif ip_cfg:
                n = _set_self_attn_processors(s.unet, cached=False, only_incompatible=True)
                self.log(f"IP-Adapter: {n} self-attention processors made kvo_cache-compatible")
        self.cn = getattr(s, "_controlnet_module", None) if self.use_controlnet else None
        if self.use_controlnet and self.cn is None:
            raise RuntimeError("ControlNet requested but the wrapper did not install a ControlNet module")
        if self.cn is not None:
            self._setup_controlnets(cn_list)
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=int(w.get("num_inference_steps", 50)),
            guidance_scale=self.guidance_scale,
            delta=self.delta,
        )
        if ip_cfg:
            self._setup_ipadapter(ip_cfg)
        load_s = time.time() - t0

        # Warm-up on a grey frame (also measures steady-state fps).
        dummy = np.full((self.height, self.width, 3), 127, np.uint8)
        n = int(w.get("warmup_frames", 10))
        for _ in range(3):
            self.process(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(n):
            self.process(dummy)
        torch.cuda.synchronize()
        fps = n / max(time.perf_counter() - t1, 1e-6)
        if getattr(s, "kvo_cache", None):  # don't start the stream remembering grey frames
            for t in s.kvo_cache:
                t.zero_()
            s.frame_idx = 0
        self.emit(
            "info",
            model=self.model_id,
            sdxl=self.is_sdxl,
            acceleration=self.acceleration,
            unet_runtime="tensorrt" if hasattr(s.unet, "engine") else "pytorch",  # the fork falls back on OOM
            **({"acceleration_note": self.accel_note} if self.accel_note else {}),
            **({"engine_dir": self.engine_dir} if self.acceleration == "tensorrt" else {}),
            t_index_list=self.t_index_list,
            cfg_type=self.cfg_type,
            warmup_fps=round(fps, 1),
            load_s=round(load_s, 1),
            worker_res=f"{self.width}x{self.height}",
            controlnets=self.cn_state(),
            cached_attn=self.cache_state(),
            ipadapter=self.ip_state(),
        )
        self.log(f"ready: load {load_s:.1f}s, warm-up {fps:.1f} fps, {self.gpu_mb():.0f} MB reserved")

    # ---------------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        if self._faceid_capture:  # between frames, on the main thread (no race with the UNet call)
            self._faceid_capture = False
            try:
                self._ip_apply(frame_bgr, "capture")
            except Exception as e:  # noqa: BLE001
                self.emit("error", msg=f"faceid capture failed: {e}")
        with torch.inference_mode():
            x = torch.from_numpy(frame_bgr).to("cuda", non_blocking=True)
            x = x.flip(-1).permute(2, 0, 1).unsqueeze(0).to(torch.float16).div_(255.0)  # 1x3xHxW RGB [0,1]
            if self.cn is not None:
                self._update_controls(x)  # the camera frame is also the control image
            y = self.stream(image=x)  # 1x3xHxW (or 3xHxW) RGB [0,1]
            if y.dim() == 4:
                y = y[0]
            if y.shape[1] != self.out_height or y.shape[2] != self.out_width:
                y = torch.nn.functional.interpolate(
                    y.unsqueeze(0).float(), size=(self.out_height, self.out_width),
                    mode="bilinear", align_corners=False,
                )[0]
            out = (y.clamp(0, 1) * 255.0).round().to(torch.uint8).flip(0).permute(1, 2, 0).contiguous()
            return out.cpu().numpy()

    # ── cached attention ──────────────────────────────────────────────────────
    def _cached_attn_config(self, w: dict) -> dict | None:
        if not _truthy(w.get("use_cached_attn", False)):
            return None
        if self.cfg_type in ("full", "initialize"):
            # the fork sizes the cache by the denoising batch; these cfg types feed the UNet more rows
            self.log(f"use_cached_attn needs cfg_type 'none' or 'self' (got {self.cfg_type!r}); cached attention off")
            return None
        mf = int(w.get("cache_maxframes", 1))
        lo = int(w.get("min_cache_maxframes", 1))
        hi = int(w.get("max_cache_maxframes", max(4, mf)))
        if not 1 <= lo <= mf <= hi:
            raise ValueError(f"need 1 <= min_cache_maxframes ({lo}) <= cache_maxframes ({mf}) <= max_cache_maxframes ({hi})")
        return {"maxframes": mf, "interval": max(1, int(w.get("cache_interval", 1))), "min": lo, "max": hi}

    def cache_state(self) -> dict:
        if not self.cached:
            return {"enabled": False}
        return {
            "enabled": self.kvo_shim.enabled if self.kvo_shim is not None else True,
            "mode": self.acceleration,
            **self.cached,
        }

    def _cache_command(self, pname: str, value) -> bool:
        if pname not in ("cache_maxframes", "cache_interval", "use_cached_attn"):
            return False
        if not self.cached:
            raise ValueError("cached attention is off (start the worker with worker.use_cached_attn: true)")
        if pname == "use_cached_attn":
            if self.acceleration == "tensorrt":
                raise ValueError("use_cached_attn is baked into the TensorRT UNet engine; restart the worker to change it")
            en = _truthy(value)
            if en and not self.kvo_shim.enabled:
                self.kvo_shim.reset()  # no stale frames from before the pause
            self.kvo_shim.enabled = en
        elif pname == "cache_maxframes":
            v, lo, hi = int(value), self.cached["min"], self.cached["max"]
            if not lo <= v <= hi:
                why = "TensorRT engine range" if self.acceleration == "tensorrt" else "worker.min/max_cache_maxframes"
                raise ValueError(f"cache_maxframes must be in [{lo}, {hi}] ({why})")
            self.stream.update_stream_params(cache_maxframes=v)
            self.cached["maxframes"] = v
        else:
            v = max(1, int(value))
            self.stream.update_stream_params(cache_interval=v)
            self.cached["interval"] = v
        self.emit("info", cached_attn=self.cache_state())
        return True

    # ── IP-Adapter / FaceID ───────────────────────────────────────────────────
    def _ipadapter_config(self, w: dict) -> dict | None:
        if not _truthy(w.get("use_ipadapter", False)):
            return None
        ip = dict(w.get("ipadapter") or {})
        typ = str(ip.get("type", "regular")).lower()
        faceid = typ == "faceid"
        missing = _missing_modules(["diffusers_ipadapter"] + (["insightface", "onnxruntime"] if faceid else []))
        if missing:
            msg = f"IP-Adapter off: {'; '.join(missing)}. Install with deploy/setup_faceid.sh (pod: WITH_FACEID=1)"
            if w.get("ipadapter_required"):
                raise RuntimeError(msg)
            self.log(msg)
            self.ip_disabled_reason = msg
            return None
        cfg = {
            "type": typ,
            "ipadapter_model_path": ip.get("ipadapter_model_path") or (
                "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin" if faceid else "h94/IP-Adapter/models/ip-adapter_sd15.bin"),
            "image_encoder_path": _resolve_hf_dir(ip.get("image_encoder_path") or "h94/IP-Adapter/models/image_encoder"),
            "scale": float(ip.get("scale", 0.8)),
            "num_image_tokens": int(ip.get("num_image_tokens", 16 if typ == "plus" else 4)),
            "style_image_key": "ipadapter_main",
        }
        if faceid:
            name = ip.get("insightface_model_name") or "buffalo_l"
            cfg["insightface_model_name"] = name
            # Diffusers_IPAdapter opens FaceAnalysis(root="~/.insightface/models") -> <root>/models/<name>
            mdir = Path("~/.insightface/models/models").expanduser() / name
            if not any(mdir.glob("*.onnx")):
                self.log(f"insightface models missing in {mdir}: insightface will try GitHub (run deploy/setup_faceid.sh)")
        self.ip_type, self.ip_key, self.ip_scale = typ, cfg["style_image_key"], cfg["scale"]
        self.ip_reference = ip.get("reference_image")
        self.faceid_bgr = bool(ip.get("insightface_bgr", True))
        return cfg

    def _setup_ipadapter(self, ip_cfg: dict) -> None:
        self.ipa = getattr(self.stream.stream, "ipadapter", None)
        if self.ipa is None:
            raise RuntimeError("IP-Adapter requested but the wrapper did not install it")
        self.faceid_state = {"locked": False}
        if self.ip_type == "faceid":
            face = getattr(self.ipa, "insightface_model", None)
            try:
                prov = face.det_model.session.get_providers()
            except Exception:  # noqa: BLE001
                prov = "?"
            self.log(f"FaceID: insightface {ip_cfg.get('insightface_model_name')} on {prov}")
            # ORT picks cuDNN kernels per input size on first use and detection retries 640..256 px
            # when no face is found: pay that once now (a grey frame has no face).
            t0 = time.perf_counter()
            try:
                self._ip_embeds(np.full((self.height, self.width, 3), 127, np.uint8))
            except ValueError:
                pass
            self.log(f"FaceID: detector warm-up {time.perf_counter() - t0:.1f}s")
        if self.ip_reference:
            try:
                self._ip_set_reference_path(self.ip_reference)
            except Exception as e:  # noqa: BLE001
                self.log(f"IP-Adapter: reference image {self.ip_reference!r} not used: {e}")

    def ip_state(self) -> dict:
        if self.ipa is None:
            return {"enabled": False, **({"reason": self.ip_disabled_reason} if self.ip_disabled_reason else {})}
        return {
            "type": self.ip_type,
            "enabled": bool(getattr(self.ipa, "enabled", True)),
            "scale": round(float(self.ip_scale), 3),
            **({"faceid": dict(self.faceid_state)} if self.ip_type == "faceid" else {"image": dict(self.faceid_state)}),
        }

    def _ip_embeds(self, bgr: np.ndarray):
        from PIL import Image

        # insightface (FaceID) expects OpenCV BGR order, the order IP-Adapter-FaceID's embeddings were
        # made with; CLIP (regular/plus adapters) expects RGB. Diffusers_IPAdapter just np.array()s the PIL.
        arr = bgr if (self.ip_type == "faceid" and self.faceid_bgr) else bgr[..., ::-1]
        return self.ipa.get_image_embeds(images=[Image.fromarray(np.ascontiguousarray(arr))])

    def _ip_apply(self, bgr: np.ndarray, source: str) -> bool:
        """Compute image tokens for `bgr` and swap them in. No face -> keep the previous tokens."""
        t0 = time.perf_counter()
        try:
            pos, neg = self._ip_embeds(bgr)
        except ValueError as e:
            if "face" not in str(e).lower():
                raise
            self.faceid_state.update(last="no_face", last_source=source, at=round(time.time(), 1))
            self.log(f"faceid: no face found in {source}; keeping {'the previous identity' if self.faceid_state.get('locked') else 'no identity'}")
            self.emit("info", ipadapter=self.ip_state())
            return False
        self.stream.stream._param_updater._embedding_cache[self.ip_key] = (pos, neg)
        self._recompose_prompt()
        ms = (time.perf_counter() - t0) * 1000
        self.faceid_state.update(locked=True, source=source, last="ok", last_source=source, ms=round(ms, 1), at=round(time.time(), 1))
        self.log(f"ipadapter: reference from {source} ({ms:.0f} ms)")
        self.emit("info", ipadapter=self.ip_state())
        return True

    def _ip_set_reference_path(self, path: str) -> bool:
        import cv2

        bgr = cv2.imread(str(Path(path).expanduser()))
        if bgr is None:
            raise ValueError(f"cannot read image {path!r}")
        return self._ip_apply(bgr, source=str(path))

    def _ip_command(self, pname: str, value) -> bool:
        if pname not in ("ipadapter_image", "faceid_capture", "faceid_clear", "ipadapter_scale", "ipadapter_enabled"):
            return False
        if self.ipa is None:
            raise ValueError("IP-Adapter is not active in this worker" + (f": {self.ip_disabled_reason}" if self.ip_disabled_reason else " (worker.use_ipadapter)"))
        if pname == "ipadapter_image":
            self._ip_set_reference_path(str(value))
            return True  # _ip_apply reported the result
        if pname == "faceid_capture":
            if _truthy(value):
                self._faceid_capture = True  # served by the next process() call, with the current frame
            return True
        if pname == "faceid_clear":
            if _truthy(value):
                self.stream.stream._param_updater._embedding_cache.pop(self.ip_key, None)
                self._recompose_prompt()  # the embedding hook falls back to zero image tokens = no identity
                self.faceid_state.update(locked=False, source=None, last="cleared", at=round(time.time(), 1))
        elif pname == "ipadapter_scale":
            s = float(value)
            self.ipa.set_scale(s)  # PyTorch processors
            self.ipa.scale = s     # read per step by the fork's hook (TensorRT scale vector, enable state)
            self.ip_scale = s
        else:
            self.ipa.enabled = _truthy(value)
        self.emit("info", ipadapter=self.ip_state())
        return True

    # ── prompts ───────────────────────────────────────────────────────────────
    def _recompose_prompt(self) -> None:
        """Re-run prompt blending + embedding hooks (appends the current IP-Adapter tokens). The text
        embedding comes from the updater's cache, so this is cheap."""
        self.stream.update_stream_params(prompt_list=[(self.prompt, 1.0)], prompt_interpolation_method="linear")

    def _refresh_sdxl_pooled(self) -> None:
        """Fork bug: prompt updates re-encode only the token embeddings; SDXL's pooled text embedding
        (added cond `text_embeds`, a big part of SDXL's prompt conditioning) keeps the first prompt's."""
        torch, s = self.torch, self.stream.stream
        if getattr(s, "add_text_embeds", None) is None:
            return
        do_cfg = float(getattr(s, "guidance_scale", 1.0)) > 1.0
        with torch.no_grad():
            out = s.pipe.encode_prompt(
                prompt=self.prompt, prompt_2=None, device=s.device, num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg, negative_prompt=self.negative_prompt, negative_prompt_2=None,
            )
        pooled, neg_pooled = out[2], out[3]
        s.add_text_embeds = torch.cat([neg_pooled, pooled], dim=0) if do_cfg else pooled
        s._sdxl_conditioning_cache.clear()  # the pipeline caches the batched cond by batch/cfg only
        s._cached_batch_size = None
        if self.cn is not None:
            self.cn._sdxl_conditioning_valid = False

    # ---------------------------------------------------------------------------
    def _setup_controlnets(self, cn_list: list[dict]) -> None:
        """Fix-ups on top of the fork's ControlNetModule:
        - the fork leaves PyTorch ControlNets on the CPU (it only moves them for TensorRT export):
          nets served by a TensorRT engine stay there (the hook uses only their model_id), the
          others are moved to cuda
        - one TensorRT engine instance per model id
        - "depth" -> FastDepth (fp16, tensor in/out) unless preprocessor_params.fast is false
        - controlnet_aux detectors (OpenPose, HED) load on the CPU -> move to cuda
        - slow preprocessors run on a background thread (see ASYNC_PREPROCESSORS)
        Control images are fed by _update_controls(), not by the fork's orchestrator: its
        pipelined mode drops frames and its PIL fallback has a key bug ('image' vs 'image_safe').
        The same images reach TensorRT ControlNet engines (the fork's hook swaps in the engine).
        """
        torch = self.torch
        cn = self.cn
        s = self.stream.stream
        engines: dict = {}
        for e in list(getattr(s, "controlnet_engines", None) or []):
            mid = getattr(e, "model_id", None)
            if mid and mid not in engines:
                engines[mid] = e
        if engines:
            s.controlnet_engines = list(engines.values())
            cn._engines_cache_valid = False
        seen: dict[str, object] = {}
        for i, m in enumerate(cn.controlnets):
            mid = cn_list[i]["model_id"]
            self.cn_meta[i]["runtime"] = "tensorrt" if mid in engines else "pytorch"
            if mid in seen:  # e.g. tile + color share control_v11f1e_sd15_tile: keep one copy
                cn.controlnets[i] = seen[mid]
                continue
            if m is not None:
                seen[mid] = cn.controlnets[i] = m if mid in engines else m.to(device="cuda", dtype=torch.float16)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.cn_async: set[int] = set()
        for i, c in enumerate(cn_list):
            name, pre_name = self.cn_names[i], c.get("preprocessor")
            params = dict(c.get("preprocessor_params") or {})
            if pre_name == "depth" and params.get("fast", True):
                cn.preprocessors[i] = FastDepth(
                    params.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf"),
                    int(params.get("detect_resolution", 378)), self.height, self.width,
                )
                self.log(f"controlnet {name}: FastDepth ({cn.preprocessors[i].params['model_name']})")
            pre = cn.preprocessors[i]
            if pre_name in ("openpose", "hed"):
                for attr in ("detector", "model"):
                    try:
                        det = getattr(pre, attr, None)
                        if det is not None and hasattr(det, "to"):
                            det.to("cuda")
                    except Exception as e:  # noqa: BLE001
                        self.log(f"controlnet {name}: could not move {attr} to cuda: {e}")
            if c.get("async", pre_name in ASYNC_PREPROCESSORS):
                self.cn_async.add(i)
            self.cn_meta[i]["async"] = i in self.cn_async
        # Background preprocessing thread for the slow ones.
        self._async_frame = None
        self._async_evt = threading.Event()
        self._async_stream = torch.cuda.Stream()
        if self.cn_async:
            threading.Thread(target=self._async_loop, daemon=True).start()
        # Prime every control image once so enabling a net later never hits an empty slot.
        x = torch.full((1, 3, self.height, self.width), 0.5, device="cuda", dtype=torch.float16)
        for i in range(len(cn.controlnets)):
            try:
                self._set_control_image(i, self._run_pre(i, x))
            except Exception as e:  # noqa: BLE001
                self.log(f"controlnet {self.cn_names[i]}: priming failed: {e}")
        self.cn_timing_ms: dict[str, float] = {}
        self.log("controlnets: " + ", ".join(f"{m['name']}={m['runtime']}" for m in self.cn_meta))

    def _run_pre(self, i: int, x):
        """Run preprocessor i on x (1x3xHxW fp16 [0,1] cuda) -> 1x3xHxW fp16 [0,1] cuda."""
        torch, F = self.torch, self.torch.nn.functional
        pre = self.cn.preprocessors[i]
        y = x if pre is None else pre.process_tensor(x[0])
        if not isinstance(y, torch.Tensor):  # PIL fallback
            y = torch.from_numpy(np.asarray(y.convert("RGB"))).permute(2, 0, 1).float().div(255.0)
        if y.dim() == 3:
            y = y.unsqueeze(0)
        y = y.to(device="cuda", dtype=torch.float16)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        if y.shape[-2:] != (self.height, self.width):
            y = F.interpolate(y.float(), size=(self.height, self.width), mode="bilinear", align_corners=False).half()
        return y.clamp_(0, 1)

    def _set_control_image(self, i: int, img) -> None:
        cn = self.cn
        with cn._collections_lock:
            cn.controlnet_images[i] = img
            cn._prepared_tensors = []
            cn._prepared_batch = None  # force the UNet hook to re-prepare at the real batch size
            cn._images_version += 1
            cn._sdxl_conditioning_valid = False  # SDXL: re-read text_embeds/time_ids (prompt may have changed)

    def _active(self) -> list[int]:
        cn = self.cn
        with cn._collections_lock:
            return [i for i in range(len(cn.controlnets)) if cn.enabled_list[i] and cn.controlnet_scales[i] > 0]

    def _update_controls(self, x) -> None:
        active = self._active()
        if any(i in self.cn_async for i in active) and not self._async_evt.is_set():
            xc = x.clone()
            ev = self.torch.cuda.Event()
            ev.record()
            self._async_frame = (xc, ev)
            self._async_evt.set()
        for i in active:
            if i not in self.cn_async:
                self._set_control_image(i, self._run_pre(i, x))

    def _async_loop(self) -> None:
        torch = self.torch
        while not self._stop:
            if not self._async_evt.wait(timeout=0.5):
                continue
            self._async_evt.clear()
            if self._async_frame is None:
                continue
            x, ev = self._async_frame
            self._async_stream.wait_event(ev)
            x.record_stream(self._async_stream)
            for i in self._active():
                if i not in self.cn_async:
                    continue
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode(), torch.cuda.stream(self._async_stream):
                        img = self._run_pre(i, x)
                    self._async_stream.synchronize()
                    self._set_control_image(i, img)
                    self.cn_timing_ms[self.cn_names[i]] = round((time.perf_counter() - t0) * 1000, 1)
                except Exception as e:  # noqa: BLE001
                    self.emit("error", msg=f"controlnet {self.cn_names[i]} preprocessing failed: {e}")
                    time.sleep(0.5)

    def cn_state(self) -> list[dict]:
        if self.cn is None:
            return []
        out = []
        with self.cn._collections_lock:
            for i, meta in enumerate(self.cn_meta):
                out.append({
                    **meta,
                    "scale": round(float(self.cn.controlnet_scales[i]), 3),
                    "enabled": bool(self.cn.enabled_list[i]),
                    **({"async_ms": self.cn_timing_ms[meta["name"]]} if meta["name"] in getattr(self, "cn_timing_ms", {}) else {}),
                })
        return out

    def _cn_index(self, name) -> int:
        if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
            i = int(name)
        elif name in self.cn_names:
            i = self.cn_names.index(name)
        else:
            raise ValueError(f"unknown controlnet {name!r} (have {self.cn_names})")
        if not 0 <= i < len(self.cn_names):
            raise ValueError(f"controlnet index {i} out of range")
        return i

    def _cn_command(self, pname: str, value) -> bool:
        """Handle controlnet_* params. Returns False if pname is not a ControlNet param."""
        if not pname.startswith("controlnet"):
            return False
        if self.cn is None:
            raise ValueError("ControlNet is not enabled in this worker (no worker.controlnets)")
        if ":" in pname:  # flat form: controlnet_scale:depth = 0.6
            base, cname = pname.split(":", 1)
            value = {"name": cname, ("scale" if base == "controlnet_scale" else "enabled"): value}
            pname = base
        if pname == "controlnet_scale":
            self.cn.update_controlnet_scale(self._cn_index(value["name"]), float(value["scale"]))
        elif pname == "controlnet_scales":
            for cname, sc in dict(value).items():
                self.cn.update_controlnet_scale(self._cn_index(cname), float(sc))
        elif pname == "controlnet_enabled":
            self.cn.update_controlnet_enabled(self._cn_index(value["name"]), _truthy(value["enabled"]))
        elif pname == "controlnet_params":
            pre = self.cn.preprocessors[self._cn_index(value["name"])]
            if pre is None:
                raise ValueError(f"controlnet {value['name']!r} has no preprocessor")
            params = dict(value.get("params") or {})
            if isinstance(getattr(pre, "params", None), dict):
                pre.params.update(params)
            for k, v in params.items():
                if hasattr(pre, k):
                    setattr(pre, k, v)
        else:
            raise ValueError(f"unknown controlnet param {pname!r}")
        self.emit("info", controlnets=self.cn_state())
        return True

    def on_command(self, name: str, payload: dict) -> None:
        if name == "set_prompt":
            text = str(payload.get("text", ""))
            self.prompt = text
            # update_stream_params, not wrapper.update_prompt(): that one clears ALL of the updater's
            # caches first, incl. the IP-Adapter/FaceID image tokens and the negative prompt.
            self.stream.update_stream_params(prompt_list=[(text, 1.0)], prompt_interpolation_method="linear")
            if self.is_sdxl:
                self._refresh_sdxl_pooled()
            self.log(f"prompt: {text[:80]}")
            return
        if name != "set_param":
            self.log(f"unknown command {name!r} ignored")
            return

        pname, value = str(payload.get("name")), payload.get("value")
        if self._cn_command(pname, value) or self._ip_command(pname, value) or self._cache_command(pname, value):
            self.log(f"set {pname}={value!r}")
            return
        if pname in ("guidance_scale", "guidance"):
            self.guidance_scale = float(value)
            self.stream.update_stream_params(guidance_scale=self.guidance_scale)
            if self.cfg_type == "none" and self.guidance_scale > 1.0:
                self.log("note: guidance_scale has no effect with cfg_type 'none'")
        elif pname == "delta":
            self.delta = float(value)
            self.stream.update_stream_params(delta=self.delta)
        elif pname == "seed":
            self.seed = int(value)
            self.stream.update_stream_params(seed=self.seed)
        elif pname == "negative_prompt":
            # The fork applies negative prompts only together with a prompt re-encode.
            self.negative_prompt = str(value or "")
            self.stream.update_stream_params(
                prompt_list=[(self.prompt, 1.0)], negative_prompt=self.negative_prompt or " ",
                prompt_interpolation_method="linear",
            )
            if self.is_sdxl:
                self._refresh_sdxl_pooled()
        elif pname == "t_index_list":
            if isinstance(value, str):
                value = [int(v) for v in value.replace(",", " ").split()]
            new = [int(v) for v in value]
            if not new:
                raise ValueError("t_index_list must not be empty")
            if self.acceleration == "tensorrt" and len(new) != len(self.t_index_list):
                # The TensorRT UNet runs a CUDA graph captured for the start batch (= list length).
                raise ValueError("with tensorrt, t_index_list length must stay the same (engine batch size)")
            self.stream.update_stream_params(t_index_list=new)
            self.t_index_list = new
        elif pname == "num_inference_steps":
            self.stream.update_stream_params(num_inference_steps=int(value))
            self.t_index_list = list(self.stream.stream.t_list)
            self.log(f"num_inference_steps={int(value)} -> t_index_list={self.t_index_list}")
        elif pname == "strength":
            # Convenience: 0..1 denoise strength (maps to t_index; keeps the list length).
            steps = len(self.stream.stream.timesteps)
            s = min(max(float(value), 0.0), 1.0)
            n = len(self.t_index_list)
            first = int(round((1.0 - s) * (steps - 1)))
            new = [min(first + i * max((steps - 1 - first) // max(n, 1), 1), steps - 1) for i in range(n)]
            self.stream.update_stream_params(t_index_list=new)
            self.t_index_list = new
        else:
            self.log(f"unknown param {pname!r}={value!r} ignored")
            return
        self.log(f"set {pname}={value!r}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # insightface imports albumentations, which phones PyPI
    sys.exit(SDWorker().run())
