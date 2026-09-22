"""Classic StreamDiffusion worker (SD-Turbo / SD1.5+LCM-LoRA img2img).

Runs in the StreamDiffusion venv (daydreamlive fork), NOT the FluxRT venv:
    <sd-venv>/bin/python sd_worker.py --in-shm ... --config cfg.json
Only imports shm_protocol (by path) from this repo; never the fluxrt package.

Config (the "worker" dict of configs/sd_config.json), all optional:
    model_id            "stabilityai/sd-turbo" | "Lykon/dreamshaper-8" | local path
    t_index_list        [35, 45]   (indices into num_inference_steps)
    num_inference_steps 50
    acceleration        "none" (PyTorch SDPA, default) | "tensorrt" | "xformers" (broken
                        in the fork -> falls back to "none" unless force_xformers)
    use_tiny_vae        true
    cfg_type            "none" | "self" | "full" | "initialize"
    guidance_scale      1.0 (turbo) / 1.2+ with cfg_type "self"
    delta               1.0
    seed                2
    use_denoising_batch true
    negative_prompt     ""
    lora_dict           {"latent-consistency/lcm-lora-sdv1-5": 1.0}  (SD1.5 models)
    use_lcm_lora        true -> adds the right LCM LoRA automatically (ignored for turbo)
    engine_dir          "engines" (relative to worker_cwd; TensorRT only)
    similar_image_filter false / similar_image_filter_threshold 0.98
    warmup_frames       10
    controlnets         list of {name, model_id, preprocessor, conditioning_scale, enabled,
                        preprocessor_params?, conditioning_channels?}. SD1.5 base only
                        (lllyasviel/control_v11*). All listed nets are loaded at start;
                        enable/scale switch live. "enabled": false costs VRAM but no time
                        (the fork skips both the preprocessor and the ControlNet pass).
    use_controlnet      default true when "controlnets" is non-empty

Live ControlNet commands (set_param):
    {"name": "controlnet_scale",   "value": {"name": "depth", "scale": 0.6}}
    {"name": "controlnet_scales",  "value": {"depth": 0.6, "canny": 0.2}}
    {"name": "controlnet_enabled", "value": {"name": "openpose", "enabled": true}}
    {"name": "controlnet_params",  "value": {"name": "canny", "params": {"low_threshold": 50}}}
    {"name": "controlnet_scale:depth", "value": 0.6}     (flat form, handy for sliders)
    {"name": "controlnet_enabled:depth", "value": false}
"""

from __future__ import annotations

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
    def setup(self) -> None:
        import torch
        from streamdiffusion import StreamDiffusionWrapper

        self.torch = torch
        w = dict(self.cfg.get("worker") or {})
        self.wcfg = w
        self.model_id = w.get("model_id", "stabilityai/sd-turbo")
        is_turbo = "turbo" in self.model_id.lower()
        self.t_index_list = list(w.get("t_index_list", [35, 45]))
        self.acceleration = w.get("acceleration", "none")
        if self.acceleration == "xformers" and not w.get("force_xformers", False):
            # The fork's patched diffusers expects attention processors to return
            # (hidden_states, kvo_cache); diffusers' XFormersAttnProcessor returns only
            # hidden_states, which is then unpacked along the BATCH dim: silently wrong
            # with 2 denoising steps, a crash with 1 or 3+. PyTorch SDPA ("none") uses the
            # fork's AttnProcessor2_0 and is just as fast on torch 2.7 / Ada.
            self.log("acceleration 'xformers' is broken in this fork; using 'none' (PyTorch SDPA)")
            self.acceleration = "none"
        self.cfg_type = w.get("cfg_type", "none" if is_turbo else "self")
        self.guidance_scale = float(w.get("guidance_scale", 1.0 if is_turbo else 1.2))
        self.delta = float(w.get("delta", 1.0))
        self.seed = int(w.get("seed", 2))
        self.negative_prompt = w.get("negative_prompt", "")
        self.prompt = self.cfg.get("default_prompt") or w.get("prompt", "")

        lora_dict = w.get("lora_dict")
        use_lcm_lora = w.get("use_lcm_lora")
        if use_lcm_lora is None and not is_turbo and not lora_dict:
            use_lcm_lora = True  # SD1.5 checkpoints need the LCM LoRA for few-step sampling

        # ControlNets (SD1.5 zoo). Names are our handles for live commands.
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

        self.log(
            f"building StreamDiffusionWrapper model={self.model_id} t={self.t_index_list} "
            f"{self.width}x{self.height} accel={self.acceleration} cfg={self.cfg_type}"
        )
        t0 = time.time()
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=self.model_id,
            t_index_list=self.t_index_list,
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
            engine_dir=w.get("engine_dir", "engines"),
            use_safety_checker=False,
            use_controlnet=self.use_controlnet,
            controlnet_config=cn_config or None,
        )
        self.cn = getattr(self.stream.stream, "_controlnet_module", None) if self.use_controlnet else None
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
        self.emit(
            "info",
            model=self.model_id,
            acceleration=self.acceleration,
            t_index_list=self.t_index_list,
            cfg_type=self.cfg_type,
            warmup_fps=round(fps, 1),
            load_s=round(load_s, 1),
            worker_res=f"{self.width}x{self.height}",
            controlnets=self.cn_state(),
        )
        self.log(f"ready: load {load_s:.1f}s, warm-up {fps:.1f} fps, {self.gpu_mb():.0f} MB reserved")

    # ---------------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
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

    # ---------------------------------------------------------------------------
    def _setup_controlnets(self, cn_list: list[dict]) -> None:
        """Fix-ups on top of the fork's ControlNetModule (non-TensorRT path):
        - the fork leaves PyTorch ControlNets on the CPU unless TensorRT is used -> move to cuda
        - "depth" -> FastDepth (fp16, tensor in/out) unless preprocessor_params.fast is false
        - controlnet_aux detectors (OpenPose, HED) load on the CPU -> move to cuda
        - slow preprocessors run on a background thread (see ASYNC_PREPROCESSORS)
        Control images are fed by _update_controls(), not by the fork's orchestrator: its
        pipelined mode drops frames and its PIL fallback has a key bug ('image' vs 'image_safe').
        """
        torch = self.torch
        cn = self.cn
        seen: dict[str, object] = {}
        for i, m in enumerate(cn.controlnets):
            mid = cn_list[i]["model_id"]
            if mid in seen:  # e.g. tile + color share control_v11f1e_sd15_tile: keep one copy
                cn.controlnets[i] = seen[mid]
                continue
            if m is not None:
                seen[mid] = m.to(device="cuda", dtype=torch.float16)
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
            en = value["enabled"]
            en = en.lower() in ("1", "true", "on", "yes") if isinstance(en, str) else bool(en)
            self.cn.update_controlnet_enabled(self._cn_index(value["name"]), en)
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
            # Single-prompt update: one CLIP encode, no rebuild.
            self.stream.update_prompt(text, warn_about_conflicts=False)
            self.log(f"prompt: {text[:80]}")
            return
        if name != "set_param":
            self.log(f"unknown command {name!r} ignored")
            return

        pname, value = str(payload.get("name")), payload.get("value")
        if self._cn_command(pname, value):
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
        elif pname == "t_index_list":
            if isinstance(value, str):
                value = [int(v) for v in value.replace(",", " ").split()]
            new = [int(v) for v in value]
            if not new:
                raise ValueError("t_index_list must not be empty")
            if self.acceleration == "tensorrt" and len(new) != len(self.t_index_list):
                # TRT engines are built for a fixed denoising-batch size.
                raise ValueError("with tensorrt, t_index_list length must stay the same (engine batch size)")
            self.stream.update_stream_params(t_index_list=new)
            self.t_index_list = new
        elif pname == "num_inference_steps":
            self.stream.update_stream_params(num_inference_steps=int(value))
            self.t_index_list = list(self.stream.stream.t_list)
            self.log(f"num_inference_steps={int(value)} -> t_index_list={self.t_index_list}")
        elif pname == "strength":
            # Convenience: 0..1 denoise strength for a single-step list (maps to t_index).
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
    sys.exit(SDWorker().run())
