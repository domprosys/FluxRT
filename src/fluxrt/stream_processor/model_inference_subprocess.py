import torch
import time
import cv2
import numpy as np
import json
import hashlib
from pathlib import Path
from safetensors.torch import load_file, save_file
from multiprocessing import Process, Value, Manager
from queue import Empty
from PIL import Image

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM, AutoConfig
from accelerate import init_empty_weights

from fluxrt.stream_processor.interpolation_model import IFNet
from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
from fluxrt.utils.shared_tensor import SharedTensor
from fluxrt.stream_processor.pipeline import Flux2KleinPipeline
from fluxrt.stream_processor.update_controller import UpdateController
from fluxrt.stream_processor.postprocessors import (
    BasePostProcessor,
    LivePortraitPostProcessor,
)

from fluxrt.flow_upscaler.upscaler_unet import UpscalerUNet
from fluxrt.flow_upscaler.flow_upscaler_pipeline import FlowUpscalerPipeline
from fluxrt.stream_processor.flux_tiny_vae import DiffusersTAEF2Wrapper


class ModelInferenceSubprocess:
    def __init__(
        self,
        config: dict,
        input_shared_tensor_name: str,
        output_batch_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
        interpolation_exp_value=None,
    ):
        self.running = Value("b", False)
        self.memory_reserved = Value("i", 0)
        self.process = None
        self.config = config
        self.height = self.config["resolution"]["height"]
        self.width = self.config["resolution"]["width"]
        self.resolution = self.config["resolution"]
        self.prompt = self.config["default_prompt"]
        self.logging = self.config.get("logging", True)
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        manager = Manager()
        self.command_queue = manager.Queue()
        self.shared_state = manager.dict()
        self.interpolation_exp = self.config.get("interpolation_exp", 1)
        # Live interpolation factor: the output buffer is allocated for up to
        # 2**_max_interp_exp frames; the active count comes from the shared Value.
        self.interpolation_exp_value = interpolation_exp_value
        self._max_interp_exp = max(self.interpolation_exp, 3)

    def enable_quantization(self):
        """
        Should be called before the subprocess is started.
        """
        self.config["enable_int8_quantization"] = True

    def init_process_state(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        # Device the Qwen3 text encoder lives on. "cpu" offloads it to system RAM
        # (low-VRAM mode): it only runs on prompt change, so the per-frame denoise
        # loop is unaffected and ~4.5GB of GPU VRAM is freed.
        self.text_encoder_device = self.config.get("text_encoder_device", "cuda")
        # Optional hard VRAM cap (fraction of total device memory) — lets you
        # simulate a smaller GPU on a bigger one, e.g. 0.5 ≈ 12GB on a 24GB card.
        frac = self.config.get("cuda_memory_fraction")
        if frac:
            torch.cuda.set_per_process_memory_fraction(float(frac), 0)
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }

    def load_models_without_quantization(self):
        device = self.device
        dtype = torch.bfloat16

        models_path = self.config["models_path"]
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=device
        )
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            f"{models_path}/transformer", local_files_only=True, device=device
        ).to(dtype)

        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{models_path}/text_encoder", local_files_only=True
        ).to(self.text_encoder_device, dtype)
        self.text_encoder.eval()
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{models_path}/tokenizer", local_files_only=True, device=device
        )

    def load_quantized_models(self):
        from optimum.quanto import requantize
        from fluxrt.stream_processor.quantized_flux2 import (
            QuantizedFlux2Transformer2DModel,
        )

        models_path = self.config["models_path"]
        int8_models_path = self.config["int8_models_path"]

        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(
            int8_models_path, local_files_only=True
        )
        qtransformer.to(device=self.device, dtype=self.dtype)
        self.transformer = qtransformer._wrapped

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=self.device
        )

        if self.text_encoder_device == "cpu":
            if getattr(self, "_skip_encoder", False):
                # Cached prompt embeds cover the whole cycle, so the encoder is
                # never used at runtime — skip loading the ~8GB Qwen3 entirely
                # (cycle-only / kiosk mode). Saves both the disk load and the
                # ~7.5-15GB of system RAM it would occupy.
                self.text_encoder = None
                self.tokenizer = None
                print(
                    "[FluxRT] skipped text encoder load (using cached prompt embeds)",
                    flush=True,
                )
            else:
                # Low-VRAM mode: keep the full Qwen3-4B encoder in system RAM (the base,
                # non-quantized weights — best conditioning quality) and run it on CPU
                # only when the prompt changes. Frees ~4.5GB of GPU VRAM.
                # Default bf16 (~7.5GB RAM). fp32 is ~2x faster to encode but uses
                # ~15GB RAM — risky if any process is orphaned, so it's opt-in via
                # text_encoder_cpu_dtype. Embeddings are cast to the transformer
                # dtype when moved to the GPU regardless.
                cpu_dtype = getattr(
                    torch, self.config.get("text_encoder_cpu_dtype", "bfloat16")
                )
                text_encoder = Qwen3ForCausalLM.from_pretrained(
                    f"{models_path}/text_encoder", local_files_only=True
                )
                text_encoder.eval()
                text_encoder.to("cpu", dtype=cpu_dtype)
                self.text_encoder = text_encoder
                self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                    f"{models_path}/tokenizer", local_files_only=True
                )
        else:
            config = AutoConfig.from_pretrained(
                f"{int8_models_path}/text_encoder", local_files_only=True
            )
            with init_empty_weights():
                text_encoder = Qwen3ForCausalLM(config)

            with open(f"{int8_models_path}/text_encoder/quanto_qmap.json", "r") as f:
                qmap = json.load(f)
            state_dict = load_file(
                f"{int8_models_path}/text_encoder/model.safetensors"
            )
            requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
            text_encoder.eval()
            text_encoder.to(self.device, dtype=self.dtype)
            self.text_encoder = text_encoder

            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                f"{int8_models_path}/tokenizer", local_files_only=True
            )

    def load_models(self):
        self.interpolation_model = IFNet()
        self.interpolation_model.load_state_dict(
            load_file("RIFE-safetensors/flownet.safetensors")
        )
        # interpolation model reqires torch.float16, not torch.bfloat16 to avoid pixelation on grid sample layers
        self.interpolation_model.to(self.device, torch.float16)
        self.interpolation_model.eval()

        if self.config.get("enable_int8_quantization", False):
            self.load_quantized_models()
        else:
            self.load_models_without_quantization()

        if self.config.get("enable_flow_upscaler", False):
            self.upscaler_unet = UpscalerUNet()
            state_dict = state_dict = load_file(
                "FlowUpscaler/flow_upscaler.safetensors"
            )
            self.upscaler_unet.load_state_dict(state_dict)
            self.upscaler_unet.to(self.device, self.dtype)
            self.upscaler_pipe = FlowUpscalerPipeline(
                self.upscaler_unet, self.scheduler
            )
        else:
            self.upscaler_pipe = None

        if self.config.get("enable_tiny_vae", False):
            self.vae = DiffusersTAEF2Wrapper(path="taef2/taef2.safetensors").to(
                self.device, self.dtype
            )
        else:
            models_path = self.config["models_path"]
            self.vae = AutoencoderKLFlux2.from_pretrained(
                f"{models_path}/vae", local_files_only=True, device=self.device
            ).to(self.dtype)

        if self.config.get("compile_models", False):
            self.transformer = torch.compile(
                self.transformer,
            )
            self.vae = torch.compile(
                self.vae,
            )
            self.interpolation_model = torch.compile(
                self.interpolation_model,
            )

        reference_image_seq_len = None
        if self.config.get("use_reference_image", False):
            reference_image_res = self.config["reference_image_resolution"]
            reference_image_seq_len = (reference_image_res["width"] // 16) * (
                reference_image_res["height"] // 16
            )

        self.update_controller = UpdateController(
            self.config,
            self.height,
            self.width,
            compression_ratio=16,
            reference_image_seq_len=reference_image_seq_len,
        )

        self.pipe = Flux2KleinPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            update_controller=self.update_controller,
            subprocess_config=self.config,
            upscaler_pipeline=self.upscaler_pipe,
        )
        if self.text_encoder_device == self.device:
            self.pipe.to(self.device)
        else:
            # Low-VRAM offload: move the GPU modules explicitly and keep the text
            # encoder on CPU. Calling pipe.to(device) would drag the CPU-resident
            # encoder onto the GPU — a transient spike (≈15GB fp32 / 7.5GB bf16)
            # that the caching allocator never releases, inflating reserved memory
            # and OOM-ing a small card during load.
            self.transformer.to(self.device)
            self.vae.to(self.device)
            if self.text_encoder is not None:
                self.text_encoder.to(self.text_encoder_device)
            torch.cuda.empty_cache()

        if self.config.get("use_lora", False):
            self.pipe.load_lora_weights(self.config.get("lora_weights_path", ""))

        self.lip_processor: BasePostProcessor | None = None
        self.lip_active = False
        lp_cfg = self.config.get("lip_transfer", {})
        if lp_cfg.get("enable", False):
            self.lip_processor = LivePortraitPostProcessor(
                models_dir=lp_cfg["models_dir"]
            )

    # ── prompt-embedding disk cache ──────────────────────────────────────────
    # When cache_prompt_embeds is set, the pre-encoded cycle embeddings are
    # saved to disk keyed by the prompt list + encode params. On a later launch
    # with the same prompts we load them instead of re-encoding — and, since the
    # text encoder is then unused (cycle-only / kiosk mode), we skip loading the
    # ~8GB Qwen3 encoder entirely. First launch (cache miss) encodes + saves.

    def _prompt_cache_params(self) -> dict:
        return {
            "max_sequence_length": 512,
            "text_encoder_out_layers": [9, 18, 27],
            "text_encoder_cpu_dtype": self.config.get(
                "text_encoder_cpu_dtype", "bfloat16"
            ),
            "models_path": self.config.get("models_path", ""),
            "dtype": str(self.dtype),
            "version": 1,
        }

    def _prompt_cache_key(self, cycle: list) -> str:
        payload = json.dumps(
            {"prompts": cycle, "params": self._prompt_cache_params()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _prompt_cache_dir(self, cycle: list) -> Path:
        return Path("prompt_embeds_cache") / self._prompt_cache_key(cycle)

    def _prompt_cache_valid(self, cycle: list) -> bool:
        d = self._prompt_cache_dir(cycle)
        manifest = d / "manifest.json"
        if not (d / "embeds.safetensors").exists() or not manifest.exists():
            return False
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        # The key already encodes prompts+params; re-verify to guard collisions.
        return m.get("prompts") == cycle and m.get("params") == self._prompt_cache_params()

    def _save_cached_cycle_embeds(self, cycle: list) -> None:
        d = self._prompt_cache_dir(cycle)
        d.mkdir(parents=True, exist_ok=True)
        tensors = {
            str(i): emb.detach().to("cpu").contiguous()
            for i, emb in enumerate(self.cycle_embeds)
        }
        save_file(tensors, str(d / "embeds.safetensors"))
        with open(d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {"prompts": cycle, "params": self._prompt_cache_params()}, f, indent=2
            )
        print(
            f"[FluxRT] cached {len(tensors)} prompt embeds -> {d}",
            flush=True,
        )

    def _load_cached_cycle_embeds(self, cycle: list) -> list:
        tensors = load_file(str(self._prompt_cache_dir(cycle) / "embeds.safetensors"))
        return [tensors[str(i)].to(self.device, self.dtype) for i in range(len(cycle))]

    def _prepare_prompt_cache(self) -> None:
        """Decide (before loading models) whether cached embeds let us skip the
        text encoder. Sets self._cache_prompts and self._skip_encoder."""
        self._cache_prompts = bool(self.config.get("cache_prompt_embeds", False))
        self._skip_encoder = False
        cycle = self.config.get("prompt_cycle") or []
        if self._cache_prompts and cycle and self._prompt_cache_valid(cycle):
            self._skip_encoder = True
            print(
                "[FluxRT] prompt-embeds cache hit — will skip loading the text encoder",
                flush=True,
            )
        elif self._cache_prompts and cycle:
            print(
                "[FluxRT] prompt-embeds cache miss — encoding once, then caching",
                flush=True,
            )

    def _encode_prompt_to_gpu(self, prompt):
        # Encode on whatever device the text encoder lives on (CPU in low-VRAM
        # mode), then move the small (~8MB) embeddings to the transformer's device.
        _t0 = time.time()
        embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.text_encoder_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        embeds = embeds.to(self.device, self.dtype)
        if self.config.get("logging", False):
            print(
                f"[FluxRT] encoded prompt on {self.text_encoder_device} in "
                f"{time.time() - _t0:.2f}s: {prompt[:50]!r}",
                flush=True,
            )
        return embeds

    def update_prompt_embeds(self, prompt):
        if self.text_encoder is None:
            print(
                "[FluxRT] live prompt change ignored — running in cached/kiosk mode "
                "(text encoder not loaded). Set cache_prompt_embeds=false (or clear "
                "prompt_embeds_cache/) to type custom prompts.",
                flush=True,
            )
            return
        self.prompt_embeds = self._encode_prompt_to_gpu(prompt)
        self.update_controller.reset_cache()

    def precompute_prompt_cycle(self):
        """Pre-encode the config's prompt_cycle list once at startup so that
        cycling between them later is instant (no per-switch CPU re-encode).
        With cache_prompt_embeds set, embeds are loaded from / saved to disk."""
        cycle = self.config.get("prompt_cycle") or []
        if getattr(self, "_skip_encoder", False):
            self.cycle_embeds = self._load_cached_cycle_embeds(cycle)
            print(
                f"[FluxRT] loaded {len(self.cycle_embeds)} cached cycle prompts "
                "(encoder skipped)",
                flush=True,
            )
        else:
            self.cycle_embeds = [self._encode_prompt_to_gpu(p) for p in cycle]
            if getattr(self, "_cache_prompts", False) and self.cycle_embeds:
                self._save_cached_cycle_embeds(cycle)
        if self.cycle_embeds:
            self.cycle_index = 0
            self.prompt_embeds = self.cycle_embeds[0]
            self.update_controller.reset_cache()
            print(
                f"[FluxRT] ready with {len(self.cycle_embeds)} cycle prompts",
                flush=True,
            )

    def _apply_prompt_index(self, idx: int) -> None:
        """[child process] Instantly switch to a pre-encoded cycle prompt."""
        if 0 <= idx < len(getattr(self, "cycle_embeds", [])):
            self.cycle_index = idx
            self.prompt_embeds = self.cycle_embeds[idx]
            self.update_controller.reset_cache()

    def set_prompt_index(self, idx: int) -> None:
        """[main process] Queue a switch to a pre-encoded cycle prompt."""
        self.command_queue.put(("set_prompt_index", idx))

    # ── live advanced generation controls ───────────────────────────────────
    # Exposed via the GUI "Advanced" panel (gated by show_advanced_controls).
    # steps/seed live in process_state (read each frame); the scheduler knobs
    # rebuild the scheduler from its base config; sigmas are passed to the pipe.

    def _init_gen_params(self) -> None:
        # Snapshot the loaded scheduler's config so we can rebuild it with
        # overrides without mutating its frozen config in place.
        self._base_sched_config = dict(self.scheduler.config)
        self._sched_overrides: dict = {}
        self.custom_sigmas = None
        self._rife_scale = float(self.config.get("rife_scale", 1.0))
        # Flow upscaler is a load-time capability (enable_flow_upscaler loads the
        # model + sizes the output 2x); this runtime flag toggles whether the
        # flow super-resolution actually runs (off = base decode + cheap resize).
        self._flow_upscale_on = bool(self.config.get("flow_upscale_on", False))
        self.pipe._flow_upscale_on = self._flow_upscale_on

    def _reset_gen_caches(self) -> None:
        # Per-timestep spatial caches and the update mask depend on the timestep
        # schedule, so clear them when the steps/schedule change.
        try:
            self.pipe.spatial_cache.clear()
        except Exception:
            pass
        self.update_controller.reset_cache()

    def _rebuild_scheduler(self) -> None:
        # FlowMatchEuler is the sampler FLUX.2-klein is trained for; the custom
        # pipeline feeds a flow-sigma schedule that only this scheduler accepts.
        new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self._base_sched_config, **self._sched_overrides
        )
        self.scheduler = new_scheduler
        self.pipe.scheduler = new_scheduler
        self._reset_gen_caches()
        print(
            f"[FluxRT] scheduler rebuilt: overrides={self._sched_overrides}",
            flush=True,
        )

    def _apply_gen_param(self, name: str, value) -> None:
        """[child process] Apply a live advanced-control change."""
        if name == "steps":
            self.process_state["steps"] = int(value)
            self._reset_gen_caches()
        elif name == "seed":
            self.process_state["seed"] = int(value)
        elif name == "sigmas":
            # value: list[float] (descending) or None for the default schedule
            self.custom_sigmas = value
            self._reset_gen_caches()
        elif name == "rife_scale":
            # RIFE optical-flow scale (1.0 = default; lower = coarser flow).
            self._rife_scale = float(value)
        elif name == "flow_upscale_on":
            self._flow_upscale_on = bool(value)
            self.pipe._flow_upscale_on = self._flow_upscale_on
            # Output frame size flips between base and 2x, so drop the stale
            # previous frame to avoid a size mismatch in interpolation.
            self.previous_frame = None
        elif name in (
            "shift",
            "use_dynamic_shifting",
            "stochastic_sampling",
            "time_shift_type",
            "use_beta_sigmas",
        ):
            self._sched_overrides[name] = value
            self._rebuild_scheduler()
        else:
            print(f"[FluxRT] ignoring unknown gen param: {name}", flush=True)

    def set_gen_param(self, name: str, value) -> None:
        """[main process] Queue a live advanced-control change."""
        self.command_queue.put(("set_gen_param", (name, value)))

    def init_shared_tensors(self):
        height, width = self.resolution["height"], self.resolution["width"]
        out_height, out_width = height, width

        if self.config.get("enable_flow_upscaler", False):
            out_height, out_width = out_height * 2, out_width * 2
        self.out_h, self.out_w = out_height, out_width

        self.input_shared_tensor = SharedTensor(
            (height, width, 3),
            name=self.input_shared_tensor_name,
        )

        # Allocate for the max interpolation factor; only the active count is
        # written each frame (live-adjustable via the shared exp Value).
        output_batch_size = 2**self._max_interp_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, out_height, out_width, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()
        self._prepare_prompt_cache()
        self.load_models()
        self._init_gen_params()
        self.cycle_embeds = []
        self.cycle_index = 0
        if self.config.get("prompt_cycle"):
            # Encode all cycle prompts once now; switching is then instant.
            self.precompute_prompt_cycle()
        else:
            self.update_prompt_embeds(self.process_state["prompt"])
        self.previous_frame = None

        if self.config.get("use_reference_image", False):
            image = cv2.imread(self.config.get("reference_image_path", ""))
            resolution = self.config.get("reference_image_resolution")
            if image is None:
                image = np.zeros(
                    (resolution["height"], resolution["width"], 3), dtype=np.uint8
                )
                print(
                    "Warning: use_reference_image is set to true but no valid reference_image_path is provided."
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resolution["width"], resolution["height"]))
            self.reference_image = Image.fromarray(image)

        target_fps = self.config.get("target_fps", None)
        self.target_base_processing_time = None
        if target_fps is not None:
            target_base_fps = target_fps / (2**self.interpolation_exp)
            self.target_base_processing_time = 1 / target_base_fps

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join()

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def set_reference_image(self, image: np.ndarray | None) -> None:
        """
        Update the reference image on the fly.
        image: numpy uint8 RGB array
        Only valid when use_reference_image is true in config.
        """
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the stream processor config"
            )
        self.command_queue.put(("set_reference_image", image))

    def set_mask(self, mask) -> None:
        """
        Update the mask on the fly.
        mask: numpy uint8 array of shape (h // compression_ratio, w // compression_ratio).
        Only valid when mask_calculation_method is set to manual in config.
        """
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.command_queue.put(("set_mask", mask))

    def set_lip_transfer(self, enabled: bool) -> None:
        self.command_queue.put(("set_lip_transfer", enabled))

    def update_process_state(self) -> None:
        """
        Called by the internal process
        """
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
                    if name == "prompt":
                        self.update_prompt_embeds(value)
                elif cmd == "set_reference_image":
                    image = payload  # numpy uint8 RGB array or None
                    resolution = self.config["reference_image_resolution"]
                    if image is not None:
                        image = cv2.resize(
                            image, (resolution["width"], resolution["height"])
                        )
                        self.reference_image = Image.fromarray(image)
                    else:
                        self.reference_image = Image.fromarray(
                            np.zeros(
                                (resolution["height"], resolution["width"], 3),
                                dtype=np.uint8,
                            )
                        )
                    self.update_controller.reset_cache()

                elif cmd == "set_mask":
                    mask = payload  # numpy uint8 array of shape (h // compression_ratio, w // compression_ratio)
                    mask_tensor = (
                        torch.from_numpy(mask)
                        .unsqueeze(0)
                        .to(self.update_controller.device)
                    )
                    self.update_controller.set_mask(mask_tensor)

                elif cmd == "set_lip_transfer":
                    self.lip_active = payload

                elif cmd == "set_prompt_index":
                    self._apply_prompt_index(payload)

                elif cmd == "set_gen_param":
                    name, value = payload
                    self._apply_gen_param(name, value)

        except Empty:
            pass

    def receive_frame(self):
        """
        Reads frame from input shared memory, converts to RGB float16 GPU tensors.
        """
        frame = self.input_shared_tensor.to_numpy()
        frame_gpu = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame_gpu

    def _current_exp(self) -> int:
        """Active interpolation factor (output = 2**exp frames), from the shared
        Value when live, clamped to the allocated max."""
        if self.interpolation_exp_value is not None:
            return max(
                0, min(int(self.interpolation_exp_value.value), self._max_interp_exp)
            )
        return self.interpolation_exp

    def interpolate_frames(self, frame):
        """
        Takes one new generated frame (torch tensor, RGB, on GPU, float16)
        Interpolates according to the active interpolation factor.
        Batches to [interpolated frames, new frame].
        """
        if self.previous_frame is None:
            self.previous_frame = frame

        exp = self._current_exp()
        if exp == 0:
            frames_out = frame
        else:
            frames = torch.cat([self.previous_frame, frame], dim=0)
            with torch.no_grad():
                for _ in range(exp):
                    B = frames.size(0)
                    prevs = frames[:-1]
                    nexts = frames[1:]
                    mids = self.interpolation_model(
                        torch.cat([prevs, nexts], dim=1), scale=self._rife_scale
                    )
                    H, W = frames.shape[2:]
                    new_frames = torch.empty(
                        2 * B - 1, 3, H, W, device=frames.device, dtype=frames.dtype
                    )
                    new_frames[0::2] = frames
                    new_frames[1::2] = mids
                    frames = new_frames
            frames_out = frames[1:]

        frames_cpu = (
            frames_out.mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

        self.previous_frame = frame

        return frames_cpu[..., ::-1]

    def send_frames(self, frames):
        # If the output buffer is sized for 2x (flow-upscaler capability) but the
        # upscaler is currently OFF, frames come back at base size -> resize up to
        # the output resolution so they fit the buffer.
        if frames.shape[1] != self.out_h or frames.shape[2] != self.out_w:
            frames = np.stack(
                [
                    cv2.resize(np.ascontiguousarray(f), (self.out_w, self.out_h))
                    for f in frames
                ]
            )
        # Write only the active frames into the (max-sized) buffer; the output
        # scheduler reads the same active count from the shared exp Value.
        n = frames.shape[0]
        self.output_batch_shared_tensor.array[:n] = frames

    def sync_fps_and_send(self, prev_time, frames):
        now = time.time()
        processing_time = now - prev_time

        if self.target_base_processing_time is not None:
            sleep_time = max(0, self.target_base_processing_time - processing_time)
            time.sleep(sleep_time)
            now = time.time()

        processing_time = now - prev_time

        self.last_processing_time.value = processing_time
        self.send_frames(frames)
        self.pack_is_ready.value = True
        self.memory_reserved.value = torch.cuda.memory_reserved() // (1024 * 1024)

        if self.logging:
            print(
                f"base fps: {(1 / processing_time):.2f}, interpolated fps: {(1 / processing_time * 2**self._current_exp()):.2f}"
            )
        return now

    def process_frame_with_pipeline(self, frame):
        """
        Takes frame as np uint8 RGB array
        Returns frame as np uint8 RGB array
        """
        input_frame = Image.fromarray(frame)

        reference_list = [input_frame]
        if self.config["use_reference_image"]:
            reference_list.append(self.reference_image)

        out = self.pipe(
            prompt_embeds=self.prompt_embeds,
            image=reference_list,
            height=self.resolution["height"],
            width=self.resolution["width"],
            guidance_scale=1.0,
            num_inference_steps=self.process_state["steps"],
            sigmas=self.custom_sigmas,
            num_images_per_prompt=1,
            generator=torch.Generator(device=self.device).manual_seed(
                self.process_state["seed"]
            ),
            output_type="np",
        )
        out_image = out.images[0]
        out_image = out_image * 255
        out_image = out_image.astype(np.uint8)
        return out_image

    def convert_np_to_torch(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame

    def process_main(self):
        self.process_init()
        prev_time = time.time()
        while self.running.value:
            # Commands are processed outside the try so a bad live setting can
            # always be undone even if it makes frames error.
            self.update_process_state()
            try:
                original_frame = self.input_shared_tensor.to_numpy()
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                frame = self.process_frame_with_pipeline(original_frame)
                if self.lip_processor is not None and self.lip_active:
                    # Note: we are getting the latest input frame again after flux processing to reduce latency.
                    original_frame = self.input_shared_tensor.to_numpy()
                    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                    frame = self.lip_processor.process(frame, original_frame)
                frame = self.convert_np_to_torch(frame)
                frames = self.interpolate_frames(frame)
                prev_time = self.sync_fps_and_send(prev_time, frames)
            except Exception as exc:  # noqa: BLE001
                # Never let one bad frame/setting freeze or kill the stream; log
                # and keep going (the last good frame stays on screen).
                print(f"[FluxRT] frame skipped due to error: {exc}", flush=True)
                time.sleep(0.05)
