import multiprocessing
from multiprocessing import Value
from fluxrt.utils import SharedTensor
from fluxrt.stream_processor.model_inference_subprocess import (
    ModelInferenceSubprocess,
)
from fluxrt.stream_processor.output_scheduler_subprocess import (
    OutputSchedulerSubprocess,
)
import json
import numpy as np


class StreamProcessor:
    def __init__(self, config_path: str):
        self.config = self.parse_config(config_path)
        self.resolution = self.config["resolution"]
        cfg_exp = self.config.get("interpolation_exp", 1)
        # Allocate the output batch for the max interpolation factor so the
        # factor can be changed live (active count comes from a shared Value).
        self._max_interp_exp = max(cfg_exp, 3)
        output_batch_size = 2 ** self._max_interp_exp

        height, width = self.resolution["height"], self.resolution["width"]
        out_height, out_width = height, width
        if self.config.get("enable_flow_upscaler", False):
            out_height, out_width = out_height * 2, out_width * 2

        self.input_shared_tensor = SharedTensor((height, width, 3), create=True)
        self.output_shared_tensor = SharedTensor(
            (out_height, out_width, 3), create=True
        )
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, out_height, out_width, 3), create=True
        )

        self.out_resolution = {"height": out_height, "width": out_width}

        multiprocessing.set_start_method("spawn", force=True)

        self.pack_is_ready = Value("b", False)
        self.last_processing_time = Value("f", 0.0)
        self.frame_written = Value("b", False)
        self.interpolation_exp_value = Value("i", cfg_exp)

        self.model_inference_subprocess = ModelInferenceSubprocess(
            self.config,
            self.input_shared_tensor.name,
            self.output_batch_shared_tensor.name,
            self.pack_is_ready,
            self.last_processing_time,
            self.interpolation_exp_value,
        )

        self.output_scheduler_subprocess = OutputSchedulerSubprocess(
            self.config,
            self.output_batch_shared_tensor.name,
            self.output_shared_tensor.name,
            self.pack_is_ready,
            self.last_processing_time,
            self.frame_written,
            self.interpolation_exp_value,
        )

    def parse_config(self, config_path: str) -> dict:
        with open(config_path, "r") as file:
            return json.load(file)

    def start(self) -> None:
        self.model_inference_subprocess.start()
        self.output_scheduler_subprocess.start()

    def get_input_tensor(self) -> SharedTensor:
        return self.input_shared_tensor

    def get_output_tensor(self) -> SharedTensor:
        return self.output_shared_tensor

    def stop(self) -> None:
        self.model_inference_subprocess.stop()
        self.output_scheduler_subprocess.stop()
        self.input_shared_tensor.close_and_unlink()
        self.output_shared_tensor.close_and_unlink()
        self.output_batch_shared_tensor.close_and_unlink()

    def set_prompt(self, prompt: str) -> None:
        self.model_inference_subprocess.set_param(name="prompt", value=prompt)

    def set_prompt_index(self, idx: int) -> None:
        """Switch to a pre-encoded prompt from config['prompt_cycle'] (instant)."""
        self.model_inference_subprocess.set_prompt_index(idx)

    def set_steps(self, steps: int) -> None:
        self.model_inference_subprocess.set_param(name="steps", value=steps)

    def set_seed(self, seed: int) -> None:
        self.model_inference_subprocess.set_param(name="seed", value=seed)

    def set_param(self, name: str, value) -> None:
        self.model_inference_subprocess.set_param(name=name, value=value)

    def set_gen_param(self, name: str, value) -> None:
        """Live advanced generation control: steps, seed, shift,
        use_dynamic_shifting, stochastic_sampling, time_shift_type, sigmas,
        rife_scale, interpolation_exp. Applies immediately."""
        if name == "interpolation_exp":
            # Shared Value read each pack by both the inference and scheduler
            # processes; clamped to the buffer's allocated max.
            self.interpolation_exp_value.value = max(
                0, min(int(value), self._max_interp_exp)
            )
            return
        self.model_inference_subprocess.set_gen_param(name=name, value=value)

    def set_reference_image(self, image: np.ndarray | None) -> None:
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the config"
            )
        self.model_inference_subprocess.set_reference_image(image)

    def set_mask(self, mask: np.ndarray) -> None:
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.model_inference_subprocess.set_mask(mask)

    def get_resolution(self) -> dict:
        return self.resolution

    def get_out_resolution(self) -> dict:
        return self.out_resolution

    def is_ready(self) -> bool:
        return bool(self.frame_written.value)

    def get_input_shared_tensor_name(self) -> str:
        return self.input_shared_tensor.name

    def get_output_shared_tensor_name(self) -> str:
        return self.output_shared_tensor.name

    def get_last_processing_time(self) -> float:
        with self.last_processing_time.get_lock():
            return self.last_processing_time.value

    def set_lip_transfer(self, enabled: bool) -> None:
        self.model_inference_subprocess.set_lip_transfer(enabled)

    def enable_quantization(self) -> None:
        self.model_inference_subprocess.enable_quantization()

    def get_reserved_memory(self) -> int:
        """Returns reserved GPU memory in MB."""
        return self.model_inference_subprocess.memory_reserved.value
