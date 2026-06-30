from multiprocessing import Process, Value
from fluxrt.utils.shared_tensor import SharedTensor
import time


class OutputSchedulerSubprocess:
    def __init__(
        self,
        config: dict,
        output_batch_shared_tensor_name: str,
        output_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
        frame_written=None,
        interpolation_exp_value=None,
    ):
        self.config = config
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.output_shared_tensor_name = output_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time
        self.frame_written = frame_written

        self.running = Value("b", False)
        self.process = None

        self.interpolation_exp = self.config.get("interpolation_exp", 1)
        # Buffer is allocated for the max factor; active count comes from the
        # shared Value each pack so the interpolation factor is live-adjustable.
        self.interpolation_exp_value = interpolation_exp_value
        self._max_interp_exp = max(self.interpolation_exp, 3)
        self.max_batch_size = 2**self._max_interp_exp

    def _active_batch_size(self) -> int:
        if self.interpolation_exp_value is not None:
            exp = max(
                0, min(int(self.interpolation_exp_value.value), self._max_interp_exp)
            )
        else:
            exp = self.interpolation_exp
        return 2**exp

    def start(self) -> None:
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self) -> None:
        self.running.value = False
        if self.process:
            self.process.join()

    def process_init(self) -> None:
        """
        Called by the internal process
        """
        height = self.config["resolution"]["height"]
        width = self.config["resolution"]["width"]

        if self.config.get("enable_flow_upscaler", False):
            height, width = height * 2, width * 2

        self.output_batch_shared_tensor = SharedTensor(
            (self.max_batch_size, height, width, 3),
            name=self.output_batch_shared_tensor_name,
        )
        self.output_shared_tensor = SharedTensor(
            (height, width, 3),
            name=self.output_shared_tensor_name,
        )

    def process_main(self) -> None:
        self.process_init()

        while self.running.value:
            if not self.pack_is_ready.value:
                continue

            batch_size = self._active_batch_size()
            proc_time = min(max(self.last_processing_time.value, 0.001), 1.0)
            sleep_interval = proc_time / batch_size

            for i in range(batch_size):
                self.output_shared_tensor.copy_from(
                    self.output_batch_shared_tensor.array[i]
                )
                if self.frame_written is not None and not self.frame_written.value:
                    self.frame_written.value = True
                if i < batch_size - 1:
                    time.sleep(sleep_interval)

            self.pack_is_ready.value = False
