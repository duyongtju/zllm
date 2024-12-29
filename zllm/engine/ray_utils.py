
from typing import Optional
from zllm.logger import init_logger
from zllm.utils import unset_cuda_visible_devices

logger = init_logger(__name__)

try:
    import ray

    class RayWorker:

        def __init__(self, init_cached_hf_modules: bool = False) -> None:
            if init_cached_hf_modules:
                # pylint: disable=import-outside-toplevel
                # todo: init_hf_modules 什么作用？
                from transformers.dynamic_module_utils import init_hf_modules

                init_hf_modules()

            unset_cuda_visible_devices()
            self.worker = None

        def init_worker(self, worker_init_fn):
            self.worker = worker_init_fn()

        def __getattr__(self, name):
            return getattr(self.worker, name)
        
        def execute_method(self, method, *args, **kwargs):
            executor = getattr(self, method)
            return executor(*args, **kwargs)

except ImportError as e:
    logger.warning(
        f"Failed to import Ray with {e!r}. "
        "For distributed inference, please install Ray with "
        "`pip install ray pandas pyarrow`."
    )
    ray = None
    RayWorker = None  # pylint: disable=invalid-name

def initialize_cluster(
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster probably with Ray.

    Args:
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed " "serving."
        )
    # Connect to a ray cluster.
    # 调试时 local_mode=True ref: https://stackoverflow.com/questions/71860615/how-to-disable-ray-for-debugging
    ray.init(address=ray_address, ignore_reinit_error=True, local_mode=True)
