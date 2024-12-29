from zllm.config import SchedulerType

from zllm.core.scheduler.vllm_scheduler import VLLMScheduler
from zllm.utils.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


SchedulerRegistry.register(SchedulerType.VLLM, VLLMScheduler)
# SchedulerRegistry.register(SchedulerType.ORCA, OrcaScheduler)
# SchedulerRegistry.register(SchedulerType.FASTER_TRANSFORMER, FasterTransformerScheduler)
# SchedulerRegistry.register(SchedulerType.SARATHI, SarathiScheduler)
# SchedulerRegistry.register(SchedulerType.SIMPLE_CHUNKING, SimpleChunkingScheduler)
