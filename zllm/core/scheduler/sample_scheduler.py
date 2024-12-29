
from typing import List
from zllm.core.datatypes.scheduler_output import SchedulerOutputs
from zllm.config.config import CacheConfig, ModelConfig, ParallelConfig
from zllm.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from zllm.core.scheduler.base_scheduler import BaseScheduler

class SampleScheduler(BaseScheduler):

    def __init__(
            self, 
            model_config: ModelConfig, 
            cache_config: CacheConfig, 
            parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, cache_config, parallel_config)


    def _schedule(self) -> SchedulerOutputs:

        running: List[Sequence] = []
        ingored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        while self.running:

            seq = self.running.pop(0)
            running.append(seq)

            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq=seq,
                )
            )
        
        while self.waiting:
            seq = self.waiting.pop(0)
            running.append(seq)

            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq=seq, prompt_chunk_len=seq.get_prompt_len()
                )
            )

        self.running = running
        return SchedulerOutputs(
            self._iteration_id,
            ignored_seq_ids=ingored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )