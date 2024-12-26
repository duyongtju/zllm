import time
from typing import List

from zllm.config.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig, VllmSchedulerConfig
from zllm.core.datatypes.scheduler_output import SchedulerOutputs
from zllm.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from zllm.core.scheduler.base_scheduler import BaseScheduler
from zllm.logger import init_logger




logger = init_logger(__name__)

class VLLMScheduler(BaseScheduler):

    def __init__(
        self, 
        model_config: ModelConfig, 
        scheduler_config: VllmSchedulerConfig,
        cache_config: CacheConfig, 
        parallel_config: ParallelConfig
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.max_num_batched_tokens = self.scheduler_config.get_max_num_batched_tokens(
            self.model_config.max_model_len
        )
        self.prompt_limit = self.max_num_batched_tokens


    def _schedule(self) -> SchedulerOutputs:
                # Fix the current time.
        now = time.monotonic()

        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_batched_tokens = 0
        
        while self.waiting:
            seq = self.waiting[0]

            if seq.arrival_time > now:
                break

            num_prompt_tokens = seq.get_len()
            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq)
                continue

            if not self.block_manager.can_allocate(seq):
                break

            if num_batched_tokens + num_prompt_tokens > self.max_num_batched_tokens:
                break

            if len(self.running) + 1 > self.scheduler_config.max_num_seqs:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += num_prompt_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )
            self.running.append(seq)
        
        if scheduled_seq_metadata_list or ignored_seq_ids:
            return SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )
    

        self.running = self.policy.sort_by_priority(now, self.running)

        running: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused(): # todo: 这个状态是什么意思？
                # The sequence group is already in the RUNNING state.
                running.append(seq)
                continue

            assert seq.prompt_stage_processing_finished

            while not self.block_manager.can_append_slot():
                if self.running:
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                self._append_slot(seq)
                running.append(seq)
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=[],
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
