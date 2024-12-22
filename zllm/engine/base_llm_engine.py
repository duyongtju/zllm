

import copy
import json
from multiprocessing import Queue
from pathlib import Path
import time
from typing import List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


from llama.generation import Llama
from llama.generation import set_up

from zllm.config.config import SystemConfig, CacheConfig, ParallelConfig
from zllm.core.datatypes.request_output import RequestOutput
from zllm.core.datatypes.sampling_params import SamplingParams
from zllm.core.datatypes.scheduler_output import SchedulerOutputs
from zllm.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from zllm.core.datatypes.step_inputs import StepInputs
from zllm.core.scheduler.sample_scheduler import SampleScheduler
from zllm.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from zllm.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from zllm.utils.threading_utils import synchronized
from zllm.worker.llama31 import ModelArgs


class BaseLLMEngine:

    def __init__(self,
                 config: SystemConfig,
                 ckpt_dir: str,
                 tokenizer_path: str,
    ) -> None:
        self.config = config

        # todo: 抽象 worker 层，增加 init_work 方法
        set_up(0, 1)
        worker = Llama.build(
            config,
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=config.model_config.max_seq_len,
            max_batch_size=config.model_config.max_batch_size,
        )

        self.worker = worker
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = worker.tokenizer

        self.seq_manager = EngineSequenceManager(
            self.tokenizer, None,
        )
        
        self.scheduler = SampleScheduler(
            model_config=config.model_config,
            cache_config=config.cache_config,
            parallel_config=config.parallel_config,
        )

        self.new_seqs: List[Sequence] = []

        self.worker.init_gpu_cache()


    def add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            seq_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()

        if not seq_id:
            seq_id = str(next(self.seq_counter))

        if prompt_token_ids is None:
            assert prompt is not None
            # prompt_token_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.config.cache_config.block_size
        eos_token_id = self.tokenizer.eos_token_id

        seq = Sequence(
            seq_id,
            prompt,
            prompt_token_ids,
            block_size,
            eos_token_id,
            arrival_time,
            sampling_params,
        )
        # Add the sequence to the scheduler.
        self.seq_manager.add_seq(seq)
        # we create a copy of the seq so that the workers
        # receive an unmodified version of the seq
        # which is unaffected by the engine's actions
        self._append_new_seq(copy.deepcopy(seq))
        self.scheduler.add_seq(seq)

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()


    @synchronized
    def _append_new_seq(self, seq: Sequence):
        self.new_seqs.append(seq)

    @synchronized
    def _get_new_seqs(
        self,
    ) -> List[Sequence]:
        new_seqs = self.new_seqs
        self.new_seqs = []
        return new_seqs
        

    def _on_step_complete(
        self,
        scheduler_outputs: SchedulerOutputs,
        ignored_seqs: List[SequenceMetadata],
        seq_metadata_list: List[SequenceMetadata],
        sampler_outputs: Optional[SamplerOutputs],
        start_time: float,
    ) -> List[RequestOutput]:
        # todo 重写这段逻辑
        self.seq_manager.on_step_completed(
            scheduler_outputs,
            sampler_outputs,
        )
        self.scheduler.on_step_completed()

        all_request_outputs = self.seq_manager.generate_request_outputs(
            ignored_seqs, seq_metadata_list
        )
        return all_request_outputs


    def step(self) -> List[RequestOutput]:
        start_time = time.perf_counter()

        scheduler_outputs = self.scheduler.schedule()

        if scheduler_outputs.is_empty():
            return []

        ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)

        # todo: 改写这段逻辑
        sampler_outputs = self.worker.execute_model(StepInputs(
            scheduler_outputs, self._get_new_seqs()
        ))

        return self._on_step_complete(
            scheduler_outputs,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
        )


