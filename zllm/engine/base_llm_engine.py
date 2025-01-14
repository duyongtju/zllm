

import copy
from functools import partial
import json
from multiprocessing import Queue
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import ray

from zllm.core.datatypes.comm_info import CommInfo
from zllm.core.scheduler.scheduler_registry import SchedulerRegistry
from zllm.engine.multiproc_utils import ProcessWorkerWrapper, ResultHandler, WorkerMonitor
from zllm.engine.ray_utils import RayWorker, initialize_cluster
from zllm.logger import init_logger
from zllm.utils import Counter, get_ip, unset_cuda_visible_devices
from zllm.config.config import SystemConfig, CacheConfig, ParallelConfig, ModelConfig
from zllm.core.datatypes.request_output import RequestOutput
from zllm.core.datatypes.sampling_params import SamplingParams
from zllm.core.datatypes.scheduler_output import SchedulerOutputs
from zllm.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from zllm.core.datatypes.step_inputs import StepInputs
from zllm.core.scheduler.sample_scheduler import SampleScheduler
from zllm.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from zllm.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from zllm.transformer_utils.tokenizer import get_tokenizer
from zllm.utils.threading_utils import synchronized
from zllm.worker.base_worker import BaseWorker

_MAX_WORKER_CONCURRENCY = 1

ModelParallelRank = Tuple[int, int]

logger = init_logger(__name__)

class BaseLLMEngine:

    def __init__(
        self,
        config: SystemConfig,
    ) -> None:
        self.config = config

        # # todo: 抽象 worker 层，增加 init_work 方法
        # set_up(0, 1)
        # worker = Llama.build(
        #     config,
        #     ckpt_dir=ckpt_dir,
        #     tokenizer_path=tokenizer_path,
        #     max_seq_len=config.model_config.max_seq_len,
        #     max_batch_size=config.model_config.max_batch_size,
        # )

        # self.worker = worker
        # self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = worker.tokenizer

        self.tokenizer = get_tokenizer(
            config.model_config.model,
            trust_remote_code=config.model_config.trust_remote_code,
            revision=config.model_config.revision,
        )

        self.seq_manager = EngineSequenceManager(
            self.tokenizer, None,
        )
        self.seq_counter = Counter()

        self.worker_map: Dict[ModelParallelRank, int] = {}

        # Initialize the cluster.
        # initialize_cluster()

        # Create the parallel GPU workers.
        self._init_workers()

        self._init_model()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Initialize the worker map.
        self._init_worker_map()
        
        # self.mark_initial_memory_profiling_done()

        self.scheduler = SchedulerRegistry.get(
            config.scheduler_config.get_type(),
            config.model_config,
            config.scheduler_config,
            config.cache_config,
            config.parallel_config,
        )

        self.new_seqs: List[Sequence] = []

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from zllm.worker.base_worker import (
            BaseWorker,  # pylint: disable=import-outside-toplevel
        )

        return BaseWorker

    def _init_workers(self):
        self._init_worker_multiproc()


    def _init_worker_multiproc(self):
        resource_mapping = self.config.replica_config.get_resource_mapping(
            self.config.parallel_config.world_size
        )
        logger.info(f"Starting workers with resource mapping: {resource_mapping}")

        self.workers: List[ProcessWorkerWrapper] = []
        result_handler = ResultHandler()

        config = copy.deepcopy(self.config)
        worker_impl = self._get_worker_impl()
        
        driver_ip = get_ip()
        self.comm_info = CommInfo(driver_ip)

        self.workers = [
            ProcessWorkerWrapper(
                result_handler,
                partial(worker_impl,
                        config,
                        local_rank=rank,
                        rank=0,
                        comm_info=self.comm_info,
                        )
            ) for rank, (node_ip, _) in enumerate(resource_mapping) ]

        worker_monitor = WorkerMonitor(self.workers, result_handler)
        result_handler.start()
        worker_monitor.start()


    def _init_workers_ray(self, **ray_remote_kwargs):
        resource_mapping = self.config.replica_config.get_resource_mapping(
            self.config.parallel_config.world_size
        )
        logger.info(f"Starting workers with resource mapping: {resource_mapping}")

        self.workers: List[RayWorker] = []

        unset_cuda_visible_devices()
        
        driver_ip = None
        for rank, (node_ip, _) in enumerate(resource_mapping):
            worker_class = ray.remote(
                num_cpus=1,
                # num_gpus=1, # we don't use ray for managing GPUs
                **ray_remote_kwargs,
            )(RayWorker)

            if node_ip:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                    resources={
                        node_ip: 0.01,
                    },
                )
            else:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                )

            if rank == 0:
                if node_ip:
                    # remove node: prefix
                    driver_ip = node_ip.split(":")[1]
                else:
                    driver_ip = get_ip()

            worker = worker_class.remote(self.config.model_config.trust_remote_code)

            self.workers.append(worker)

        self.comm_info = CommInfo(driver_ip)

        # Initialize torch distributed process group for the workers.
        config = copy.deepcopy(self.config)
        worker_impl = self._get_worker_impl()

        for rank, worker in enumerate(self.workers):
            local_rank = resource_mapping[rank][1]
            promise = worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: worker_impl(
                    config,
                    local_rank,
                    rank,
                    self.comm_info,
                )
            )
            ray.get(promise)

        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _init_model(self) -> None:
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # # Get the maximum number of blocks that can be allocated on GPU.
        # num_gpu_blocks_across_workers = self._run_workers(
        #     "profile_num_available_blocks",
        #     get_all_outputs=True,
        #     block_size=self.config.cache_config.block_size,
        #     gpu_memory_utilization=self.config.worker_config.gpu_memory_utilization,
        # )

        # # Since we use a shared centralized controller, we take the minimum
        # # number of blocks across all workers to make sure all the memory
        # # operators can be applied to all workers.
        # num_gpu_blocks = min(num_gpu_blocks_across_workers)
        # # FIXME(woosuk): Change to debug log.
        # logger.info(f"# GPU blocks: {num_gpu_blocks}")

        # if num_gpu_blocks <= 0:
        #     raise ValueError(
        #         "No available memory for the cache blocks. "
        #         "Try increasing `gpu_memory_utilization` when "
        #         "initializing the engine."
        #     )
        # max_blocks_per_request = math.ceil(
        #     self.config.model_config.max_model_len / self.config.cache_config.block_size
        # )
        # if num_gpu_blocks < max_blocks_per_request:
        #     raise ValueError(
        #         f"Not enough available memory to schedule a request will maximum allowed length {self.config.model_config.max_model_len}. "
        #         f"Need {max_blocks_per_request}, available {num_gpu_blocks} gpu blocks. "
        #         f"Try decreasing `max_batch_size`, `max_model_len`."
        #     )
        # self.config.cache_config.num_gpu_blocks = num_gpu_blocks

        # Initialize the cache.
        self._run_workers(
            "init_cache_engine",
            cache_config=self.config.cache_config,
            get_all_outputs=True,
        )

    def _init_worker_map(self) -> None:
        model_parallel_ranks = self._run_workers(
            "get_model_parallel_ranks",
            get_all_outputs=True,
        )

        self.worker_map = {mp_rank: i for i, mp_rank in enumerate(model_parallel_ranks)}

    def sync_execute_model(self, *args, **kwargs)-> SamplerOutputs:
        sampler_outputs = self._run_workers(
            "sync_execute",
            get_all_outputs=True,
            *args,
            **kwargs,
        )
        return sampler_outputs[0]

    def get_model_config(self) -> ModelConfig:
        return self.config.model_config

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
        sampler_outputs = self.sync_execute_model(step_inputs=StepInputs(
            scheduler_outputs, self._get_new_seqs()
        ))

        return self._on_step_complete(
            scheduler_outputs,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
        )

    def _run_workers(
        self, 
        method:str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    )-> Any:
        return self._run_workers_multiproces(
            method,
            *args,
            get_all_outputs=get_all_outputs,
            ignore_output=ignore_output,
            **kwargs
        )


    def _run_workers_multiproces(
        self, 
        method:str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    )-> Any:
        
        all_outputs: List[Any] = []
        for worker in self.workers:
            output = worker.execute_method(method, *args, **kwargs)
            all_outputs.append(output)
        
        if ignore_output:
            return
        
        results = [ output.get() for output in all_outputs]

        if get_all_outputs:
            return results

        # Make sure all workers have the same results.
        result = results[0]
        for other_result in all_outputs[1:]:
            assert result == other_result
        return result


    def _run_workers_ray(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = partial(worker.execute_method.remote, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if ignore_output:
            return

        while True:
            try:
                all_outputs = ray.get(all_outputs, timeout=0)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0)
                continue

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def _run_worker_ray(
        self,
        model_parallel_rank: ModelParallelRank,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        worker = self.workers[self.worker_map[model_parallel_rank]]
        executor = partial(worker.execute_method.remote, method)

        output = executor(*args, **kwargs)

        while True:
            try:
                output = ray.get(output, timeout=0)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0)
                continue

        return output

