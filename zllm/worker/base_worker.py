

import os
from pathlib import Path
import random
import time
from typing import List, Optional

import numpy as np
import torch
from zllm.core.datatypes.sequence import SamplerOutputs
from zllm.core.datatypes.step_inputs import StepInputs
from zllm.model_executor.parallel_utils.tensor_parallel.random import model_parallel_cuda_manual_seed
from zllm.config.config import CacheConfig, ModelArgs, ParallelConfig, SystemConfig
from zllm.core.datatypes.scheduler_output import SchedulerOutputs
from zllm.core.datatypes.sequence import Sequence
from zllm.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
from zllm.model_executor.model_runner import ModelRunner
from zllm.model_executor.parallel_utils.parallel_state import get_pipeline_model_parallel_rank, get_tensor_model_parallel_rank, initialize_model_parallel, model_parallel_is_initialized
from zllm.utils.threading_utils import exit_on_error, synchronized
from zllm.worker.llama31 import Llama, Transformer
from zllm.logger import init_logger
from zllm.core.datatypes.comm_info import CommInfo

logger = init_logger(__name__)

seed = 7

class BaseWroker:
    def __init__(
        self, 
        config: SystemConfig,
        local_rank: int,
        rank: int,
        comm_info: CommInfo,
    ):
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.comm_info = comm_info

        self.cache_engine = None
        self.gpu_cache = None

        self.seq_manager = None     
        
    def get_model(self, model_config: ModelArgs, ckpt_dir: str):

        assert os.path.isdir(ckpt_dir), f"{ckpt_dir} is not a valid directory"

        local_rank = 0
        torch.cuda.set_device(local_rank)
        torch.manual_seed(seed)

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) == 1, f"{ckpt_dir} should contain exactly one checkpoint"

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.bfloat16)
        else:
            torch.set_default_tensor_type(torch.float16)
        model = Transformer(model_config)
        model.load_state_dict(torch.load(checkpoints[0], ))

    @torch.inference_mode()
    @synchronized
    def init_model(self):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["KINETO_LOG_LEVEL"] = "5"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        logger.info(f"Worker {self.rank} is using device {self.local_rank}")
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        _init_distributed_environment(
            self.config.parallel_config,
            self.rank,
            self.comm_info.distributed_init_method,
        )
        
        self.tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_model_parallel_rank = get_pipeline_model_parallel_rank()

        self.is_tensor_parallel_rank_zero = self.tensor_model_parallel_rank == 0
        self.is_first_pipeline_stage = self.pipeline_model_parallel_rank == 0
        self.is_last_pipeline_stage = (
            self.pipeline_model_parallel_rank
            == self.config.parallel_config.pipeline_parallel_size - 1
        )

        logger.info(
            f"Initializing worker {self.rank} on device {self.device}, "
            f"tensor parallel rank {self.tensor_model_parallel_rank} "
            f"and pipeline parallel rank {self.pipeline_model_parallel_rank}."
        )

        # Initialize the model.
        set_random_seed(self.config.model_config.seed)
        self.model_runner = ModelRunner(
            self.config,
            self.device,
            self.rank,
        )
        logger.info(f"Model initialized on worker {self.rank}.")

    @synchronized
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        torch.cuda.set_device(self.device)

        self.config.cache_config = cache_config

        self.model_runner.init_kv_cache(cache_config.num_gpu_blocks)

        self.seq_manager = WorkerSequenceManager(
            self.config,   
        )

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)


    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)

        self.worker_ready_event.set()

        while True:
            step_inputs = self.enqueue_socket.recv_pyobj()

            for new_seq in step_inputs.new_seqs:
                self.seq_manager.add_seq(new_seq)

            output = self.execute_model(step_inputs.scheduler_outputs)

            if not self.is_tensor_parallel_rank_zero:
                continue

            self.output_socket.send_pyobj(output)

    def sync_execute(self, step_inputs: StepInputs):
        # todo: 删除这个方法，用 _execution_loop 从 zmp 里取出调度结果
        for new_seq in step_inputs.new_seqs:
            self.seq_manager.add_seq(new_seq)
        
        output = self.execute_model(step_inputs.scheduler_outputs)

        return output


    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Optional[SamplerOutputs]:
        torch.cuda.synchronize()
        # batch_stage_start_time = time.monotonic()

        _, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)

        sampler_outputs = self.model_runner.run(
            seq_metadata_list,
        )

        self.on_step_completed(scheduler_outputs, sampler_outputs)

        torch.cuda.synchronize()

        # batch_stage_end_time = time.monotonic()

        # self.metrics_store.on_batch_stage_end(
        #     seq_metadata_list,
        #     scheduler_outputs,
        #     self.tensor_model_parallel_rank,
        #     self.pipeline_model_parallel_rank,
        #     batch_stage_start_time,
        #     batch_stage_end_time,
        # )

        return sampler_outputs



def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)
