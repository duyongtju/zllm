"""  model runner """

from typing import List, Optional, Tuple
import torch

from zllm.attention.base_attention_wrapper import BaseAttentionWrapper
from zllm.config.config import SystemConfig
from zllm.core.datatypes.sequence import SequenceMetadata
from zllm.model_executor.layers.sampler import Sampler
from zllm.model_executor.model_loader import get_model
from zllm.model_executor.utils import pad_to_alignment


class ModelRunner:

    def __init__(
        self,
        config: SystemConfig,
        device: torch.device,
        rank: int,
    ):
        self.config = config
        self.device = device
        self.rank = rank    

        self.attention_backend_wrapper: BaseAttentionWrapper = BaseAttentionWrapper(
            self.config.model_config,
            self.config.cache_config,
            self.config.parallel_config,
            self.device,
        )

        self.model = get_model(self.config.model_config)

        self.sampler: Optional[Sampler] = None
        self.sampler = Sampler(
            self.model.lm_head.weight, self.model.config.vocab_size
        )
    

    def init_kv_cache(self, num_gpu_blocks: int):
        self.attention_backend_wrapper.init_gpu_cache(num_gpu_blocks)

    def _prepare_inputs(
        self,
        seq_metadata_list: List[SequenceMetadata]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens: List[int] = []
        input_positions: List[int] = []

        # sampler 需要知道每个 sequence 的 prompt chunk size
        cur_prompt_chunk_lens: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            cur_prompt_chunk_tokens = (
                seq_metadata.seq.get_next_prompt_chunk_token_ids(prompt_chunk_len)
            )
            cur_prompt_chunk_len = len(cur_prompt_chunk_tokens)
            cur_prompt_chunk_lens.append(cur_prompt_chunk_len)
            processed_prompt_len = (
                seq_metadata.seq.get_num_prompt_tokens_stage_processed()
            )
            cur_total_len = processed_prompt_len + cur_prompt_chunk_len

            input_tokens.extend(cur_prompt_chunk_tokens)
            input_positions.extend(range(processed_prompt_len, cur_total_len))
        
        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            generation_token = seq_metadata.seq.get_last_token_id()
            input_tokens.append(generation_token)

            context_len = seq_metadata.seq.get_len()
            position = context_len - 1
            input_positions.append(position)

        #todo: need pad to be a mutiple of 8?
        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        # input_tokens = pad_to_alignment(input_tokens, multiple_of=8)
        # input_positions = pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )

        return tokens_tensor, positions_tensor
    
    @torch.inference_mode()
    def profill_num_available_blocks(
        self, 
        block_size: int,
        gpu_memory_utilization: float,
    )-> Tuple[int, int]:
        raise Exception('not implemented')

    def run(
        self, 
        seq_metadata_list: List[SequenceMetadata],
    ) -> torch.Tensor:
        
        input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)

        self.attention_backend_wrapper.begin_forward(seq_metadata_list)

        output = self.model(
            hidden_states=input_tokens,
            positions=input_positions,
            attention_backend_wrapper = self.attention_backend_wrapper,
        )

        output = self.sampler(output, seq_metadata_list)

        return output