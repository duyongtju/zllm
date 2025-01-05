
from abc import abstractmethod
import random
from typing import List, Mapping, Optional, Tuple
from torch import nn
import torch
import math


import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
import flashinfer


from zllm.config.config import CacheConfig, ModelConfig, ParallelConfig
from zllm.core.datatypes.sequence import Sequence, SequenceMetadata

class BaseAttentionWrapper:
    def __init__(
        self, 
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parall_config: ParallelConfig,
        device: torch.device,
    ):
        # super().__init__(args)
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config

        self.device = device, 
        self.num_q_heads = model_config.get_num_q_heads(parall_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parall_config)
        self.head_dim = model_config.get_head_size()
        self.dtype = model_config.dtype
        self.block_size = cache_config.block_size
        self.num_layers = model_config.get_num_layers(parall_config)
        self.num_gpu_blocks: int = cache_config.num_gpu_blocks

        self.kv_cache = None
        self.flash_attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128*1024*1024, dtype=torch.uint8, device="cuda")
        )
        
    def init_gpu_cache(
        self, 
        num_gpu_blocks: int,
    ) -> None:

        self.num_gpu_blocks = num_gpu_blocks

        self.layered_kv_cahce = []
        for _ in range(self.num_layers):
            kv_cache = self.get_cache_block(
                self.num_gpu_blocks, dtype=self.dtype, device="cuda",
            )
            self.layered_kv_cahce.append(kv_cache)

    def get_cache_block(self, num_blocks: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        print((
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            kwargs,
        ))
        return (
            torch.randn(
                num_blocks, 
                self.block_size, 
                self.num_kv_heads, 
                self.head_dim,
                **kwargs
            ), 
            torch.randn(
                num_blocks, 
                self.block_size, 
                self.num_kv_heads, 
                self.head_dim,
                **kwargs
            )
        )

    def begin_forward(self, seq_metadata_list: List[SequenceMetadata]):
        # prepare qo_indptr for example [0, 33, 44, 55, 66, 77, 88, 100]
        # paged_kv_indices [ 1, 2, 3, 4, 5 ]
        # paged_kv_indptr [ 0, 2, 5, ...] seq_1 page: 1 2; seq_2 page 3 4 5  
        # paged_kv_last_page_len [1, 7, 14, 4, 3, 1, 16]  1 <= paged_kv_last_page_len <= page_size
        qo_indptr = [0]        
        kv_page_indices = []
        kv_page_indptr = [0]
        kv_page_last_page_len = []

        self.block_tables = {}

        self.contains_decode = False
        self.contains_encode = False

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt: # decode phase
                continue
            
            self.contains_decode = True

            context_len = seq_metadata.seq.get_len()

            qo_indptr.append(qo_indptr[-1]+1)

            num_blocks_in_use = (context_len + self.block_size -1) // self.block_size

            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1]+num_blocks_in_use)
            kv_page_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

            self.block_tables[seq_metadata.seq.seq_id] = seq_metadata.block_table
        
        for seq_metadata in seq_metadata_list:                
            if not seq_metadata.is_prompt: # prefill phase
                continue
            
            self.contains_encode = True

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = (
                seq_metadata.seq.get_num_prompt_tokens_stage_processed()
            )
            current_total_len = processed_prompt_len + prompt_chunk_len

            qo_indptr.append(qo_indptr[-1] + prompt_chunk_len)
            num_blocks_in_use = (
                current_total_len+self.block_size -1
            ) // self.block_size

            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(
                kv_page_indptr[-1] + num_blocks_in_use
            )
            kv_page_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

            self.block_tables[seq_metadata.seq.seq_id] = seq_metadata.block_table
        
        qo_indptr = self.to_int_tensor(qo_indptr)
        kv_page_indptr = self.to_int_tensor(kv_page_indptr)
        kv_page_indices = self.to_int_tensor(kv_page_indices)
        kv_page_last_page_len = self.to_int_tensor(kv_page_last_page_len)

        # print(f"qo_indptr {qo_indptr} \npaged_kv_indptr {kv_page_indptr}")
        # print(f"paged_kv_indices {kv_page_indices} \npaged_kv_last_page_len {kv_page_last_page_len}")
        # print(f"self.block_tables {self.block_tables}")

        self.flash_attn_wrapper.plan(
            qo_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_page_last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            causal=True
        )

        self.append_qo_indptr_tensor = qo_indptr
        self.append_kv_page_indices_tensor = kv_page_indices
        self.append_kv_page_indptr_tensor = kv_page_indptr
        self.append_kv_last_page_len_tensor = kv_page_last_page_len

    def end_forward(self):
        self.flash_attn_wrapper.end_forward()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache_idx: int,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        
        query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
        key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
        value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
        
        # if layer_cache_idx == 0:
        #     print(f"layer {layer_cache_idx} \n"+"="*20)
        #     print(f"query {query.shape}")
        #     print(f"key {key.shape}")
        #     print(f"value {value.shape}")
        #     print(f"layered_kv_cahce[{layer_cache_idx}][0]: {self.layered_kv_cahce[layer_cache_idx][0].shape}")
        #     print(f"self.append_qo_indptr_tensor {self.append_qo_indptr_tensor}")
        #     print(f"self.append_kv_page_indices_tensor {self.append_kv_page_indices_tensor}")
        #     print(f"self.append_kv_page_indptr_tensor {self.append_kv_page_indptr_tensor}")
        #     print(f"self.append_kv_last_page_len_tensor {self.append_kv_last_page_len_tensor}")
        #     print("\n\n")

        flashinfer.append_paged_kv_cache(
            key,
            value,
            self.append_qo_indptr_tensor,
            self.layered_kv_cahce[layer_cache_idx],
            self.append_kv_page_indices_tensor,
            self.append_kv_page_indptr_tensor,
            self.append_kv_last_page_len_tensor
        )

        output = self.flash_attn_wrapper.forward(
            query,
            self.layered_kv_cahce[layer_cache_idx],
            causal=True,
            pos_encoding_mode="NONE",
            sm_scale=softmax_scale,
        )
        output = output.reshape(-1, self.num_q_heads * self.head_dim)
        return output


    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32, device="cuda")


    def forward_llama(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        # attention_wrapper: BaseAttentionWrapper = None,
    ):
        bsz, seqlen, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # attention
        if self.kv_cache is not None:
            # print("xq.shape", xq.shape)
            # print("xk.shape", xk.shape)
            # if self.layer is not None and self.layer == 0:
            #     print(f"k_cache.shape {self.kv_cache[0].shape}")
            #     print(f"cached_seq_len {self.cached_seq_len}")
            #     print(f"block_tables {self.block_tables}")
            output = flash_attn_with_kvcache(
                xq, 
                self.kv_cache[0], 
                self.kv_cache[1],
                xk,
                xv,
                cache_seqlens=self.cached_seq_len,
                block_table=self.block_tables,
                softmax_scale=None,
                causal=True
            )
            self.cached_seq_len.add_(seqlen)
            # print(f"flash output: {output.shape}")
            output = output.contiguous().view(bsz, seqlen, -1)
            #self.cached_seq_len = torch.add(self.cached_seq_len, seqlen)
        else:
            # KV cache update
            if self.cache is not None:
                # update the KV cache with current KV and get all the previous KVs
                xk, xv = self.cache.update(start_pos, xk, xv)
            # repeat k/v heads if n_kv_heads < n_heads (GQA)
            xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            # make heads be a batch dim
            xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
            if self.flash:
                output = F.scaled_dot_product_attention(xq, xk, xv, mask)
            else:
                scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
            # concatenate all the heads
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # output projection
        proj = self.wo(output)
        return proj
    
   
def apply_rotary_emb(x, freqs_cis):
    # shape gymnastics let's go
    # x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
