
from abc import abstractmethod
import random
from typing import List, Mapping, Optional, Tuple
from torch import nn
import torch
from xformers import ops as xops
import math


import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
import flashinfer


from zllm.config.config import ModelArgs, CacheConfig
from zllm.core.datatypes.sequence import Sequence, SequenceMetadata


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # self.flash = args.flash # use flash attention?
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1 # AK: model parallel size is 1 for 1 GPU
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False )
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # will be KVCache object managed by inference context manager
        # self.cache = None

    def forward(
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
        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            xk, xv = self.cache.update(start_pos, xk, xv)
        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
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


class BaseAttentionWrapper(Attention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.model_config: ModelArgs = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.kv_cache = None
        # self.cache = None

        self.flash_attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128*1024*1024, dtype=torch.uint8, device="cuda")
        )
        # self.cache_config: CacheConfig = kwargs.pop('cache_config', {})
        # self.cache = self.init_cache_engine()

        # self.attn = Attention(self.model_config)
        
    def init_gpu_cache(self, bs: int, 
                       max_seq_len: int, 
                       num_gpu_blocks: int,
                       block_size: int = 256,
                       ) -> None:

        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.kv_dtype = self.wk.weight.dtype
        self.kv_device = self.wk.weight.device
        self.n_layers = self.model_config.n_layers

        self.device = self.kv_device

        self.layered_kv_cahce = []
        for i in range(self.n_layers):
            kv_cache = self.get_cache_block(
                self.num_gpu_blocks, dtype=self.kv_dtype, device=self.kv_device,
            )
            self.layered_kv_cahce.append(kv_cache)

        num_blocks_per_seq = (max_seq_len + self.block_size - 1)//self.block_size
        blocks = random.sample(range(0, num_gpu_blocks), num_blocks_per_seq*bs)
        block_tables = []
        for i in range(bs):
            seq_block_start = i*num_blocks_per_seq
            seq_block_end = (i+1)*num_blocks_per_seq
            block_tables.append(blocks[seq_block_start:seq_block_end])
        self.block_tables = torch.tensor(block_tables, device=self.kv_device, dtype=torch.int32)
        self.cached_seq_len = torch.tensor([0]*bs, device=self.kv_device, dtype=torch.int32)

    def init_gpu_cache2(self,
                       num_gpu_blocks: int,
                       block_size: int = 256,
                       ) -> None:

        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.kv_dtype = self.wk.weight.dtype
        self.kv_device = self.wk.weight.device
        self.n_layers = self.model_config.n_layers

        self.device = self.kv_device

        self.layered_kv_cahce = []
        for _ in range(self.n_layers):
            kv_cache = self.get_cache_block(
                self.num_gpu_blocks, dtype=self.kv_dtype, device=self.kv_device,
            )
            self.layered_kv_cahce.append(kv_cache)

    def get_cache_block(self, num_blocks: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn(
            num_blocks,
            self.block_size,
            self.n_kv_heads,
            self.head_dim,
            **kwargs,
        ), torch.randn(num_blocks,
            self.block_size,
            self.n_kv_heads,
            self.head_dim,
            **kwargs,))

    def begin_forward(self, seq_list: List[Sequence]):
        # prepare qo_indptr for example [0, 33, 44, 55, 66, 77, 88, 100]
        # paged_kv_indices [ 1, 2, 3, 4, 5 ]
        # paged_kv_indptr [ 0, 2, 5, ...] seq_1 page: 1 2; seq_2 page 3 4 5  
        # paged_kv_last_page_len [1, 7, 14, 4, 3, 1, 16]  1 <= paged_kv_last_page_len <= page_size
        qo_indptr = [0]        
        paged_kv_indices = []
        paged_kv_indptr = [0]
        paged_kv_last_page_len = []

        append_qo_indptr = [0]
        append_kv_page_indices = []
        append_kv_page_indptr = [0]
        append_kv_last_page_len = []


        for seq in seq_list:                
            seq_id = int(seq.seq_id)
            if seq.prompt_processing_finished: # prefill phase
                continue
            last_page_len = seq.get_prompt_len() % self.block_size or self.block_size

            qo_indptr.append(qo_indptr[-1]+len(seq.prompt_token_ids))
            kv_page_start = 0
            kv_page_num = (len(seq.prompt_token_ids) + self.block_size -1) // self.block_size
            paged_kv_indices.extend(self.block_tables[seq_id][kv_page_start:kv_page_num])
            paged_kv_indptr.append(paged_kv_indptr[-1]+kv_page_num)
            paged_kv_last_page_len.append(last_page_len)

            append_qo_indptr.append(append_qo_indptr[-1]+len(seq.prompt_token_ids))
            append_kv_page_start = 0
            append_kv_page_end = (len(seq.prompt_token_ids) + self.block_size -1) // self.block_size
            append_kv_page_indices.extend(self.block_tables[seq_id][append_kv_page_start:append_kv_page_end])
            append_kv_page_indptr.append(append_kv_page_indptr[-1]+append_kv_page_end-append_kv_page_start)
            append_kv_last_page_len.append(last_page_len)

        for seq in seq_list:
            seq_id = int(seq.seq_id)
            if not seq.prompt_processing_finished: # decode phase
                continue
            context_len = len(seq.get_token_ids())
            last_page_len = context_len%self.block_size or self.block_size

            qo_indptr.append(qo_indptr[-1]+1)

            kv_page_start = (context_len + self.block_size -1) // self.block_size -1
            kv_page_num = (context_len + self.block_size -1) // self.block_size

            paged_kv_indices.extend(self.block_tables[seq_id][kv_page_start:kv_page_num])
            paged_kv_indptr.append(paged_kv_indptr[-1]+1)
            paged_kv_last_page_len.append(last_page_len)

            append_qo_indptr.append(append_qo_indptr[-1]+1)
            append_kv_page_indices.extend(self.block_tables[seq_id][kv_page_start:kv_page_num])
            append_kv_page_indptr.append(append_kv_page_indptr[-1]+1)
            append_kv_last_page_len.append(last_page_len)
        
        qo_indptr = self.to_int_tensor(qo_indptr)
        paged_kv_indptr = self.to_int_tensor(paged_kv_indptr)
        paged_kv_indices = self.to_int_tensor(paged_kv_indices)
        paged_kv_last_page_len = self.to_int_tensor(paged_kv_last_page_len)

        self.append_qo_indptr_tensor = self.to_int_tensor(append_qo_indptr)
        self.append_kv_page_indices_tensor = self.to_int_tensor(append_kv_page_indices)
        self.append_kv_page_indptr_tensor = self.to_int_tensor(append_kv_page_indptr)
        self.append_kv_last_page_len_tensor = self.to_int_tensor(append_kv_last_page_len)

        print(f"qo_indptr {qo_indptr} \npaged_kv_indptr {paged_kv_indptr}")
        print(f"paged_kv_indices {paged_kv_indices} \npaged_kv_last_page_len {paged_kv_last_page_len}")
        print(f"self.block_tables {self.block_tables}")

        self.flash_attn_wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.model_config.n_heads,
            self.n_kv_heads,
            self.model_config.dim,
            self.block_size,
            causal=True
        )

    def begin_forward2(self, seq_list: List[SequenceMetadata]):
        # prepare qo_indptr for example [0, 33, 44, 55, 66, 77, 88, 100]
        # paged_kv_indices [ 1, 2, 3, 4, 5 ]
        # paged_kv_indptr [ 0, 2, 5, ...] seq_1 page: 1 2; seq_2 page 3 4 5  
        # paged_kv_last_page_len [1, 7, 14, 4, 3, 1, 16]  1 <= paged_kv_last_page_len <= page_size
        qo_indptr = [0]        
        paged_kv_indices = []
        paged_kv_indptr = [0]
        paged_kv_last_page_len = []

        append_qo_indptr = [0]
        append_kv_page_indices = []
        append_kv_page_indptr = [0]
        append_kv_last_page_len = []

        self.block_tables = {}

        for seq in seq_list:                
            if not seq.is_prompt: # prefill phase
                continue
            context_len = seq.num_prompt_tokens         

            last_page_len = context_len % self.block_size or self.block_size

            qo_indptr.append(qo_indptr[-1]+context_len)
            kv_page_start = 0
            kv_page_num = (context_len + self.block_size -1) // self.block_size
            paged_kv_indices.extend(seq.block_table[kv_page_start:kv_page_num])
            paged_kv_indptr.append(paged_kv_indptr[-1]+kv_page_num)
            paged_kv_last_page_len.append(last_page_len)

            append_qo_indptr.append(append_qo_indptr[-1]+context_len)
            append_kv_page_start = 0
            append_kv_page_end = (context_len + self.block_size -1) // self.block_size
            append_kv_page_indices.extend(seq.block_table[append_kv_page_start:append_kv_page_end])
            append_kv_page_indptr.append(append_kv_page_indptr[-1]+append_kv_page_end-append_kv_page_start)
            append_kv_last_page_len.append(last_page_len)

            self.block_tables[seq.seq.seq_id] = seq.block_table

        for seq in seq_list:
            if seq.is_prompt: # decode phase
                continue
            context_len = len(seq.seq.get_token_ids())
            last_page_len = context_len%self.block_size or self.block_size

            qo_indptr.append(qo_indptr[-1]+1)

            kv_page_start = (context_len + self.block_size -1) // self.block_size -1
            kv_page_num = (context_len + self.block_size -1) // self.block_size

            paged_kv_indices.extend(seq.block_table[kv_page_start:kv_page_num])
            paged_kv_indptr.append(paged_kv_indptr[-1]+1)
            paged_kv_last_page_len.append(last_page_len)

            append_qo_indptr.append(append_qo_indptr[-1]+1)
            append_kv_page_indices.extend(seq.block_table[kv_page_start:kv_page_num])
            append_kv_page_indptr.append(append_kv_page_indptr[-1]+1)
            append_kv_last_page_len.append(last_page_len)

            self.block_tables[seq.seq.seq_id] = seq.block_table
        
        qo_indptr = self.to_int_tensor(qo_indptr)
        paged_kv_indptr = self.to_int_tensor(paged_kv_indptr)
        paged_kv_indices = self.to_int_tensor(paged_kv_indices)
        paged_kv_last_page_len = self.to_int_tensor(paged_kv_last_page_len)

        self.append_qo_indptr_tensor = self.to_int_tensor(append_qo_indptr)
        self.append_kv_page_indices_tensor = self.to_int_tensor(append_kv_page_indices)
        self.append_kv_page_indptr_tensor = self.to_int_tensor(append_kv_page_indptr)
        self.append_kv_last_page_len_tensor = self.to_int_tensor(append_kv_last_page_len)

        print(f"qo_indptr {qo_indptr} \npaged_kv_indptr {paged_kv_indptr}")
        print(f"paged_kv_indices {paged_kv_indices} \npaged_kv_last_page_len {paged_kv_last_page_len}")
        print(f"self.block_tables {self.block_tables}")

        self.flash_attn_wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.model_config.n_heads,
            self.n_kv_heads,
            self.model_config.dim,
            self.block_size,
            causal=True
        )


    def end_forward(self):
        self.flash_attn_wrapper.end_forward()

    def forward2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache_idx: int,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        
        query = query.contiguous().reshape(-1, self.model_config.n_heads, self.head_dim)
        key = key.contiguous().reshape(-1, self.model_config.n_kv_heads, self.head_dim)
        value = value.contiguous().reshape(-1, self.model_config.n_kv_heads, self.head_dim)
        
        if layer_cache_idx == 0:
            print(f"layer {layer_cache_idx} \n"+"="*20)
            print(f"query {query.shape}")
            print(f"key {key.shape}")
            print(f"value {value.shape}")
            print(f"layered_kv_cahce[{layer_cache_idx}][0]: {self.layered_kv_cahce[layer_cache_idx][0].shape}")
            print(f"self.append_qo_indptr_tensor {self.append_qo_indptr_tensor}")
            print(f"self.append_kv_page_indices_tensor {self.append_kv_page_indices_tensor}")
            print(f"self.append_kv_page_indptr_tensor {self.append_kv_page_indptr_tensor}")
            print(f"self.append_kv_last_page_len_tensor {self.append_kv_last_page_len_tensor}")
            print("\n\n")

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
        return output


    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32, device="cuda")


    def forward(
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
            if self.layer is not None and self.layer == 0:
                print(f"k_cache.shape {self.kv_cache[0].shape}")
                print(f"cached_seq_len {self.cached_seq_len}")
                print(f"block_tables {self.block_tables}")
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
    

    def raw_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        # attention_wrapper: BaseAttentionWrapper = None,
    ):
        return super(BaseAttentionWrapper, self).forward(
            x, start_pos, freqs_cis, mask,
        )
   
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

class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        seqlen = xk.size(1)
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv
