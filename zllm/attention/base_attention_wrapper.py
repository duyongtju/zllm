
from abc import abstractmethod
import random
from typing import List, Mapping, Optional, Tuple
from torch import nn
import torch
from xformers import ops as xops
import math


import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from zllm.config.config import ModelArgs, CacheConfig
from zllm.datatypes.sequence import Sequence


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = args.flash # use flash attention?
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
    def __init__(self, args: ModelArgs, layer=0):
        super().__init__(args)
        self.model_config: ModelArgs = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.kv_cache = None
        self.cache = None
        self.layer=layer

        # self.cache_config: CacheConfig = kwargs.pop('cache_config', {})
        # self.cache = self.init_cache_engine()

        # self.attn = Attention(self.model_config)

    
    # def load_state_dict(self, state_dict, strict=True):
    #     self.attn.load_state_dict(state_dict, strict=strict)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     start_pos: int,
    #     freqs_cis: torch.Tensor,
    #     mask: Optional[torch.Tensor],
    # ):
    #     return super().forward(x, start_pos, freqs_cis, mask)
        # return self.attn.forward(x, start_pos, freqs_cis, mask)
        
    def init_gpu_cache(self, bs: int, 
                       max_seq_len: int, 
                       num_gpu_blocks: int,
                       block_size: int = 256,
                       ) -> None:
        if not self.model_config.paged:
            return

        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.kv_dtype = self.wk.weight.dtype
        self.kv_device = self.wk.weight.device
        self.n_layers = self.model_config.n_layers

        self.kv_cache = self.get_cache_block(
            self.num_gpu_blocks, dtype=self.kv_dtype, device=self.kv_device,
        )

        num_blocks_per_seq = (max_seq_len + self.block_size - 1)//self.block_size
        blocks = random.sample(range(0, num_gpu_blocks), num_blocks_per_seq*bs)
        block_tables = []
        for i in range(bs):
            block_tables.append(blocks[i*num_blocks_per_seq:(i+1)*num_blocks_per_seq])
        self.block_tables = torch.tensor(block_tables, device=self.kv_device, dtype=torch.int32)
        self.cached_seq_len = torch.tensor([0]*bs, device=self.kv_device, dtype=torch.int32)


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
        block_tables = []
        for seq in seq_list:
            blocks = [logical_block.block_number for logical_block in seq.logical_token_blocks]    
            block_tables.append(blocks)
        self._step_block_tables = torch.tensor(block_tables, dtype=torch.int32)

        cached_seq_len = []
        for seq in seq_list:
            # todo: fixme 加入已缓存的 token 长度
            cached_seq_len.append(0)
        self.cached_seq_len = torch.tensor(cached_seq_len, dtype=torch.int32)

    # def kv_cache_dtype(self) -> torch.dtype:
    #     self.attn.wq.weight.dtype
    
    # def kv_cache_device(self) -> torch.device:
    #     self.attn.wq.weight.device

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
