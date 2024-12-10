import math
import random
import time
import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from xformers import ops as xops

torch.set_default_dtype(torch.float16)
torch.set_default_device("cuda:0")

bs = 16
seq_len = 200
n_head = 16
head_dim = 64
max_seqlen = 5120


query_states = torch.randn((bs, n_head, seq_len, head_dim), dtype = torch.float16)
key_states = torch.randn((bs, n_head, seq_len, head_dim), dtype = torch.float16)
value_states = torch.randn((bs, n_head, seq_len, head_dim), dtype = torch.float16)

flash_query_states = query_states.transpose(1, 2)
flash_key_states = key_states.transpose(1, 2)
flash_value_states = value_states.transpose(1, 2)

num_blocks = 200
block_size = 256
dtype = torch.float16
key_cache = torch.randn(num_blocks, block_size, n_head, head_dim, dtype=dtype)
value_cache = torch.randn_like(key_cache, dtype=dtype)

cache_lens=[seq_len for _ in range(bs)]
cache_seqlens_tensor=torch.tensor(cache_lens, dtype=torch.int32)

block_table = []
num_block_per_seq = 5
blocks_ids = random.sample(range(num_blocks), bs*num_block_per_seq)
for i in range(bs):
    block_table.append(blocks_ids[i*num_block_per_seq:(i+1)*num_block_per_seq])
block_table = torch.tensor(block_table, dtype=torch.int32)

def flash_attn(q, k, v):
    # output = flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=True)
    output = flash_attn_with_kvcache(q, 
                                     key_cache, 
                                     value_cache, 
                                     k, v, 
                                     cache_seqlens=cache_seqlens_tensor, 
                                     block_table=block_table, 
                                     softmax_scale=None, 
                                     causal=True)
    return output

def standard_attention(query, key, value):
    attention_mask = torch.tril(torch.ones((max_seqlen, max_seqlen), dtype = torch.bool)).view(1, 1, max_seqlen, max_seqlen)
    attention_mask = attention_mask.to(torch.float16)
    attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float16).min
    # attention_mask = attention_mask[:,:, cache_len:]

    attn_output = []

    bs, _, seq_len, _ = query.shape
    for i in range(bs):
        cache_len = cache_lens[i]
        n_block = (cache_lens[i] + block_size - 1)//block_size
        cached_key, cached_value = [], []

        for j in range(n_block):
            n_kv_in_block = min(block_size, cache_len-j*block_size)

            key_block = key_cache[block_table[i][j]]
            cached_key.append(key_block[:n_kv_in_block])
            
            value_block = value_cache[block_table[i][j]]
            cached_value.append(value_block[:n_kv_in_block])

        # print(f"cached_key[0].shape {cached_key[0].shape}, key.shape {key.shape}")
        key_state = torch.concat((*cached_key, key[i].transpose(0,1)), dim=0)
        key_state = torch.transpose(key_state, 0, 1)

        value_state = torch.concat((*cached_value, value[i].transpose(0, 1)), dim=0)
        value_state = torch.transpose(value_state, 0, 1)

        _query = query[i]
        # print(f"_query {_query.shape}, key_state {key_state.shape}")
        attn_weights = torch.matmul(_query, key_state.transpose(1, 2)) / math.sqrt(head_dim)
        
        mask = attention_mask[:,:,cache_len:cache_len+seq_len,:cache_len+seq_len]
        # print(f"attn_weights {attn_weights.shape}, mask {mask.shape}")
        attn_weights = attn_weights + mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # print(f"attn_weights {attn_weights.shape}, value_state {value_state.shape}")
        output = torch.matmul(attn_weights, value_state)
        output = output.transpose(1, 2)

        attn_output.append(output)

    return torch.cat(attn_output, dim=0)

flash_attn_output = flash_attn(
        flash_query_states, flash_key_states, flash_value_states
    )
standard_attention_output = standard_attention(
        query_states, key_states, value_states
    )
print(f"flash_attn_output {flash_attn_output.shape}, standard_attention_output {standard_attention_output.shape}")
print(torch.allclose(flash_attn_output, standard_attention_output, rtol=2e-3, atol=2e-3))


print((flash_attn_output - standard_attention_output).abs().max())
print((flash_attn_output - standard_attention_output).abs().mean())

# compare between xformer and flash_attention
# xformer_output = xops.memory_efficient_attention(
#     flash_query_states, flash_key_states, flash_value_states, attn_bias=xops.LowerTriangularMask()
#     )
# print(torch.allclose(flash_attn_output, xformer_output, rtol=2e-3, atol=2e-3))


start = time.time()
for i in range(200):
    flash_output = flash_attn(flash_query_states, flash_key_states, flash_value_states)
print("Flash Attention Time: ", time.time()- start)


start = time.time()
for i in range(200):
    standard_attention_output = standard_attention(query_states, key_states, value_states)
print("Standard Attention Time: ", time.time()- start)
