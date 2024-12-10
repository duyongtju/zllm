import math
import time
import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from xformers import ops as xops

torch.set_default_dtype(torch.float16)
torch.set_default_device("cuda:0")

bs = 32
seq_len = 512
n_head = 16
head_dim = 64

query_states = torch.randn((bs, n_head, seq_len, head_dim), dtype = torch.float16)
key_states = torch.randn((bs, n_head, seq_len, head_dim), dtype = torch.float16)
value_states = torch.randn((bs, n_head, seq_len, head_dim), dtype = torch.float16)

flash_query_states = query_states.transpose(1, 2)
flash_key_states = key_states.transpose(1, 2)
flash_value_states = value_states.transpose(1, 2)

# query_states = torch.randn((bs, n_head, seq_len, head_dim), dtype=torch.float16).cuda()
# key_states = torch.randn((bs, n_head, seq_len, head_dim), dtype=torch.float16).cuda()
# value_states = torch.randn((bs, n_head, seq_len, head_dim), dtype=torch.float16).cuda()

# flash_query_states = query_states.transpose(1, 2)
# flash_key_states = key_states.transpose(1, 2)
# flash_value_states = value_states.transpose(1, 2)


def flash_attn(q, k, v):
    output = flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=True)
    return output
flash_attn_output = flash_attn_func(
    flash_query_states, flash_key_states, flash_value_states,
    causal=True
)

attention_mask = torch.tril(torch.ones((seq_len, seq_len), dtype = torch.bool)).view(1, 1, seq_len, seq_len)
attention_mask = attention_mask.to(torch.bfloat16)
attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float16).min

def standard_attention(query, key, value, attention_mask):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    # print("attn_output.shape", attn_output.shape)
    return attn_output

standard_attention_output = standard_attention(query_states, key_states, value_states, attention_mask)
print(torch.allclose(flash_attn_output, standard_attention_output, rtol=2e-3, atol=2e-3))


# compare between xformer and flash_attention
xformer_output = xops.memory_efficient_attention(
    flash_query_states, flash_key_states, flash_value_states, attn_bias=xops.LowerTriangularMask()
    )
print(torch.allclose(flash_attn_output, xformer_output, rtol=2e-3, atol=2e-3))

print((flash_attn_output - standard_attention_output).abs().max())
print((flash_attn_output - standard_attention_output).abs().mean())


# start = time.time()
# for i in range(100):
#     flash_output = flash_attn(flash_query, flash_key, flash_value)
# print("Flash Attention Time: ", time.time()- start)


# start = time.time()
# for i in range(100):
#     standard_attention_output = standard_attention(query, key, value, attention_mask)
# print("Standard Attention Time: ", time.time()- start)
