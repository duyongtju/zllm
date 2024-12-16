"""
BatchPrefillWithPagedKVCacheWrapper: https://docs.flashinfer.ai/api/prefill.html#flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run
page table layout: https://docs.flashinfer.ai/tutorials/kv_layout.html
"""

import math
import torch
import flashinfer
import torch.nn as nn

num_layers = 32
num_qo_heads = 64
num_kv_heads = 64
head_dim = 128
max_num_pages = 128
page_size = 16
# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)
batch_size = 7
nnz_qo = 100
qo_indptr = torch.tensor(
    [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
)
paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
paged_kv_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)
# 1 <= paged_kv_last_page_len <= page_size
paged_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
kv_cache_at_layer = torch.randn(
    num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
)
# create auxiliary data structures for batch prefill attention
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    causal=True,
)

outputs = []
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    # compute batch prefill attention, reuse auxiliary data structures
    o = prefill_wrapper.run(q, kv_cache)
    outputs.append(o)

print(outputs[0].shape)
# torch.Size([100, 64, 128])

# below is another example of creating custom mask for batch prefill attention
mask_arr = []
qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
for i in range(batch_size):
    mask_i = torch.tril(
        torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
        diagonal=(kv_len[i] - qo_len[i]),
    )
    mask_arr.append(mask_i.flatten())

mask = torch.cat(mask_arr, dim=0)
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    custom_mask=mask,
)
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    # compute batch prefill attention, reuse auxiliary data structures
    o_custom = prefill_wrapper.run(q, kv_cache)
    assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)


def ref_attention():

    seq_1_len = qo_indptr[1] - qo_indptr[0]
    seq_1_q = q_at_layer[0][:seq_1_len]

    seq_1_blocks = [paged_kv_indices[i].cpu().item() for i in range(paged_kv_indptr[1] - paged_kv_indptr[0])]
    # print(seq_1_blocks)
    seq_1_key_state = []
    seq_1_value_state = []

    for i, block_id in enumerate(seq_1_blocks):
        # print(f"block_id {block_id}")
        k_block = kv_cache_at_layer[0, block_id , 0]
        v_block = kv_cache_at_layer[0,block_id,1]
        if i == len(seq_1_blocks)-1:
            seq_1_key_state.append(k_block[:paged_kv_last_page_len[0]]) 
            seq_1_value_state.append(v_block[:paged_kv_last_page_len[0]]) 
        else:
            seq_1_key_state.append(k_block)
            seq_1_value_state.append(v_block)
    seq_1_key_state = torch.concat(seq_1_key_state, dim=0)        
    seq_1_value_state = torch.concat(seq_1_value_state, dim=0)        

    q_len = seq_1_q.size(0)
    kv_len = seq_1_value_state.size(0)
    attention_mask = torch.tril(
        torch.full((q_len, kv_len), 1, device='cuda:0'),
        diagonal=(kv_len-q_len)
    )
    attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float16).min

    seq_1_q = seq_1_q.transpose(0,1)
    seq_1_key_state = seq_1_key_state.transpose(0,1)
    seq_1_value_state = seq_1_value_state.transpose(0,1)

    attn_weights = torch.matmul(seq_1_q, seq_1_key_state.transpose(1,2))/math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(seq_1_q.dtype)
    ref_output = torch.matmul(attn_weights, seq_1_value_state)
    ref_output = ref_output.transpose(0, 1)
    return ref_output

ref_output = ref_attention()
seq_1_len = qo_indptr[1] - qo_indptr[0]
layer1_seq1_output = outputs[0][:seq_1_len]
print(torch.allclose(ref_output, layer1_seq1_output, rtol=1e-3, atol=1e-3))
