import json
import os
from pathlib import Path
import time

import torch
from zllm.attention.base_attention_wrapper import BaseAttentionWrapper, ModelArgs, Attention
from zllm.worker.llama31 import precompute_freqs_cis
from zllm.datatypes.sequence import Sequence


def test_attention_wrapper():
    ckpt_dir: str = "/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct-oooooooold/original"

    # param_path = os.path.join(ckpt_dir, "params.json")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len = 5120,
        max_batch_size = 16,
        paged=True,
        **params,
    )
    print(model_args)

    assert model_args.dim % model_args.n_heads == 0
    torch.set_default_dtype(torch.float16)
    torch.set_default_device('cuda:0')

    bsz = 2
    seqlen = 10
    x = torch.randn((bsz, seqlen, model_args.dim), dtype=torch.float16)

    freqs_cis = precompute_freqs_cis(
        model_args.dim//model_args.n_heads, 
        model_args.max_seq_len*2,
        model_args.rope_theta, 
        model_args.use_scaled_rope
    )


    attn_wrapper = BaseAttentionWrapper(model_args)

    start_pos=0
    freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, start_pos), device=x.device), mask]
        )

    # wrapper_output = attn_wrapper.forward(x, start_pos, freqs_cis, mask)

    raw_output = attn_wrapper.raw_forward(x, start_pos, freqs_cis, mask)

    attn_wrapper.init_gpu_cache(bsz, model_args.max_seq_len, 1024, 256)
    # seq_list = [    
    #         Sequence(
    #             seq_id=i, prompt=str(i),
    #             prompt_token_ids=[range(10)], block_size=256,
    #             eos_token_id=10000,
    #             arrival_time=time.time(),
    #             sampling_params=None,
    #         ) for i in range(bsz)
    #     ]
    # attn_wrapper.begin_forward(seq_list)
    wrapper_output = attn_wrapper.forward(x, start_pos, freqs_cis, mask)

    print(torch.allclose(wrapper_output, raw_output, rtol=2e-3, atol=2e-3))
    print(wrapper_output[0][0][:5])


test_attention_wrapper()