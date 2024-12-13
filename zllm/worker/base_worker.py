

import os
from pathlib import Path
import time
from typing import List

import torch
from zllm.config.config import ModelArgs
from zllm.datatypes.sequence import Sequence
from zllm.worker.llama31 import Llama, Transformer

seed = 7

class BaseWroker(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Llama

        self.model = Llama.build(
            ckpt_dir="",
            tokenizer_path="",
            max_seq_len=1024,
            max_batch_size=8,
            
        )
        
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


    def complete(self, seqs: List[Sequence]):

        attention_wrapper = None
