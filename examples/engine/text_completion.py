import copy
import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from zllm.config.config import CacheConfig, ModelConfig, ParallelConfig, SystemConfig, VllmSchedulerConfig
from zllm.core.datatypes.request_output import RequestOutput
from zllm.core.datatypes.sampling_params import SamplingParams
from zllm.engine.base_llm_engine import BaseLLMEngine

ckpt_dir = '/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct/'
model_config: ModelConfig = ModelConfig(
    model=ckpt_dir,
    trust_remote_code=True,
    dtype='bfloat16'
)
scheduler_config: VllmSchedulerConfig = VllmSchedulerConfig(
    max_num_seqs=16,
    max_batched_tokens=8*1024,
)
cache_config = CacheConfig(
    num_gpu_blocks=200,
    block_size=256,
)
parallel_config = ParallelConfig(
    pipeline_parallel_size=1,
    tensor_parallel_size=1,    
)
config = SystemConfig(
    model_config=model_config,
    cache_config=cache_config,
    parallel_config=parallel_config,
)

def main():

    engine = BaseLLMEngine(config)

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    sampling_params = SamplingParams(temperature=0.0,
                                    stop=['<|end_of_text|>', '<|eot_id|>'],
                                    max_tokens=128)
    
    inputs = copy.deepcopy(prompts)
    results = generate(engine, inputs, sampling_params)


    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result.text}")
        print("\n==================================\n")


def generate(engine, inputs, sampling_params):
    # example_prompt = "Who is the president of the United States?"

    outputs: List[RequestOutput] = [None]*len(inputs)
    
    request_id = -1
    while engine.has_unfinished_requests() or inputs:
        if inputs:
            request_id +=1 
            prompt = inputs.pop(0)
            engine.add_request(
                prompt=prompt,
                sampling_params=sampling_params,
                seq_id=str(request_id),
            )

        request_outputs: List[RequestOutput] = engine.step()

        for output in request_outputs:
            if output.finished: 
                idx = int(output.seq_id)
                outputs[idx] = output

    for output in request_outputs:
        idx = int(output.seq_id)
        outputs[idx] = output
   
    return outputs
    
if __name__ == "__main__":
    main()