import copy
import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from llama.tokenizer import ChatFormat, Dialog, Tokenizer
from zllm.config.config import CacheConfig, ModelArgs, ParallelConfig, SystemConfig
from zllm.core.datatypes.request_output import RequestOutput
from zllm.core.datatypes.sampling_params import SamplingParams
from zllm.engine.base_llm_engine import BaseLLMEngine


def main():

    ckpt_dir = '/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct-oooooooold/original/'
    tokenizer_path = '/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct-oooooooold/'
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        **params,
    )
    cache_config = CacheConfig(
        block_num=200,
        block_size=256,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,    
    )

    config = SystemConfig(
        model_config=model_args,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )

    engine = BaseLLMEngine(config, ckpt_dir, tokenizer_path)


    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    chats = tokenizer.apply_chat_template(dialogs, tokenize=False)
    
    sampling_params = SamplingParams(temperature=0.0,
                                    stop=['<|end_of_text|>', '<|eot_id|>'],
                                    max_tokens=50)
    

    results = generate(engine, chats, sampling_params)

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"Assistant: {result.text}"
        )
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