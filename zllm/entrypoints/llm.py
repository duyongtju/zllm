

from typing import List, Optional, Sequence, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from zllm.core.datatypes.request_output import RequestOutput
from zllm.core.datatypes.sampling_params import SamplingParams
from zllm.engine.arg_util import EngineArgs
from zllm.engine.llm_engine import LLMEngine
from zllm.utils import Counter


class LLM:

    def __init__(self) -> None:
        
        engine_args = EngineArgs(
            model='/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct/',
            trust_remote_code=True,
            max_num_seqs=16,
            max_batched_tokens=8*1024,
            num_gpu_blocks=200,
            block_size=256,
            tensor_parallel_size=1
        )

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args
        )
        self.request_counter = Counter()


    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer
    
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None
    )-> List[RequestOutput]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        self._validate_and_add_requests(
            inputs=prompts,
            params=sampling_params,
        )

    def _validate_and_add_requests(
        self,
        inputs: Optional[Union[str, List[str]]] = None,
        params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params
            )

    def _add_request(
        self,
        inputs: str,
        params: SamplingParams
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    inputs,
                                    params)

    def _run_engine(self):
                # Run the engine.
        outputs: List[Union[RequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    total_in_toks += len(output.prompt_token_ids)
                    total_out_toks += len(output.token_ids)
        return sorted(outputs, key=lambda x: int(x.seq_id))
 