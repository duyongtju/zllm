

import json
import logging
from typing import AsyncGenerator

from sse_starlette import EventSourceResponse
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from zllm.config.config import CacheConfig, ModelConfig, ParallelConfig, SystemConfig, VllmSchedulerConfig
from zllm.core.datatypes.sampling_params import SamplingParams
from zllm.engine.async_llm_engine import AsyncLLMEngine
from zllm.utils import random_uuid
from zllm.logger import init_logger

logger = init_logger(__name__)
logger.setLevel(logging.DEBUG)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine: AsyncLLMEngine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(request_id, prompt, sampling_params)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            if request_output.finished:
                yield '\0'
                break    
            prompt = request_output.prompt
            text_outputs = prompt + request_output.text
            ret = {"text": text_outputs}
            logger.debug(f"ret {ret}")
            # yield (json.dumps(ret) + "\0").encode("utf-8")
            yield (json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf-8")

            # yield {"text": text_outputs}

    if stream:
        # headers = {
        #     "Content-Type": "text/event-stream; charset=utf-8",
        #     "Transfer-Encoding": "chunked",
        # }
        # return EventSourceResponse(stream_results())
        return StreamingResponse(stream_results(), media_type="text/event-stream")

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = prompt + final_output.text
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
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

    engine = AsyncLLMEngine.from_system_config(
        config, verbose=True
    )


    app.root_path = '/home/duyong/src/github.com/karpathy/nano-llama31'
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level="debug",
        # ssl_keyfile=config.ssl_keyfile,
        # ssl_certfile=config.ssl_certfile,
        # ssl_ca_certs=config.ssl_ca_certs,
        # ssl_cert_reqs=config.ssl_cert_reqs,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
