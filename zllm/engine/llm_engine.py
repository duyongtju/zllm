from zllm.config.config import SystemConfig
from zllm.engine.arg_util import EngineArgs
from zllm.engine.base_llm_engine import BaseLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig)-> "BaseLLMEngine":
        engine = BaseLLMEngine(config)
        return engine
    
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
    ):
        system_config = engine_args.create_engine_config()
        engine = BaseLLMEngine(system_config)
        return engine
