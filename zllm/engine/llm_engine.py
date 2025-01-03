from zllm.config.config import SystemConfig
from zllm.engine.base_llm_engine import BaseLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig)-> "BaseLLMEngine":
        engine = BaseLLMEngine(config)
        return engine