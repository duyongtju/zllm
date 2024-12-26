


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = 'facebook/opt-125m'
    served_model_name: Optional[Union[str, List[str]]] = None
    tokenizer: Optional[str] = None
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    config_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    enable_memory_tiering: bool = False
    enable_layered_transfer: bool = False
    enable_disk_swap: bool = False
    disk_swap_config: str = "swap.cfg"
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = False
    swap_space: float = 4  # GiB
    cpu_offload_gb: float = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_context_len_to_capture: Optional[int] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = 1
    max_prompt_adapter_token: int = 0
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: Optional[Union[str, torch.dtype]] = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    num_scheduler_steps: int = 1
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_cpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    guided_decoding_backend: str = 'outlines'
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_model_quantization: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    qlora_adapter_name_or_path: Optional[str] = None
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    otlp_traces_endpoint: Optional[str] = None
    collect_detailed_traces: Optional[str] = None
    disable_async_output_proc: bool = False
    override_neuron_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default=EngineArgs.model,
            help='Name or path of the huggingface model to use.')
        parser.add_argument(
            '--tokenizer',
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help='Name or path of the huggingface tokenizer to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            '--skip-tokenizer-init',
            action='store_true',
            help='Skip initialization of tokenizer and detokenizer')
        parser.add_argument(
            '--revision',
            type=nullable_str,
            default=None,
            help='The specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=nullable_str,
            default=None,
            help='The specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=nullable_str,
            default=None,
            help='Revision of the huggingface tokenizer to use. '
            'It can be a branch name, a tag name, or a commit id. '
            'If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-mode',
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=['auto', 'slow', 'mistral'],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            'always use the slow tokenizer. \n* '
            '"mistral" will always use the `mistral_common` tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument('--download-dir',
                            type=nullable_str,
                            default=EngineArgs.download_dir,
                            help='Directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface.')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=[f.value for f in LoadFormat],
            help='The format of the model weights to load.\n\n'
            '* "auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available.\n'
            '* "pt" will load the weights in the pytorch bin format.\n'
            '* "safetensors" will load the weights in the safetensors format.\n'
            '* "npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading.\n'
            '* "dummy" will initialize the weights with random values, '
            'which is mainly for profiling.\n'
            '* "tensorizer" will load the weights using tensorizer from '
            'CoreWeave. See the Tensorize vLLM Model script in the Examples '
            'section for more information.\n'
            '* "bitsandbytes" will load the weights using bitsandbytes '
            'quantization.\n')
        parser.add_argument(
            '--config-format',
            default=EngineArgs.config_format,
            choices=[f.value for f in ConfigFormat],
            help='The format of the model config to load.\n\n'
            '* "auto" will try to load the config in hf format '
            'if available else it will try to load in mistral format ')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='Data type for model weights and activations.\n\n'
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            'BF16 precision for BF16 models.\n'
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.')
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
            'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')