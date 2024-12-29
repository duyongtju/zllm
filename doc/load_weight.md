# zllm load weight

zllm 中的 model 都实现了 load_weight 方法，该方法用于加载模型的权重，以 llama 为例进行说明：

## 先看看权重文件中有哪些权重

``` python
from zllm.model_executor.weight_utils import hf_model_weights_iterator

model_path = '/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct'
for name, weight in hf_model_weights_iterator(model_path):
    print(name)
```

输出
```text
// 这里省略了一些权重，只列出部分, 顺序也有调整
lm_head.weight
model.norm.weight
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
...
model.layers.31.input_layernorm.weight
model.layers.31.self_attn.k_proj.weight
model.layers.31.self_attn.o_proj.weight
model.layers.31.self_attn.q_proj.weight
model.layers.31.self_attn.v_proj.weight
model.layers.31.post_attention_layernorm.weight
model.layers.31.mlp.gate_proj.weight
model.layers.31.mlp.up_proj.weight
model.layers.31.mlp.down_proj.weight
```

## zllm 中的模型结构

```python
from zllm.config.config import ParallelConfig
from zllm.model_executor.models.llama import LlamaModel
from transformers import LlamaConfig

from zllm.worker.base_worker import_init_distributed_environment

parallel_config = ParallelConfig(
    pipeline_parallel_size=1, tensor_parallel_size=1,
)
_init_distributed_environment(
    parallel_config,
    rank=0,
    distributed_init_method='tcp://127.0.0.1:35500'
)

model_path = '/home/duyong/model-zoos/meta-llama/Meta-Llama-3.1-8B-Instruct'
config = LlamaConfig.from_pretrained(model_path)

model = LlamaModel(config)

model
```

输出
```text
LlamaModel(
  (embed_tokens): VocabParallelEmbedding()
  (layers): ModuleList(
    (0-31): 32 x LlamaDecoderLayer(
      (self_attn): LlamaAttention(
        (qkv_proj): ColumnParallelLinear()
        (o_proj): RowParallelLinear()
        (rotary_emb): Llama3RotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_up_proj): ColumnParallelLinear()
        (down_proj): RowParallelLinear()
        (act_fn): SiluAndMul()
      )
      (input_layernorm): RMSNorm()
      (post_attention_layernorm): RMSNorm()
    )
  )
  (norm): RMSNorm()
)
```

## 权重文件可以分成这几类

- embedding权重： embed_tokens
- TP 且合并的权重：self_attn.qkv_proj, mlp.gate_up_proj
- 仅 TP 权重：self_attn.o_proj, mlp.down_proj
- 非 TP 权重：各实例都有相同权重的层：self_attn.rotary_emb, input_layernorm, post_attention_layernorm, mlp.act_fn, norm

zllm 中加载权重的方法

```python
class LlamaForCausalLM(nn.Module):
    # 其他方法

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        weight_suffixes = ["weight"]

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        pp_model_parallel_rank = get_pipeline_model_parallel_rank()

        assert self.config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_size

        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1

        q_proj_shard_size = self.config.hidden_size // tp_size
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * self.config.num_key_value_heads
            // tp_size
        )
        # 推理的时候 q_proj, k_proj, v_proj 三个操作是合并为 qkv_proj, qkv_proj 权重分布为 [ q_proj + k_proj + v_proj, : ]
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                continue

            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(str(layer_id), str(new_layer_id))

            # 处理attention层合并并且TP的权重
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                # 这里将 qkv_proj 权重切片为三个部分，分别是 q, k, v 的投影矩阵，分片对应不同的 offset 和 shard_size，然后赋值给对应的参数。
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            # 处理MLP层合并并且TP的权重
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]

            # 处理embedding权重，例如 'model.embed_tokens'、'lm_head'
            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                continue

            # 其他权重 layer_norm、mlp 其中权重加载，例如 
            # 'model.layers.1.input_layernorm.weight'、'model.layers.1.mlp.down_proj.weight'
            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )
```
