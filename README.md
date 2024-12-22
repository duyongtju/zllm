# zllm

这个仓库是一个正在进行中的项目，主要目标是提供一个类似于 vllm 的大模型推理框架，从零开始构建。

此外，在 examples 目录下包括了一些示例代码，用于演示如何使用这个框架进行大模型的推理。


**已支持 feature**

- **TP inference**: 利用 fairscale 通过 Tensor Parallelism (TP) 多卡推理，通过修改 official llama code，引入自定义的 load_weights 方法，从而在不同的卡上加载不同的权重。
- **flashinfer**: 实现了基于flash attention的推理，引入 flashinfer 库，实现了高效的注意力计算。可以在 examples/attention 目录下找到相关示例代码。
- **continuous batch**
  - block_manager: 实现了基于block的内存管理，通过将模型权重分割成多个块（blocks），并在推理过程中动态地加载和卸载这些块。
  - scheduler: 实现了基于block的调度器，scheduler 内部维护 waitting、running 队列，目前的调度策略还比较简单，在一个调度周期内，会将 waitting 队里和 running 队里中的所有请求取出，进行推理，所以在一个调度周期内会同时 prefill 和 decode。后续有优化，支持更丰富的调度策略。

**未来计划**

- **引入worker层**: 目前 worker 和 engine 层是在一个进程中，engine 将调度的结果交给 worker 层，后续计划将这部分逻辑分离到独立的进程中去，并在 worker 中增加 init_distributed_environment, init_model, init_model, execute_model 等方法，从而实现 worker 层和 engine 层的解耦。计划采用 ray 管理进程，zmq 作为通信层。
- **从HF加载模型**: 目前只支持 Llama3.1 官方原始的权重，也就是 .bin 格式，计划支持加载 transformer 格式的模型和 tokenizer，因为 HF 上收集了很多模型。
- **支持更多调度策略**: 目前只支持简单的调度策略，后续计划支持更丰富的调度策略，包括设置调度预算、驱逐、抢占等。
- **性能优化**： torch compiler, wq、wk、wv 算子融合, 引入 ROE 算子 等。
- **benchmark**: 增加benchmark。
- **支持更多模型**: 目前只支持 llama3.1，后续计划支持其他大语言模型。
- **支持更多硬件**: 目前只支持基于CUDA的GPU推理，后续计划支持华为910加速卡。
- **支持多模态模型**: 目前只支持文本生成，后续计划支持图像、视频等模态的推理。
- 其他
  - 量化
  - prefix cache
  - prompt cache



### let's go

Download the official `llama-models` repo, e.g. inside this project's directory is ok:

```bash
git clone https://github.com/meta-llama/llama-models.git
```

Download a model, e.g. the Llama 3.1 8B (base) model:

```bash
cd llama-models/models/llama3_1
chmod u+x download.sh
./download.sh
```

You'll have to enter a "URL from the email". For this you have to request access to Llama 3.1 [here](https://llama.meta.com/llama-downloads/). Then when it asks which model, let's enter `meta-llama-3.1-8b`, and then again one more time `meta-llama-3.1-8b` to indicate the base model instead of the instruct model. This will download about 16GB of data into `./Meta-Llama-3.1-8B` - 16GB because we have ~8B params in 2 bytes/param (bfloat16).

```bash
cd examples/engine
# 修改模型路径
python text_completion.py
I believe the meaning of life is
>  to find your passion and pursue it with all your heart. I believe that life is too short to waste time on things that don't bring you joy. I believe that everyone has a unique purpose and that it's our job to discover what that is and live it out.
I believe that life is a journey, not a destination. I believe that the journey is just as important as the destination, and that the experiences we have along the way shape us into the people we are meant to be. I believe that we should take risks, try new things, and push ourselves outside of our comfort zones.
I believe that love is the most

==================================

Simply put, the theory of relativity states that 
> 1) the laws of physics are the same for all observers in uniform motion relative to one another and 2) the speed of light is always constant, regardless of the motion of the observer or the source of light. This theory was developed by Albert Einstein in the early 20th century and has been widely accepted as a fundamental principle of modern physics.
The first part of the theory, special relativity, was introduced in 1905 and posits that the laws of physics are the same for all observers in uniform motion relative to one another. This means that time and space are relative, and their measurement depends on the observer's frame

==================================

A brief message congratulating the team on the launch:

        Hi everyone,

        I just 
>  wanted to take a moment to congratulate the team on the launch of our new product. It's been a long and challenging journey, but the end result is truly something to be proud of.

        I know that there have been many late nights, early mornings, and countless cups of coffee consumed along the way. But it's all been worth it to see our vision come to life.

        I'm so grateful to be a part of this team and to have had the opportunity to work alongside such talented and dedicated individuals. Your hard work and commitment have not gone unnoticed, and I'm honored to be a part of this team.

        Thank

==================================

Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>
>  fromage
        chocolate => chocolat
        strawberry => fraise
        pineapple => ananas
        kiwi => kiwi
        orange => orange
        lemon => citron
        grapefruit => pamplemousse
        watermelon => pastèque
        peach => pêche
        pear => poire
        apple => pomme
        banana => banane
        mango => mangue
        papaya => papaye
        watermelon => pastèque
        coconut => noix de coco
        avocado => avocat
        tomato => tomate
        cucumber => concom

==================================
```
