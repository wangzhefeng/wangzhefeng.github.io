---
title: LLM 模型之 Gemma Model Fine-tuning
author: 王哲峰
date: '2024-03-23'
slug: llm-fine-tuning-gemma
categories:
  - nlp
  - deeplearning
  - model
tags:
  - model
---

<style>
details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
}
summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
}
details[open] {
    padding: .5em;
}
details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
}
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Gemma 模型介绍](#gemma-模型介绍)
    - [Prompt 提示词格式](#prompt-提示词格式)
    - [探索未知领域](#探索未知领域)
    - [演示](#演示)
        - [Gemma 指令模型互动对话体验](#gemma-指令模型互动对话体验)
        - [使用 Hugging Face Transformers](#使用-hugging-face-transformers)
        - [JAX 权重](#jax-权重)
    - [与 Google Cloud 集成](#与-google-cloud-集成)
    - [与推理端点集成](#与推理端点集成)
    - [使用 Hugging Face TRL 进行微调](#使用-hugging-face-trl-进行微调)
        - [额外资源](#额外资源)
- [Gemma 模型中文指令微调](#gemma-模型中文指令微调)
    - [Hugging Face TRL + Colab GPU](#hugging-face-trl--colab-gpu)
    - [Keras + Kaggle TPU -\> Hugging Face](#keras--kaggle-tpu---hugging-face)
    - [苹果 MLX 框架](#苹果-mlx-框架)
    - [总结](#总结)
- [参考](#参考)
</p></details><p></p>

# Gemma 模型介绍

> Gemma: Google 最新推出开源大语言模型(Google's new open LLM)

2024 年 2 月 22 日，Google 发布了一系列最新的开放式大型语言模型 —— Gemma！
Gemma 提供两种规模的模型，每种规模的模型都包含基础版本和经过指令调优的版本：

* 7B 参数模型，针对消费级 GPU 和 TPU 设计，确保高效部署和开发；
* 2B 参数模型则适用于 CPU 和移动设备。

Gemma 是 Google 基于 Gemini 技术推出的四款新型大型语言模型（LLM），提供了 2B 和 7B 两种不同规模的版本，每种都包含了预训练基础版本和经过指令优化的版本。所有版本均可在各类消费级硬件上运行，无需数据量化处理，拥有高达 8K tokens 的处理能力：

* gemma-7b：7B 参数的基础模型。
* gemma-7b-it：7B 参数的指令优化版本。
* gemma-2b：2B 参数的基础模型。
* gemma-2b-it：2B 参数的指令优化版本。

在 Hagging Face Hub 上，你可以找到这四个公开可访问的模型（包括两个基础模型和两个经过调优的模型）。此次发布的亮点包括：

* Hugging Face Hub 上的模型，包括模型说明和授权信息
* Hugging Face Transformers 的集成
* 与 Google Cloud 的深度集成
* 与推理端点 (Inference Endpoints) 的集成
* 使用 Hugging Face TRL 在单个 GPU 上对 Gemma 进行微调的示例

## Prompt 提示词格式

Gemma 的基础模型不限定特定的提示格式。如同其他基础模型，
它们能够根据输入序列生成一个合理的续接内容，适用于零样本或少样本的推理任务。
这些模型也为针对特定应用场景的微调提供了坚实的基础。
指令优化版本则采用了一种极其简洁的对话结构：

```xml
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn>
```

要有效利用这一格式，必须严格按照上述结构进行对话。
我们将演示如何利用 `transformers` 库中提供的聊天模板简化这一过程。

## 探索未知领域

尽管技术报告提供了关于基础模型训练和评估过程的信息，
但关于数据集构成和预处理的具体细节则较为欠缺。
据悉，这些模型是基于来自互联网文档、编程代码和数学文本等多种数据源训练而成，
经过严格筛选，以排除含有敏感信息和不适内容的数据。

对于 <span style='border-bottom:1.5px dashed red;'>Gemma 的指令优化模型</span>，
关于 <span style='border-bottom:1.5px dashed red;'>微调数据集</span> 以及与 <span style='border-bottom:1.5px dashed red;'>顺序微调技术（SFT）</span> 和 <span style='border-bottom:1.5px dashed red;'>基于人类反馈的强化学习（RLHF）</span> 相关的超参数设置，细节同样未公开。

## 演示

### Gemma 指令模型互动对话体验

可以在 Hugging Chat 上体验与 Gemma 指令模型的互动对话！点击此处访问：

* https://hf.co/chat?model=google/gemma-7b-it

### 使用 Hugging Face Transformers

借助 Transformers 的 4.38 版本，你可以轻松地使用 Gemma 模型，并充分利用 Hugging Face 生态系统内的工具，包括：

* 训练和推理脚本及示例
* 安全文件格式（`safetensors`）
* 集成了诸如 bitsandbytes（4位量化）、PEFT（参数效率微调）和 Flash Attention 2 等工具
* 辅助工具和帮助器，以便使用模型进行生成
* 导出模型以便部署的机制

另外，Gemma 模型支持 `torch.compile()` 与 CUDA 图的结合使用，
在推理时可实现约 4 倍的速度提升！

1. 确保使用的是最新版本的 `transformers`：

```bash
$ pip install -U "transformers==4.38.1" --upgrade
```

2. 以下代码片段展示了如何结合 `transformers` 使用 `gemma-7b-it`。
   运行此代码需大约 18 GB 的 RAM，适用于包括 3090 或 4090 在内的消费级 GPU。

```python
import torch
from transformers import AutoTokenizer, pipline

# ? model load
model = "google/gemma-7b-it"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# pipeline
pipeline = pipeline(
    "text-generation",
    model = model,
    model_kwargs = {
        "torch_dtype": torch.bfloat16
    },
    device = "cuda",
)

# TODO
messages = [
    {
        "role": "user",
        "content": "Who are you? Please, answer in pirate-speak.",
    }
]

# prompt
prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenizer = False,
    add_generation_prompt = True,
)

# outputs
outputs = pipeline(
    prompt,
    max_new_tokens = 256,
    add_special_tokens = True,
    do_sample = True,
    temperature = 0.7,
    top_k = 50,
    top_p = 0.95,
)

print(outputs[0]["generated_text"][len(prompt):])
```

```
Avast me, me hearty. I am a pirate of the high seas, ready to pillage and plunder. Prepare for a tale of adventure and booty!
```

简单介绍一下这段代码:

* 代码段展示了如何利用 `bfloat16` 数据类型进行模型推理，该数据类型是所有评估中使用的参考精度。
  如果你的硬件支持，使用 float16 可能会更快。
* 还可以将模型自动量化，以 8 位或 4 位模式加载。以 4 位模式加载模型大约需要 9 GB 的内存，
  使其适用于多种消费级显卡，包括 Google Colab 上的所有 GPU。
  以下是以 4 位加载生成 pipeline 的方法：

```python
pipeline = pipeline(
    "text-generation",
    model = model,
    model_kwargs = {
        "torch_dtype": torch.float16,
        "quantization_config": {
            "load_in_4bit": True
        }
    },
    device = "cuda",
)
```

更多关于如何使用 transformers 和模型的详情，请参阅模型卡片。

### JAX 权重

所有 Gemma 模型变种都可以用 PyTorch 或 JAX / Flax 使用。
若要加载 Flax 权重，你需要按照以下方式使用仓库中的 `flax` 修订版本：

```python
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxGemmaForCausalLM

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"

model, params = FlaxGemmaForCausalLM.from_pretrained(
    model_id,
    dtype = jnp.bfloat16,
    revision = "flax",
    _do_init = False,
)

inputs = tokenizer(
    "Valencia and Málaga are", 
    return_tensors = "np", 
    padding = True,
)
output = model.generate(
    input,
    params = params, 
    max_new_tokens = 20, 
    do_sample = False,
)
output_text = tokenizer.batch_decode(
    output.sequences,
    skip_special_tokens = True,
)
```

```
['Valencia and Málaga are two of the most popular tourist destinations in Spain. Both cities boast a rich history, vibrant culture,']
```

如果你在 TPU 或多个 GPU 设备上运行，可以利用 `jit` 和 `pmap` 来编译和并行执行推理任务。

## 与 Google Cloud 集成

你可以通过 Vertex AI 或 Google Kubernetes Engine (GKE) 在 Google Cloud 上部署和训练 Gemma，利用 文本生成推理 和 Transformers 实现。

要从 Hugging Face 部署 Gemma 模型，请访问模型页面并点击部署 -> Google Cloud。
这将引导你进入 Google Cloud Console，在那里你可以通过 Vertex AI 或 GKE 一键部署 Gemma。
文本生成推理为 Gemma 在 Google Cloud 上的部署提供支持，
你也可以通过 Vertex AI Model Garden 直接访问 Gemma。

要在 Hugging Face 上微调 Gemma 模型，请访问 模型页面 并点击 训练 -> Google Cloud。
这将引导你进入 Google Cloud Console，在那里你可以在 Vertex AI 或 GKE 上访问笔记本，
以在这些平台上微调 Gemma。

## 与推理端点集成

可以在 Hugging Face 的 推理端点 上部署 Gemma，该端点使用文本生成推理作为后端。
文本生成推理 是由 Hugging Face 开发的可用于生产环境的推理容器，旨在简化大型语言模型的部署。
它支持连续批处理、令牌流式传输、多 GPU 张量并行加速推理，并提供生产就绪的日志记录和跟踪功能。

要部署 Gemma 模型，请访问 HF Hub 模型页面 并点击 部署 -> 推理端点。
有关 使用 Hugging Face 推理端点部署 LLM的更多信息，请参阅我们之前的博客文章。
推理端点通过文本生成推理支持 消息 API，
使你可以通过简单地更换 URL 从其他封闭模型切换到开放模型。

```python
from openai import OpenAI

# initialize the client but point it to TGI
client = OpenAI(
    base_url = "<ENDPOINT_URL>" + "/v1/",  # replace with your endpoint url
    api_key = "<HF_API_TOKEN>",  # replace with your token
)
chat_completion = client.chat.completions.create(
    model="tgi",
    messages = [
        {
            "role": "user", 
            "content": "Why is open-source software important?"
        },
    ],
    stream = True,
    max_tokens = 500
)

# iterate and print stream
for message in chat_completion:
    print(message.choices[0].delta.content, end = "")
```

## 使用 Hugging Face TRL 进行微调

在消费级 GPU 上训练大型语言模型既是技术上的挑战，也是计算上的挑战。
本节将介绍 Hugging Face 生态系统中可用的工具，
这些工具可以帮助你高效地在消费级 GPU 上训练 Gemma。

一个微调 Gemma 的示例命令如下。我们利用 4 位量化和 QLoRA（一种参数效率微调技术）来减少内存使用，
目标是所有注意力块的线性层。值得注意的是，与密集型 Transformer 不同，
MLP 层（多层感知器层）因其稀疏性不适合与 PEFT（参数效率微调）技术结合使用。

首先，安装 Hugging Face TRL 的最新版本并克隆仓库以获取 训练脚本：

```bash
$ pip install -U transformers
$ pip install git+https://github.com/huggingface/trl
$ git clone https://github.com/huggingface/trl
$ cd trl
```

然后运行脚本：

```bash
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml --num_processes=1 \
    examples/scripts/sft.py \
    --model_name google/gemma-7b \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --save_steps 20_000 \
    --use_peft \
    --peft_lora_r 16 --peft_lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --load_in_4bit
```

在单个 A10G GPU 上，这个训练过程大约需要 9 小时。
通过调整 `--num_processes` 参数为你可用的 GPU 数量，
可以实现并行化训练，从而缩短训练时间。

### 额外资源

* Hub 上的模型
* 开放 LLM 排行榜
* Hugging Chat 上的聊天演示
* Gemma 官方博客
* Gemma 产品页面
* Vertex AI 模型花园链接
* Google Notebook 教程

# Gemma 模型中文指令微调

谷歌在 2 月 21 日放出开放权重的 Gemma 系列大模型，包括 2B 和 7B 两个大小，
并且有预训练和指令微调两个版本。虽然 Gemma 的预训练数据里面包含多种语言，
不过在官方的技术报告里，明确指出了做指令微调的时候只用了英文。

经过英文指令微调的 Gemma 模型，仍然保留一定程度的指令跟随能力，可以理解一部分中文指令，
但有些时候我们未必希望使用官方的指令微调模型，而是希望将预训练模型重新进行中文指令微调，
来达到我们的要求，所以在这里我们就分享 3 个方法来进行 Gemma 的中文指令微调。

注意在这里我们统一使用 [gemma-2b](https://hf.co/google/gemma-2b-it) 这个模型，
同时我们选择了 [Hello-SimpleAI/HC3-Chinese](https://hf.co/datasets/Hello-SimpleAI/HC3-Chinese) 数据集来作为微调数据。
这个数据集有不同主题的问答，包括问题，人类回答和 ChatGPT 回答，
涵盖了金融，百科，法律等等诸多题材，我们只使用 `baike` 这个子集，
且不使用 ChatGPT 回答。

## Hugging Face TRL + Colab GPU





## Keras + Kaggle TPU -> Hugging Face

> Keras 使用 JAX 后端


## 苹果 MLX 框架



## 总结

这里介绍了 3 个针对谷歌 gemma-2b 模型进行中文指令微调的简单方法，
这里展示的所有的代码都在 GitHub 上，并且配有有 2 个简短的视频讲解。

* 本文展示的代码
    - https://github.com/windmaple/Gemma-Chinese-instruction-tuning
* 两个讲解视频
    - https://www.bilibili.com/video/BV14x4y1C7wi/
    - https://www.bilibili.com/video/BV1YH4y177t9/

当然这里我们使用的数据，模型和算力都是比较小的，
所以完成以后的模型性能肯定不能和最先进的模型相比，
我们更重要的是分享思路方便大家学习。
如果有足够资源的同学可以自己去探索收集更多的中文数据，
使用 7b 模型，使用多 GPU/TPU 分布式训练等途径来打造更强大的自由模型。







# 参考

* [Gemma: Google 最新推出开源大语言模型](https://mp.weixin.qq.com/s?__biz=Mzk0MDQyNTY4Mw==&mid=2247490714&idx=1&sn=fd6f17929d52e11efe98b2e254ead7aa&chksm=c2e0b426f5973d3014a0f46c0f1d17aefb045031b2c244aa7081a6e41e38d654cb1023acf68d&scene=21#wechat_redirect)

