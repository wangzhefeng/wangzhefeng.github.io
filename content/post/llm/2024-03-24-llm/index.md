---
title: LLM：Large Language Model
author: 王哲峰
date: '2024-03-24'
slug: llm
categories:
  - nlp
  - deeplearning
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

- [LLM 介绍](#llm-介绍)
    - [LLM 发展历程](#llm-发展历程)
    - [LLM 对比](#llm-对比)
- [LLM 获取方式](#llm-获取方式)
    - [LLM 名单](#llm-名单)
    - [LLM 本体](#llm-本体)
- [Prompt 提示词格式](#prompt-提示词格式)
- [LLM 调用示例](#llm-调用示例)
    - [查看模型性能](#查看模型性能)
    - [下载模型](#下载模型)
    - [模型调用](#模型调用)
- [LLM 微调](#llm-微调)
- [参考](#参考)
</p></details><p></p>

# LLM 介绍

## LLM 发展历程

大模型全称大语言模型，英文 Large Language Model，缩写为 LLM。
2022 年 ChatGPT 的出现标志着人工智能正式进入大模型时代，
但在此之前大模型已经走过了很长的发展历程。

自从图灵测试提出以来，如何通过机器智能理解人类语言一直是重要的研究问题，
逐渐发展成为独立的研究领域，即自然语言处理（Natural Language Processing, NLP）。
而在自然语言处理领域，过去几十年里，语言建模一直是主要的研究方法，随着深度学习的进步，
逐渐从统计语言建模发展为神经网络建模。

近几年，随着 BERT 等模型在 NLP 的各种任务上表现出优异的性能，
预训练模型（Pre-trained language models, PLM）被广泛认为是提高机器文本处理能力的有效方法。
预训练+微调这一“组合技”，即首先在大规模通用数据集上进行预训练，
再用具体任务的少量数据进行微调的方法，在各种应用场景中广泛应用并达到了很好的效果。

在预训练模型被证明有效之后，有研究发现将模型“扩大”有助于提高性能，
其中“扩大”包含两层含义，一方面是将模型加深结构、增加参数，
另一方面是提高训练的数据量。顺着这一方向，一系列模型被提出，
其中比较有名的有谷歌的 T5（参数量 11B）和 Open AI 的 GPT2（参数量 1.5B）。
人们惊讶地发现模型变大的同时，不仅在原有能力上表现更优异，而且涌现出强悍的理解能力，
在原先表现很差的许多复杂任务上都有巨大的突破。为了将这些模型与以前的加以区分，
大模型（Large Language Model, LLM）概念被提出，通常用来指代参数量在数十亿、
百亿甚至千亿以上的模型。大家都意识到这是一条潜力巨大的发展道路，
许多机构都着手于大模型的工作，大模型时代正式开始，其发展脉络可参见下图。

![img](images/procession.png)

发展到现在，世界各地许多机构都研发了自己的大模型，形成了千帆共进、百舸争流的局面。
虽然大模型理论基础和技术细节复杂深奥，很难精通掌握，但在大模型时代，
了解如何调用常见的大模型并不困难，并且很有裨益。
接下来本文将介绍大模型的常用获取途径和调用方式。

## LLM 对比

| 模型                     | 许可证          | 商业使用            | 与训练大小[tokens] | 排行榜分数(由高到低) |
|-------------------------|----------------|--------------------|------------------|-------------------|
| LLama 2 70B Chat (参考)  | Llama 2 许可证  | :white_check_mark: | 2T              | 67.87              |
| Gemma-7B                | Gemma 许可证    | :white_check_mark: | 6T              | 63.75             |
| DeciLM-7B               | Apache 2.0     | :white_check_mark: | 未知             | 61.55             |
| PHI-2 (2.7B)            | MIT            | :white_check_mark: | 1.4T            | 61.33            |
| Mistral-7B-v0.1         | Apache 2.0     | :white_check_mark: | 未知             | 60.97             |
| Llama 2 7B              | Llama 2 许可证  | :white_check_mark: | 2T              | 54.32             |
| Gemma 2B                | Gemma 许可证    | :white_check_mark: | 2T              | 46.51             |

> LLM 排行榜特别适用于衡量预训练模型的质量，而不太适用于聊天模型。我们鼓励对聊天模型运行其他基准测试，如 MT Bench、EQ Bench 和 lmsys Arena。

# LLM 获取方式

模型的获取途径分为两部分，一部分是获取模型名单，了解模型信息，
以便挑选适合自己的模型，另一部分是获取模型本体和使用手册。

## LLM 名单

获取模型名单通常有以下几个途径：

1. 权威数据集的排行榜

国内国外有许多大模型测评数据集，大模型发布的论文中一般都会在最权威的几个数据集上测评，
因此在论文中就可以了解到这些数据集。此处推荐两个：

* 在 ICLR 上发布的 [MMLU 数据集](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)，
  主要评测英文文本的各项能力；
* 由上交、清华和爱丁堡大学共同完成的 [C-Eval 数据集](https://cevalbenchmark.com/)，主要评测中文能力。

在这些数据集的排行榜上，不仅可以看到许多模型的排名，还可以查看其参数量、
每个子任务的表现和项目链接，有助于根据需要选择合适的模型。

1. 论文

论文作为大模型发布的主要阵地，也是不可或缺的手段。想要获取全面的模型名单，
综述文章是一个很好的途径，因其一般会系统地梳理一段时间内的所有进展；
若要寻找某一具体领域或某一系列的的模型，可以从某一篇论文入手，从其引用论文中寻找。

3. 公众号

除了上述两种方法外，也可以从诸如机器之心等输出深度学习内容的公众号中了解最新的大模型。

在这里需要说明的一点是，选择模型时一定要注意自己的算力资源和任务，
因为参数量越大的模型虽然性能通常更好，但正式部署所需的算力资源也是成倍上升，
而且并不是每个任务都需要用到能力最强的大模型，中小体量的模型已经可以胜任大部分文本处理任务了。

## LLM 本体

在选定了要使用的模型后，需要寻找模型本体和使用手册。

在开源的模型中，一小部分模型在专门的官网上，通过其官网即可下载。
但是绝大部分模型可以在以下三个开源社区搜索名称即可找到：

* [魔塔社区: https://modelscope.cn/home](https://modelscope.cn/home)
* [Hugging Face: https://huggingface.com](https://huggingface.com)
* [Github: https://github.com](https://github.com) 

而对于不开源的模型，则通常需要在其官网付费购买调用次数，之后根据官网使用说明调用即可。

# Prompt 提示词格式

# LLM 调用示例

下载模型本体后即可根据使用手册调用，与模型进行对话。
此部分我们以阿里巴巴发布的参数量 14B 的千问模型 `Qwen-14B-Chat` 为例（模型名最后为 `Chat` 结尾表示从），
展示从模型下载、环境配置到正式调用的过程。

## 查看模型性能

从 `C-Eval` 公开访问的模型榜单查询到 `Qwen-14B`，排名 14，从表中即可看到模型在各方面的得分。

![img](images/c-evalmodels.png)

## 下载模型

在确定使用到 `Qwen-14B` 后，在 Hugging Face 中搜索模型名，
即可找到模型介绍界面和下载信息：https://huggingface.co/Qwen/Qwen-14B-Chat。

1. 首先我们根据使用说明进行环境配置，使用大模型的环境配置通常并不复杂，
   在终端中输入以下语句安装调用模型需要的第三方库：
  
```bash
$ pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
```

2. 环境配置完成后需要下载模型本体，
   而 Hugging Face 的一大好处就在于发布于其上的模型通常都可以通过 `transformer` 包自动下载，
   例如 `Qwen-14B-Chat` 模型只需要以下几行代码即可自动下载并加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载分词器(若本地没有则自动下载)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-14B-Chat", 
    trust_remote_code = True,
)

# 加载模型(若本地没有则自动下载)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B-Chat",
    device_map = "auto",  # device_map 参数代表模型部署的位置，auto 代表自动推断 cpu 和 gpu 个数并均匀分布，此外还可手动指定，例如"cuda:0"表示只部署在第一块显卡上
    trust_remote_code = True,
).eval()
```

但需要注意的是，运行代码时需要科学上网，否则无法连接到 Hugging Face 网站。若已经将模型手动下载到本地，
则只需要将 `AutoTokenizer.from_pretrained` 和 `AutoModelForCausalLM.from_pretrained` 两个函数中的第一个参数改为本地模型文件夹路径即可。

## 模型调用

加载模型结束后，即可通过模型的函数进行交互，以 `Qwen-14B-Chat` 为例，
我们向其打招呼 `“你好”`，代码如下。与模型对话，第一个参数传入分词器，第二个参数传入对话文本，
`history` 参数传入之前的对话历史，若没有则传入 `None`，函数返回模型回应和对话历史。

```python
response, history = model.chat(
    tokenizer, 
    "你好", 
    history = None
)
print(response)
```

代码运行后可以得到模型输出为：


```
“你好！很高兴为你提供帮助。”
```

我们成功得到了模型的回答。在此基础上我们继续与其对话，
传入指令`“给我讲一个年轻人奋斗创业最终取得成功的故事”`，
并且传入上次的对话历史。

```python
response, history = model.chat(
    tokenizer, 
    "给我讲一个年轻人奋斗创业最终取得成功的故事。", 
    history = history
)
print(response)
```

代码运行后得到以下输出：

```
“这是一个关于一个年轻人奋斗创业最终取得成功的故事。
故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。
从小，李明就立下了一个目标：要成为一名成功的企业家。
为了实现这个目标，李明勤奋学习，考上了大学。
在大学期间，他积极参加各种创业比赛，获得了不少奖项。
他还利用课余时间去实习，积累了宝贵的经验。
毕业后，李明决定开始自己的创业之路。
他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。
他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
最终，李明成功地获得了一笔投资，开始了自己的创业之路。
他成立了一家科技公司，专注于开发新型软件。
在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。
他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。”
```

可以看到，模型给出了相当完整且合理的回答。
后续即可根据自己的需要进行对话，或者从头开始重新对话。

# LLM 微调

> Fine-Tune LLMs


# 参考

* [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl)
* [大模型简介与调用方法](https://mp.weixin.qq.com/s/P6ke-B9Q40Hy0S2M5V6w3Q)
