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

- [LLM 模型介绍](#llm-模型介绍)
  - [LLM 模型对比](#llm-模型对比)
  - [Prompt 提示词格式](#prompt-提示词格式)
</p></details><p></p>

# LLM 模型介绍

## LLM 模型对比

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

## Prompt 提示词格式


