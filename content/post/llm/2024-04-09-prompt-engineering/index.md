---
title: LLM 架构--提示工程
subtitle: Prompt Engineering
author: 王哲峰
date: '2024-04-09'
slug: prompt-engineering
categories:
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

- [提示工程概览](#提示工程概览)
  - [提示工程简介](#提示工程简介)
  - [模型设置](#模型设置)
- [提示技术](#提示技术)
- [提示应用](#提示应用)
- [Prompt Hub](#prompt-hub)
- [模型](#模型)
- [风险和误用](#风险和误用)
- [LLM 研究](#llm-研究)
- [参考](#参考)
</p></details><p></p>

# 提示工程概览

## 提示工程简介

提示工程（Prompt Engineering）是一门较新的学科，关注提示词开发和优化，
帮助用户将大语言模型（Large Language Model, LLM）用于各场景和研究领域。
掌握了提示工程相关技能将有助于用户更好地了解大型语言模型的能力和局限性。

* 研究人员可利用提示工程来提升大语言模型处理复杂任务场景的能力，如问答和算术推理能力。
* 开发人员可通过提示工程设计、研发强大的工程技术，实现和大语言模型或其他生态工具的高效接轨。

提示工程不仅仅是关于设计和研发提示词。它包含了与大语言模型交互和研发的各种技能和技术。
提示工程在实现和大语言模型交互、对接，以及理解大语言模型能力方面都起着重要作用。
用户可以通过提示工程来提高大语言模型的安全性，也可以赋能大语言模型，
比如借助专业领域知识和外部工具来增强大语言模型能力。

测试环境：

* OpenAPI Playground
* 模型：`gpt-3.5-turbo`
* 模型配置：
    - `temperature = 1`
    - `top_p = 1`

## 模型设置

使用提示词时，您会通过 API 或直接与大语言模型进行交互。你可以通过配置一些参数以获得不同的提示结果。
调整这些设置对于提高响应的可靠性非常重要，你可能需要进行一些实验才能找出适合您的用例的正确设置。
以下是使用不同 LLM 提供程序时会遇到的常见设置：

* Temperature：简单来说，temperature 的参数值越小，模型就会返回越确定的一个结果。
  如果调高该参数值，大语言模型可能会返回更随机的结果，也就是说这可能会带来更多样化或更具创造性的产出。
  我们目前也在增加其他可能 token 的权重。在实际应用方面，对于质量保障（QA）等任务，
  我们可以设置更低的 temperature 值，以促使模型基于事实返回更真实和简洁的结果。
  对于诗歌生成或其他创造性任务，你可以适当调高 temperature 参数值。
* Top_p：同样，使用 top_p（与 temperature 一起称为核采样的技术），可以用来控制模型返回结果的真实性。
  如果你需要准确和事实的答案，就把参数值调低。如果你想要更多样化的答案，就把参数值调高一些。
  一般建议是改变 Temperature 和 Top P 其中一个参数就行，不用两个都调整。
* Max Length：您可以通过调整 max length 来控制大模型生成的 token 数。
  指定 Max Length 有助于防止大模型生成冗长或不相关的响应并控制成本。
* Stop Sequences：stop sequence 是一个字符串，可以阻止模型生成 token，
  指定 stop sequences 是控制大模型响应长度和结构的另一种方法。
  例如，您可以通过添加 “11” 作为 stop sequence 来告诉模型生成不超过 10 个项的列表。
* Frequency Penalty：frequency penalty 是对下一个生成的 token 进行惩罚，
  这个惩罚和 token 在响应和提示中出现的次数成比例，frequency penalty 越高，
  某个词再次出现的可能性就越小，这个设置通过给 重复数量多的 Token 设置更高的惩罚来减少响应中单词的重复。
* Presence Penalty：presence penalty 也是对重复的 token 施加惩罚，但与 frequency penalty 不同的是，
  惩罚对于所有重复 token 都是相同的。出现两次的 token 和出现 10 次的 token 会受到相同的惩罚。
  此设置可防止模型在响应中过于频繁地生成重复的词。如果您希望模型生成多样化或创造性的文本，
  您可以设置更高的 presence penalty，如果您希望模型生成更专注的内容，您可以设置更低的 presence penalty。

与 `temperature` 和 `top_p` 一样，一般建议是改变 `frequency penalty` 和 `presence penalty` 其中一个参数就行，
不要同时调整两个。


# 提示技术

# 提示应用

# Prompt Hub



# 模型

# 风险和误用


# LLM 研究


# 参考

* [Prompt Learning](https://zhuanlan.zhihu.com/p/442486331)
* [提示工程指南](https://www.promptingguide.ai/zh)
