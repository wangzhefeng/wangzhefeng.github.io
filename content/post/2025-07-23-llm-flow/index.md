---
title: LLM 四个阶段
author: wangzf
date: '2025-07-23'
slug: llm-flow
categories:
  - AI
tags:
  - note
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

- [随机初始化的 LLM](#随机初始化的-llm)
- [预训练 Pre-training](#预训练-pre-training)
- [指令微调 Instruction Fine-tuning](#指令微调-instruction-fine-tuning)
- [偏好微调 Preference Fine-tuning, PFT](#偏好微调-preference-fine-tuning-pft)
- [推理微调 Reasoning Fine-tuning](#推理微调-reasoning-fine-tuning)
</p></details><p></p>

从零开始训练大语言模型（LLM），使其能够应用于真实场景的四个阶段，包括以下内容：

* 预训练（Pre-training）
* 指令微调（Instruction Fine-tuning）
* 偏好微调（Preference Fine-tuning）
* 推理微调（Reasoning Fine-tuning）

<video src="images/llm-flow.mp4"></video>
<!-- <iframe height=500 width=500 src="images/llm-flow.mp4"> -->

以下是各阶段的详细解释：

## 随机初始化的 LLM

<video src="images/random-initialized-llm.mp4"></video>

在最初，模型是完全随机初始化的，没有任何知识。
这时如果你问它“什么是 LLM？”，它可能会胡言乱语，比如“try peter hand and hello 448Sn”。
因为它还没有接触任何数据，参数全是随机的。

## 预训练 Pre-training

<video src="images/pretraining.mp4"></video>

在这个阶段，我们通过大规模语料库训练模型进行“下一个 token 的预测”任务，从而让模型掌握语言的基本规则。
它会学到语法、常识、世界知识等。

但此时的模型并不善于对话，它只是“继续文本”，而不是理解和回应指令。

## 指令微调 Instruction Fine-tuning

<video src="images/instruction-fine-tuning.mp4"></video>

为了让模型更善于交互，我们引入指令微调。
具体是用“指令-回复（instruction-response）”的成对数据训练模型，
帮助它学习如何遵循用户的提示，如何格式化回复。

经过这一步，模型已经能做很多事情，比如：

- 回答问题
- 总结内容
- 编写代码 等等

到这一步，模型基本上：

- 吃遍了互联网上几乎所有的原始文本内容；
- 消耗了大量预算用于获取高质量的人工指令-回复标注数据。

那么我们如何进一步提升模型质量呢？

这时，就进入了强化学习（RL）的阶段。

## 偏好微调 Preference Fine-tuning, PFT

<video src="images/preference-fine-tuning.mp4"></video>

你可能见过 ChatGPT 提示你选择“你更喜欢哪个回答？”的界面。
这不仅是为了收集用户反馈，更是极其宝贵的“人类偏好数据”。

OpenAI 就是利用这类数据进行偏好微调。

流程如下：

- 用户在两个回答中选择一个更喜欢的；
- 使用这些选择训练一个“奖励模型（Reward Model）”，来预测用户的偏好；
- 然后通过强化学习算法（如 PPO）来更新 LLM 的参数。

这个过程被称为 RLHF（Reinforcement Learning with Human Feedback），
它能帮助模型在没有明确“正确答案”的情况下，更好地对齐人类意图。

## 推理微调 Reasoning Fine-tuning

<video src="images/reasoning-fine-tuning.mp4"></video>

对于数学、逻辑等推理任务来说，往往只有一个正确答案，且存在清晰的解题步骤。
在这种场景下，我们不再依赖人类偏好，而是直接使用“正确性”作为奖励信号。

这个过程叫做 基于可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards）。

具体流程：

- 模型生成某个问题的回答；
- 系统将其与已知正确答案进行比较；
- 根据是否正确，给予模型正向或负向的奖励。

GRPO（Generalized Rejection-penalized Policy Optimization） 是 DeepSeek 推出的一种代表性方法。

★ 总结：LLM 从零到实用的四个训练阶段

1. 随机初始化：模型无任何知识；
2. 预训练：学习语言结构和世界常识；
3. 指令微调：学会理解指令、格式化回复；
4. 偏好/推理微调：进一步提升对齐人类意图与推理能力。

