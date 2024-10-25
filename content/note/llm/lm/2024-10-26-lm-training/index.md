---
title: 语言模型训练
author: wangzf
date: '2024-10-26'
slug: lm-training
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

- [目标函数](#目标函数)
    - [Decoder-only 模型](#decoder-only-模型)
    - [Encoder-only 模型](#encoder-only-模型)
        - [单向到双向](#单向到双向)
        - [BERT](#bert)
        - [RoBERTa](#roberta)
    - [Encoder-Decoder 模型](#encoder-decoder-模型)
- [优化算法](#优化算法)
    - [SGD](#sgd)
    - [Adam](#adam)
    - [AdaFactor](#adafactor)
    - [混合精度训练](#混合精度训练)
    - [学习率](#学习率)
    - [初始化](#初始化)
- [分布式训练](#分布式训练)
</p></details><p></p>

# 目标函数

三类语言模型的目标函数：

* Decoder-only 模型（例如，GPT-3）：计算单向上下文嵌入(contextual embeddings)，一次生成一个 token
* Encoder-only 模型（例如，BERT）：计算双向上下文嵌入
* Encoder-Decoder 模型（例如，T5）：编码输入，解码输出

可以使用任何模型（例如，LSTM、Transformers）将 token 序列映射到上下文嵌入(contextual embeddings)中：

`$$\phi: V^{L} \rightarrow \mathbb{R}^{d\times L}$$`

`$$[\text{the, mouse, ate, the, cheese}]^{5} \stackrel{\phi}{\Rightarrow} 
 \left[
 \left(\begin{array}{c}1 \\ 0.1\end{array}\right),
 \left(\begin{array}{l}0 \\ 1\end{array}\right),
 \left(\begin{array}{l}1 \\ 1\end{array}\right),
 \left(\begin{array}{c}1 \\ -0.1\end{array}\right),
 \left(\begin{array}{c}0 \\ -1\end{array}\right)
 \right]^{2 \times 5}$$`

## Decoder-only 模型

自回归语言模型定义了一个条件分布：

`$$p(x_{i}|x_{1:i-1})$$`

将其定义如下：

* 将 `$x_{1:i-1}$` 映射到上下文嵌入 `$\phi(x_{1:i-1})$`；
* 应用嵌入矩阵 `$E \in \mathbb{R}^{V\times d}$` 来获得每个 token 的得分 `$E \phi(x_{1:i-1})_{i-1}$`；
* 对其进行指数化和归一化，得到预测 `$x_{i}$` 的分布。

简洁地：

`$$p(x_{i+1}|x_{1:i}) = \text{Softmax}(E \phi(x_{1:i})_{i})$$`

最大似然函数：

设 `$\theta$` 是大语言模型地所有参数，设 `$D$` 是由一组序列组成的训练数据。
可以根据最大似然原理，定义以下负对数似然目标函数：

`$$\begin{align}
O(\theta) 
&= -\sum_{x\in D}\log p_{\theta}(x) \\
&= -\sum_{x\in D}\sum_{i=1}^{L}\log p_{\theta}(x_{i}|x_{1:i-1})
\end{align}$$`

有很多的方法可以有效地优化这个目标函数。

## Encoder-only 模型

### 单向到双向

使用上述最大似然可以训练得到 Decoder-only 模型，它会产生（单向）上下文嵌入。
但如果不需要生成，可以提供更强的双向上下文嵌入。

### BERT

首先介绍 BERT 的目标函数，它包含以下两个部分：

* 掩码语言模型(Masked language modeling)
* 下一句预测(Next sentence prediction)

以自然语言推理（预测隐含、矛盾或中性）任务中的序列为例：

`$$x_{1:L}=[[\text{CLS}], \text{all, animals, breathe}, [\text{SEP}], \text{cats, breathe}]$$`

其中有两个特殊的 token：

* `[CLS]`：包含用于驱动分类任务的嵌入；
* `[SEP]`：用于告诉模型第一个序列（例如，前提）与第二个序列（例如，假设）的位置。

**BERT 模型定义为：**

`$$\text{BERT}(x_{1:L})=\text{TransformerBlock}^{24}( \\
\text{EmbedTokenWithPosition}(x_{1:L}) +
\text{SentenceEmbedding}(x_{1:L})) \in \mathbb{R}^{d\times L}$$`

其中，`$\text{SentenceEmbedding}(x_{1:L})$` 根据序列返回以下两个矢量之一：

* 对于 `$[\text{SEP}]$` 左边的，返回 `$e_{A}\in \mathbb{R}^{d}$`
* 对于 `$[\text{SEP}]$` 右边的，返回 `$e_{B}\in \mathbb{R}^{d}$`

![img](images/bert.png)

BERT-large 有 `$n_{\text{heads}}=16$` 个 Attention 头，
并且 `$d_{\text{model}}=1024$`，总共 355M 个参数。

**掩码语言模型：**

掩码语言模型的基本思想是通过加噪然后预测来进行训练：

```
[the, [MASK], ate, [MASK], cheese] => [the, mouse, ate, the, cheese]
```

更普遍地说，可以将其视为类似于去噪自动编码器，
其中映射有噪声/不完整版本 `$\tilde{x}_{1:L}$`，​并尝试重建原始 `$x_{1:L}$`。

`$$\tilde{x}_{1:L} \Rightarrow x_{1:L}$$`

* 建模：首先定义模型分布。给定输入 `$\tilde{x}_{1:L}$` 及其上下文嵌入，模型独立地预测每个 token：

    `$$P(x_{i}|\tilde{x}_{1:L}) = \text{Softmax}(E \phi(\tilde{x}_{1:L})_{i})$$`

* 掩码：定义了一个（随机）噪声函数 `$A(\tilde{x}_{1:L}|x_{1:L})$`：

    `$$\underbrace{x_{1:L}}_{\text{original}} \stackrel{A}{\Rightarrow} \underbrace{\tilde{x}_{1:L}}_{\text{noised}}$$`

    以下是 `$A$` 的定义：

    - 假设 `$I \subset \{1, \cdots, L\}$` 代表所有位置中随机的 15%；
    - 对于每个 `$i \in I$`：
        - 以 0.8 的概率，`$\tilde{x}_{i} \leftarrow [\text{MASK}]$`
        - 以 0.1 的概率，`$\tilde{x}_{i} \leftarrow x_{i}$`
        - 以 0.1 的概率，`$\tilde{x}_{i} \leftarrow \text{random word from} V$`

* 减少分布偏移：如果总是使用 `[MASK]` 来替换 `$I$` 中选定的 token，则：
    - 在训练期间，输入到 BERT 的都是带 `[MASK]` 的序列。
    - 而在测试时，我们会输入没有 `[MASK]` 的句子，这将导致分布发生变化。
      一种启发式的解决方法是在 20% 的时间内(此处指训练的时间)用真实单词替换。

**下一句预测：**

回想一下，BERT 是在拼接好的成对句子上训练的。下一句预测的目标是预测第二句是否跟随第一句。

```
[[CLS], the, mouse, ate, the, cheese, [SEP], it, was, full] => 1
[[CLS], the, mouse, ate, the, cheese, [SEP], hello, world] => 0
```

然后使用 `[CLS]` 的嵌入来做二分类。

**数据集：**

`$D$` 是按如下方式构造的一组样本 `$(x_{1:L}, c)$`：

* 令 `$A$` 是语料库中的一个句子；
* 以 0.5 的概率，`$B$` 是下一句话；
* 以 0.5 的概率，`$B$` 是语料库中的一个随机句子；
* 令 `$x_{1:L} = [[\text{CLS}], A, [\text{SEP}], B]$`；
* 令 `$c$` 表示 `$B$` 是否是下一句。

**训练目标：**

BERT 的训练目标是：

`$$O(\theta) = \sum_{(x_{1:L} c) \in D} \underbrace{E_{I, \tilde{x}_{1:L} \sim A(\cdot|x_{1:L} I)}\Bigg[-\sum_{i \in I}\log p_{\theta}(\tilde{x}_{i}|x_{1:L})\Bigg]}_{\text{masked language modeling}} + \underbrace{-\log p(c|\phi(x_{1:L})_{1})}_{\text{next sentence prediction}}$$`

稍后将讨论训练，这里简要总结一下 BERT：

* BERT（以及 ELMo 和 ULMFiT）表明，一个统一的体系结构（Transformer）可以用于多个分类任务；
* BERT 真正将 NLP 社区转变为 **预训练+微调** 的范式；
* BERT 显示了深度双向上下文嵌入的重要性，尽管通过模型大小和微调策略可能会弥补这一点，
  比如：[p-tuning](https://arxiv.org/pdf/2103.10385)。

### RoBERTa

RoBERTa 对 BERT 进行了以下改进：

* 删除了下一句预测这一目标函数（发现它没有帮助）；
* 使用更多数据训练（16GB 文本 `$\Rightarrow$` 160GB 文本）；
* 训练时间更长；
* RoBERTa 在各种基准上显著提高了 BERT 的准确性（例如，在 SQuAD 上由 81.8 到 89.4）。

## Encoder-Decoder 模型





# 优化算法

## SGD

> 随机梯度下降

## Adam

> Adaptive Moment Estimation

## AdaFactor

## 混合精度训练

## 学习率

## 初始化

# 分布式训练

