---
title: Seq2Seq
subtitle: Encoder-Decoder
author: wangzf
date: '2022-04-05'
slug: dl-seq2seq
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

- [Seq2Seq 模型简介](#seq2seq-模型简介)
- [编码器](#编码器)
- [解码器](#解码器)
- [训练模型](#训练模型)
- [总结](#总结)
- [参考](#参考)
</p></details><p></p>


# Seq2Seq 模型简介

自然语言处理的很多应用中，输入和输出都可以是不定长序列。当输入和输出都是不定长序列时，
可以使用 **编码器—解码器(Encoder-Decoder)** 或者 **Seq2Seq 模型**。
这两个模型本质上都用到了两个循环神经网络(RNN)，分别叫做 **编码器(Encoder)** 和 **解码器(Decoder)**。

* 编码器用来分析输入序列
* 解码器用来生成输出序列

以机器翻译为例，输入可以是一段不定长的英语文本序列，输出可以是一段不定长的法语文本序列，例如

* 英语输入: `They`、`are`、`watching`、`.`
* 法语输出: `Ils`、`regardent`、`.`

![img](images/seq2seq.png)

上图描述了使用 Encoder-Decoder 将上述英语句子翻译成法语句子的一种方法:

* 序列表示特殊符号
    - 在训练数据集中，可以在每个句子后附上特殊符号 `<eos>` (end of sequence)以表示序列的终止
    - 解码器在最初时间步的输入用到了一个表示序列开始的特殊符号 `<bos>` (beginning of sequence)
* 编码器
    - 编码器每个时间步的输入依次为：英语句子中的单词、标点、特殊符号 `<eos>`
    - 使用编码器在最终时间步的隐藏状态作为输入句子的表征或编码信息
* 解码器
    - 解码器在各个时间步中使用输入句子的编码信息和上个时间步的输出以及隐藏状态作为输入 
    - 希望解码器在各个时间步能正确依次输出翻译后的：法语单词、标点和特殊符号 `<eos>`

# 编码器

编码器的作用是把一个不定长的输入序列变换为一个定长的背景变量(上下文向量, context vector) `$c$`，
并在该背景变量中编码输入序列信息。编码器可以使用循环神经网络（RNN）。

考虑批量大小为 1 的时序数据样本。假设输入序列是 `$x_{1}, \cdots, x_{T}$`，
例如 `$x_i$` 是输入句子中的第 `$i$` 个词。在时间步 `$t$`，
循环神经网络将输入 `$x_{t}$` 的特征向量 `$\boldsymbol{x}_{t}$` 和上个时间步的隐藏状态 `$\boldsymbol{h}_{t-1}$` 
变换为当前时间步的隐藏状态 `$\boldsymbol{h}_{t}$`。可以用函数 `$f$` 表达循环神经网络隐藏层的变换:

`$$\boldsymbol{h}_{t} = f(\boldsymbol{x}_{t}, \boldsymbol{h}_{t-1})$$` 

接下来，编码器通过自定义函数 `$q$` 将各个时间步的隐藏状态变换为背景变量:

`$$\boldsymbol{c} = q(\boldsymbol{h}_{1}, \boldsymbol{h}_{2}, \ldots, \boldsymbol{h}_{T})$$`

例如，当选择 `$q(\boldsymbol{h}_{1}, \ldots, \boldsymbol{h}_{T}) = \boldsymbol{h}_{T}$` 时，
背景变量是输入序列最终时间步的隐藏状态 `$h_{T}$`。

以上描述的编码器是一个单向的循环神经网络，每个时间步的隐藏状态只取决于该时间步及之前的输入子序列。
也可以使用双向循环神经网络构造编码器。在这种情况下，编码器每个时间步的隐藏状态同时取决于该时间
步之前和之后的子序列(包括当前时间步的输入)，并编码了整个序列的信息。

# 解码器

编码器输出的背景变量 `$\boldsymbol{c} = q(\boldsymbol{h}_{1}, \boldsymbol{h}_{2}, \ldots, \boldsymbol{h}_{T})$` 编码了整个输入序列 `$x_{1}, \cdots, x_{T}$` 的信息。

给定训练样本中的输出序列 `$y_{1}, y_2, \ldots, y_{T'}$`，对每个时间步 `$t'$`，
解码器输出 `$y_{t'}$` 的条件概率将基于之前的输出序列 `$y_{1}, y_2, \ldots, y_{t'-1}$` 和背景变量 `$c$`，
即 `$P(y_{t^\prime} \mid y_{1}, \ldots, y_{t^\prime-1}, \boldsymbol{c})$`。

为此，可以使用另一个循环神经网络作为解码器。在输出序列的时间步 `$t'$`，
解码器将上一时间步的输出 `$y_{t'−1}$` 以及背景变量 `$c$` 作为输入，
并将它们与上一时间步的隐藏状态 `$s_{t'-1}$` 变换为当前时间步的隐藏状态 `$s_{t'}$`。
因此，可以用函数 `$g$` 表达解码器隐藏层的变换:

`$$\boldsymbol{s}_{t^\prime} = g(y_{t^\prime-1}, \boldsymbol{c}, \boldsymbol{s}_{t^\prime-1})$$` 

有了解码器的隐藏状态后，可以使用自定义的输出层和 softmax 运算来计算 `$P(y_{t^\prime} \mid y_{1}, \ldots, y_{t^\prime-1}, \boldsymbol{c})$`，
例如，基于当前时间步的解码器隐藏状态 `$\boldsymbol{s}_{t^\prime}$`、上一时间步的输出 `$y_{t^\prime-1}$`，
以及背景变量 `$c$` 来计算当前时间步输出 `$y_{t^\prime}$` 的概率分布。

# 训练模型

根据最大似然估计，可以最大化输出序列基于输入序列的条件概率

`$$\begin{split}\begin{aligned}
P(y_{1}, \ldots, y_{T'} \mid x_{1}, \ldots, x_{T})
&= \prod_{t'=1}^{T'} P(y_{t'} \mid y_{1}, \ldots, y_{t'-1}, x_{1}, \ldots, x_{T})\\
&= \prod_{t'=1}^{T'} P(y_{t'} \mid y_{1}, \ldots, y_{t'-1}, \boldsymbol{c}),
\end{aligned}\end{split}$$` 

并得到该输出序列的损失

`$$- \log P(y_{1}, \ldots, y_{T'} \mid x_{1}, \ldots, x_{T}) = -\sum_{t'=1}^{T'} \log P(y_{t'} \mid y_{1}, \ldots, y_{t'-1}, \boldsymbol{c}).$$` 

在模型训练中，所有输出序列损失的均值通常作为需要最小化的损失函数。在上面图所描述的模型预测中，
需要将解码器在上一个时间步的输出作为当前时间步的输入。与此不同，在训练中也可以将标签
序列(训练集的真实输出序列)在上一个时间步的标签作为解码器在当前时间步的输入。这叫作 **强制教学(teacher forcing)**。

# 总结

* encoder-decoder 可以输入并输出不定长的序列
* encoder-decoder 使用了两个循环神经网络
* 在 encoder-decoder 的训练中，可以采用强制教学

# 参考

* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1409.0473)
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
