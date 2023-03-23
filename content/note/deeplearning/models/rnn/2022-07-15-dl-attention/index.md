---
title: Attention
subtitle: Augmented
author: 王哲峰
date: '2022-07-15'
slug: dl-attention
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
</style>

<details><summary>目录</summary><p>

- [为什么需要 Attention](#为什么需要-attention)
- [Attention 机制](#attention-机制)
  - [全局对齐权重](#全局对齐权重)
  - [权重计算函数](#权重计算函数)
- [Attention 类型](#attention-类型)
  - [Soft-Attention](#soft-attention)
  - [Hard-Attention](#hard-attention)
  - [Global-Attention](#global-attention)
  - [Local-Attention](#local-attention)
  - [Self-Attention](#self-attention)
- [Attention 带来的算法改进](#attention-带来的算法改进)
- [参考](#参考)
</p></details><p></p>

seq2seq 模型虽然强大, 但如果仅仅是单一使用的话, 效果会大打折扣。
本节要介绍的注意力模型就是基于 encoder-deocder 框架下的一种模拟人类注意力直觉的一种模型

# 为什么需要 Attention

让我们从循环神经网络的老大难问题——机器翻译问题入手。我们知道，普通的用目标语言中的词语来代替原文中的对应词语是行不通的，
因为从语言到另一种语言时词语的语序会发生变化。比如英语的 “red” 对应法语的 “rouge”，英语的 “dress” 对应法语 “robe”，
但是英语的 “red dress” 对应法语的 “robe rouge”

为了解决这个问题，创造了 Encoder-Decoder 结构的循环神经网络。它先通过一个 Encoder 循环神经网络读入所有的待翻译句子中的单词，
得到一个包含原文所有信息的中间隐藏层，接着把中间隐藏层状态输入 Decoder 网络，一个词一个词的输出翻译句子。
这样子无论输入中的关键词语有着怎样的先后次序，由于都被打包到中间层一起输入后方网络，
Encoder-Decoder 网络都可以很好地处理这些词的输出位置和形式了

但是问题在于，中间状态由于来自于输入网络最后的隐藏层，一般来说它是一个大小固定的向量。
既然是大小固定的向量，那么它能储存的信息就是有限的，当句子长度不断变长，
由于后方的 Decoder 网络的所有信息都来自中间状态，中间状态需要表达的信息就越来越多。
如果句子的信息是在太多，Decoder 网络就有点把握不住了

比如现在你可以尝试把下面这句话一次性记住并且翻译成中文：

> It was the best of times, it was the worst of times, it was the age of wisdom, 
> it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, 
> it was the season of Light, it was the season of Darkness, it was the spring of hope, 
> it was the winter of despair, we had everything before us, we had nothing before us, 
> we were all going direct to Heaven, we were all going direct the other way — in short, 
> the period was so far like the present period, that some of its noisiest authorities insisted on its being received, 
> for good or for evil, in the superlative degree of comparison only. -- A Tale of Two Cities, Charles Dickens.

别说翻译了，对于人类而言，光是记住这个句子就有着不小的难度。如果不能一边翻译一边回头看，我们想要翻译出这个句子是相当不容易的。
Encoder-Decoder 网络就像我们的短时记忆一样，存在着容量的上限，在语句信息量过大时，中间状态就作为一个信息的瓶颈阻碍翻译了

可惜我们不能感受到 Encoder-Decoder 网络在翻译这个句子时的无奈。
但是我们可以从人类这种翻译不同句子时集中注意力在不同的语句段的翻译方式中受到启发，
得到循环神经网络中的 Attention 机制

# Attention 机制

现在把 Encoder 网络中的隐藏层记为 `$h^{(t)}$`，把 Decoder 网络中的隐藏层记为 `$H^{(t)}$`，第 `$t$` 个输出词 `$y^{(t)}$`。
因此，原先的 Decoder 网络中的式子就可以写作：

`$$H^{(t)} = f(H^{(t - 1)}, y^{(t - 1)})$$`




## 全局对齐权重

## 权重计算函数








# Attention 类型

* Soft-Attention
* Hard-Attention
* Global-Attention
* Local-Attention
* Self-Attention


## Soft-Attention




## Hard-Attention




## Global-Attention




## Local-Attention




## Self-Attention







# Attention 带来的算法改进


Attention机制为机器翻译任务带来了曙光，具体来说，它能够给机器翻译任务带来以下的好处：

Attention显著地提高了翻译算法的表现。它可以很好地使Decoder网络注意原文中的某些重要区域来得到更好的翻译。

Attention解决了信息瓶颈问题。原先的Encoder-Decoder网络的中间状态只能存储有限的文本信息，现在它已经从繁重的记忆任务中解放出来了，它只需要完成如何分配注意力的任务即可。

Attention减轻了梯度消失问题。Attention在网络后方到前方建立了连接的捷径，使得梯度可以更好的传递。

Attention提供了一些可解释性。通过观察网络运行过程中产生的注意力的分布，我们可以知道网络在输出某句话时都把注意力集中在哪里；而且通过训练网络，我们还得到了一个免费的翻译词典（soft alignment）！还是如下图所示，尽管我们未曾明确地告诉网络两种语言之间的词汇对应关系，但是显然网络依然学习到了一个大体上是正确的词汇对应表。




Attention 代表了一种更为广泛的运算。我们之前学习的是Attention机制在机器翻译问题上的应用，但是实际上Attention还可以使用于更多任务中。我们可以这样描述Attention机制的广义定义：

给定一组向量Value和一个查询Query，Attention是一种分配技术，它可以根据Query的需求和内容计算出Value的加权和。
Attention，在这种意义下可以被认为是大量信息的选择性总结归纳，或者说是在给定一些表示（query）的情况下，用一个固定大小的表示（ 
 ）来表示任意许多其他表示集合的方法（Key）。

# 参考

* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 
* [NLP中的Attention原理和源码解析](https://zhuanlan.zhihu.com/p/43493999)
* [图解 Attention](https://zhuanlan.zhihu.com/p/265182368)
* [Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)
