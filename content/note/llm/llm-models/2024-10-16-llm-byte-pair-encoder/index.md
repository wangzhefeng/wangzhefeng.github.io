---
title: BPP 算法
subtitle: Byte Pair Encoder
author: 王哲峰
date: '2024-10-16'
slug: llm-byte-pair-encoder
categories:
  - llm
tags:
  - algorithm
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

- [BPE 算法简介](#bpe-算法简介)
- [BPE 算法流程](#bpe-算法流程)
    - [BPE 构建词表](#bpe-构建词表)
    - [BPE 编码](#bpe-编码)
    - [BPE 解码](#bpe-解码)
- [BPE算法优缺点](#bpe算法优缺点)
- [参考](#参考)
</p></details><p></p>

# BPE 算法简介

通常 NLP 的分词有两个最简单和直接的思路：

1. 按照空格分开（在英文里就是按照单词分开），例如 `"I have a cat"` 可以分为 `['I', 'have', 'a', 'cat']`；
2. 按字符进行分割，例如 `"I have a cat"` 可以分为 `['I', 'h', 'a', 'v', 'e', 'a', 'c', 'a' , 't']`。

但这两种都有各自的弊端。

* 第一种方法，以单词为粒度分割
    - (1) 在训练过程中会导致词汇表很庞大；
    - (2) 测试过程中可能会存在 OOV 的问题；
    - (3) 泛化能力差，模型学到的 `"old"`、`"older"`、
      `"oldest"` 之间的关系无法泛化到 `"smart"`、`"smarter"`、`"smartest"`。
* 第二种方法，以字符为粒度分割
    - 这种方式构建的词汇表会有限，并且也能解决前一种方法带来的 OOV 的问题；
    - 但是字符的粒度太细，会导致丧失单词本身具有的语义信息。

因此，在单词和字符两个粒度之间取平衡，**基于子词(subword)** 的算法被提出了，
典型的就是 **BPE 算法(BPE, Byte Pair Encoder, 字节对编码)**。

# BPE 算法流程

## BPE 构建词表

1. 确定词表大小，即 subword 的最大个数 `$V$`；
2. 在每个单词最后添加一个 `</w>`，并且统计每个单词出现的频率；
3. 将所有单词拆分为单个字符，构建出初始的词表，此时词表的 subword 其实就是字符；
4. 挑出频次最高的字符对，比如说 `t` 和 `h` 组成的 `th`，
   将新字符加入词表，然后将语料中所有该字符对融合(merge)，即所有 `t` 和 `h` 都变为 `th`。
   新字符依然可以参与后续的 merge，有点类似哈夫曼树，BPE 实际上就是一种贪心算法；
5. 重复 3，4 的操作，直到词表中单词数量达到预设的阈值 `$V$` 或者下一个字符对的频数为 1；

## BPE 编码

词表构建完成后，需要对训练语料进行编码，编码流程如下：

1. 将词表中的单词按长度大小，从长到短就行排序；
2. 对于语料中的每个单词，遍历排序好的词表，
   判断词表中的单词/子词（subword）是否是该字符串的子串，如果匹配上了，
   则输出当前子词，并继续遍历单词剩下的字符串。
3. 如果遍历完词表，单词中仍然有子字符串没有被匹配，那我们将其替换为一个特殊的子词，比如 `<unk>`。

具个例子，假设我们现在构建好的词表为

```
(“errrr</w>”, 
“tain</w>”, 
“moun”, 
“est</w>”, 
“high”, 
“the</w>”, 
“a</w>”)
```

对于给定的单词 `mountain</w>`，其分词结果为：`[moun, tain</w>]`

## BPE 解码

语料解码就是将所有的输出子词拼在一起，直到碰到结尾为 `<\w>`。举个例子，假设模型输出为：

```
["moun", "tain</w>", "high", "the</w>"]
```

那么其解码的结果为：

```
["mountain</w>", "highthe</w>"]
```

# BPE算法优缺点

* 优点
    - BPE 算法是介于字符和单词粒度之间的一种以 subword 为粒度的分词算法。
        - (1) 能够解决 OOV 问题；
        - (2) 减少词汇表大小；
        - (3) 具有一定的泛化能力。
* 缺点：
    - 是基于统计的分词算法，对语料依赖性很强，如果语料规模很小，则效果一般不佳。

# 参考

* [详解BPE算法（Bype Pair Encoder）](https://zhuanlan.zhihu.com/p/589086649)
