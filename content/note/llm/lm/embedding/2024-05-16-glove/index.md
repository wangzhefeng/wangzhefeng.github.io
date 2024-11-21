---
title: GloVe
author: wangzf
date: '2024-05-16'
slug: glove
categories:
  - nlp
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

- [GloVe 简介](#glove-简介)
- [GloVe 词向量简介](#glove-词向量简介)
- [GloVe vs Word2Vec](#glove-vs-word2vec)
- [参考](#参考)
</p></details><p></p>

# GloVe 简介

> 全局词向量表示，Global Vectors for Word Representation, GloVe

除了 Word2Vec 之外，常用的通过训练神经网络的方法得到词向量的方法还包括 
GloVe 词向量、fasttext 词向量等等

# GloVe 词向量简介
   
GloVe 词向量直译为全局的词向量表示，跟 Word2Vec 词向量一样本质上是基于词共现矩阵来进行处理的。
GloVe 词向量模型基本步骤如下:

1. 基于词共现矩阵收集词共现信息
    - 假设 `$X_{ij}$` 表示词汇 `$i$` 出现在词汇 `$j$` 上下文的概率，首先对语料库进行扫描，
    对于每个词汇，我们定义一个 `window_size`，即每个单词向两边能够联系到的距离，在一句话中如果一个词距离中心词越远，
    我们给与这个词的权重越低
2. 对于每一组词，都有

    `$$\omega_{i}^{T}\omega_{j} + b_{i} + b_{j} = log(X_{ij})$$`
    
    - 其中， `$\omega_{i}$` 表示中心词向量，
      `$\omega_{j}$` 表示上下文词向量，
      `$b_{i}$` 和 `$b_{j}$` 均表示上下文词的常数偏倚
3. 定义 GloVe 模型损失函数

    `$$J = \sum_{i=1}^{V}\sum_{j=1}^{V}f(X_{ij})(\omega_{i}^{T}\omega_{j} + b_{i} + b_{j} - log X_{ij})^{2}$$`

    - 其中，加权函数 `$f$` 可以帮助我们避免只学习到一个常见词的词向量模型， `$f$` 函数的选择原则在于既不给常见词(this/of/and)以过分的权重，
      也不回给一些不常见词(durion)太小的权重，参考形式如下:

    `$$\begin{split}
    f(X_{ij})= \left \{
    \begin{array}{rcl}
    (\frac{X_{ij}}{x_{max}})^{\alpha}, & & {如果 X_{ij} < x_{max}} \\
    1,                                 & & {否则}                  \\
    \end{array}
    \right.
    \end{split}$$`

4. 计算余弦相似度
    - 为了衡量两个单词在语义上的相近性，我们采用余弦相似度来进行度量。余弦相似度的计算公式如下:

    `$$CosineSimilarity(u, v)=\frac{uv}{||u||_{2} ||v||_{2}}=cos(\theta)$$`

    - 基于余弦相似度的词汇语义相似性度量:

    ![img](images/cosine_similarity.png)

5. 语义类比
    - 有了词汇之间的相似性度量之后，便可以基于此做进一步分析，
      比如要解决 `a is to b as c is to _` 这样的语义填空题，
      可以利用词汇之间的余弦相似性计算空格处到底填什么单词.

# GloVe vs Word2Vec

![img](images/GloVe_vs_word2vec.png)

# 参考

* [基于 PyTorch 实现的 Glove 词向量的实例](https://blog.csdn.net/a553181867/article/details/104837957)
* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
* [Glove Python](https://github.com/maciejkula/glove-python)
