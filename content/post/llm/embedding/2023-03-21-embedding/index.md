---
title: Word Embedding
subtitle: Sequence Representation
author: 王哲峰
date: '2023-03-21'
slug: embedding
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

- [语言模型基础](#语言模型基础)
    - [最小语义单位 Token 与 Embedding](#最小语义单位-token-与-embedding)
        - [Token](#token)
        - [Bag Of Words](#bag-of-words)
        - [Embedding](#embedding)
    - [语言模型](#语言模型)
        - [贪心搜索和集束搜索](#贪心搜索和集束搜索)
        - [简单语言模型--N-Gram](#简单语言模型--n-gram)
        - [复杂语言模型--RNN](#复杂语言模型--rnn)
        - [最强表示架构--Transformer](#最强表示架构--transformer)
        - [生成语言模型--GPT](#生成语言模型--gpt)
        - [利器强化学习--RLHF](#利器强化学习--rlhf)
- [词汇表征](#词汇表征)
    - [文本向量化简介](#文本向量化简介)
    - [文本向量化方法](#文本向量化方法)
        - [离散表示](#离散表示)
        - [分布式表示](#分布式表示)
    - [文本向量化流程](#文本向量化流程)
- [词集和词袋模型](#词集和词袋模型)
    - [词集模型: One-Hot 表征](#词集模型-one-hot-表征)
        - [One-Hot 算法简介](#one-hot-算法简介)
        - [One-Hot 的优缺点](#one-hot-的优缺点)
        - [One-Hot 算法示例](#one-hot-算法示例)
        - [One-Hot 算法实现](#one-hot-算法实现)
    - [词袋模型](#词袋模型)
        - [词袋模型算法](#词袋模型算法)
        - [词袋模型示例](#词袋模型示例)
        - [词袋模型优缺点](#词袋模型优缺点)
        - [词袋模型实现](#词袋模型实现)
    - [Bi-gram 和 N-gram](#bi-gram-和-n-gram)
        - [Bi-gram 和 N-gram 算法简介](#bi-gram-和-n-gram-算法简介)
        - [Bi-gram 和 N-gram 算法示例](#bi-gram-和-n-gram-算法示例)
        - [Bi-gram 和 N-gram 优缺点](#bi-gram-和-n-gram-优缺点)
        - [Bi-gram 和 N-gram 算法实现](#bi-gram-和-n-gram-算法实现)
    - [TF-IDF](#tf-idf)
        - [TF-IDF 算法简介](#tf-idf-算法简介)
        - [TF-IDF 算法示例](#tf-idf-算法示例)
        - [TF-IDF 算法优缺点](#tf-idf-算法优缺点)
        - [TF-IDF 算法实现](#tf-idf-算法实现)
    - [共现矩阵](#共现矩阵)
        - [共现矩阵算法简介](#共现矩阵算法简介)
        - [共现矩阵算法示例](#共现矩阵算法示例)
        - [共现矩阵算法实现](#共现矩阵算法实现)
    - [Count Vector](#count-vector)
        - [Count Vector 算法简介](#count-vector-算法简介)
        - [Count Vector 算法示例](#count-vector-算法示例)
        - [Count Vector 算法实现](#count-vector-算法实现)
- [Word Embedding](#word-embedding)
    - [神经网络语言模型](#神经网络语言模型)
        - [NNLM 模型简介](#nnlm-模型简介)
        - [NNLM 模型解释](#nnlm-模型解释)
    - [word2vec](#word2vec)
        - [word2vec 简介](#word2vec-简介)
        - [word2vec 核心思想](#word2vec-核心思想)
        - [word2vec 优点](#word2vec-优点)
        - [word2vec 语言模型](#word2vec-语言模型)
        - [word2vec 工具](#word2vec-工具)
            - [word2vec 版本](#word2vec-版本)
            - [Gensim word2vec 示例](#gensim-word2vec-示例)
            - [训练一个 word2vec 词向量模型](#训练一个-word2vec-词向量模型)
    - [CBOW](#cbow)
        - [CBOW 模型概述](#cbow-模型概述)
        - [CBOW 模型简介](#cbow-模型简介)
        - [CBOW 模型解释](#cbow-模型解释)
    - [Word Embedding 模型](#word-embedding-模型)
        - [霍夫曼树](#霍夫曼树)
        - [层级 Softmax](#层级-softmax)
        - [负例采样](#负例采样)
    - [Skip-gram](#skip-gram)
        - [Skip-gram 模型概述](#skip-gram-模型概述)
        - [Skip-gram 简介](#skip-gram-简介)
        - [Skip-gram 模型解释](#skip-gram-模型解释)
    - [GloVe](#glove)
        - [GloVe 词向量简介](#glove-词向量简介)
        - [GloVe vs word2vec](#glove-vs-word2vec)
    - [fasttext](#fasttext)
        - [fasttext 算法简介](#fasttext-算法简介)
        - [fasttext 算法实现](#fasttext-算法实现)
- [参考](#参考)
</p></details><p></p>

# 语言模型基础

## 最小语义单位 Token 与 Embedding

### Token

首先，解释一下如何将自然语言文本表示成计算机所能识别的数字。对于一段文本来说，
要做的首先就是把它变成一个个 **Token**。可以将 Token 理解为一小块，可以是一个字，
也可以是两个字的词，或三个字的词。也就是说，给定一个句子，有多种获取不同 Token 的方式，
可以分词，也可以分字。

英文现在都使用 **子词**，子词把不在 **词表** 里的词或不常见的词拆成比较常见的片段。比如单词 `annoyingly` 会被拆分成如下两个子词，`"annoying"` 表示比较常见的片段，
`"##"` 表示和前一个 Token 是直接拼接的，没有空格。中文现在基本使用 **字+词** 的方式。

```python
["annoying", "##ly"]
```

这里不直接解释为什么这么做，但可以想一下完全的字或词的效果，
拿英文举例更直观。如果只用 26 个英文字母（词表），虽然词表很小（加上各种符号可能也就 100 多个），
但粒度太细，每个 Token（即每个字母）几乎没法表示语义；如果用词，这个粒度又有点太大，
词表很难覆盖所有词。而子词可以同时兼顾词表大小和语义表示，是一种折中的做法。中文稍微简单一些，
就是字+词，字能独立表示意义，比如“是”、“有”、“爱”；词是由一个以上的字组成的语义单位，
一般来说，把词拆开可能会丢失语义，比如“长城”、“情比金坚”。
当然，中文如果非要拆成一个个字也不是不可以，具体要看任务类型和效果。

### Bag Of Words

> 词袋模型，Bag Of Words(BOW)

当句子能够表示成一个个 Token 时，就可以用数字来表示这个句子了，
最简单的方法就是将每个 Token 用一个数字来表示，但考虑这个数字的大小其实和 Token 本身没有关系，
这种单调的表达方式其实只是一种字面量的转换，并不能表示丰富的语言信息。
稍微多想一点，因为已经有了一个预先设计好的词表，
那么是不是可以用词表中的每个 Token 是否在句子中出现来表示？
如果句子中包含某个 Token，对应位置为 1，否则为 0，
这样每句话都可以表示成长度（长度等于词表大小）相同的 1 和 0 组成的数组。
更进一步地，还可以将“是否出现”改成“频率”以凸显高频词。

事实上，在很长一段时间里，在 NLP任务中，自然语言都是用这种方法表示的，它有个名字，
叫做 **词袋模型(bag of words, BOW)**。从名字来看，词袋模型就像一个大袋子，
能把所有的词都装进来。文本中的每个词都被看作独立的。忽略词之间的顺序和语法，
只关注词出现的次数。在词袋模型中，每个文本（句子）可以表示为一个向量，向量的每个维度对应一个词，
维度的值表示这个词在文本中出现的次数。这种表示方法如下表所示，
每一列表示一个 Token，每一行表示一个句子，每个句子可以表示成一个长度（就是词表大小）固定的向量，
比如第一个句子可以表示为 `$[3, 1, 1, 0, 1, 1, 0, \cdots]$`。

|              | 爱 | 不 | 对 | 古琴 | 你 | 完 | 我 | ..... |
|--------------|----|----|----|----|----|----|----|------------|
| 对你爱爱爱不完 | 3 | 1 | 1 | 0 | 1 | 1 | 0 | |
| 我爱你        | 1 | 0 | 0 | 0 | 1 | 0 | 1 | |

这里的词表是按照拼音排序的，但这个顺序其实不重要（不妨思考一下为什么）。
另外，注意这里只显示了 7 列，也就是词表中的 7 个 Token，但实际上，
词表中的 Token 一般都在“万”这个级别。所以，上表中的省略号实际上省略了上万个 Token。

### Embedding

> 词向量, word vector
> 
> 词嵌入, word embedding

词袋模型中对文本的表示法很好，不过有两个比较明显的问题：

1. 由于词表一般比较大，导致向量维度比较高，并且比较稀疏（大量的 0），计算起来不太方便；
2. 由于忽略了 Token 之间的顺序，导致部分语义丢失。比如“你爱我”和“我爱你”的向量表示一模一样，
   但其实意思不一样。
   
于是，**词向量(词嵌入)** 出现了，它是一种稠密表示方法。简单来说，
一个 Token 可以表示成一定数量的小数（一般可以是任意多个，
专业叫法是 **词向量维度**，根据所用的模型和设定的参数而定），
一般数字越多，模型越大，表示能力越强。不过即使再大的模型，这个维度也会比词表小很多。
如下面的代码示例所示，每一行的若干（词向量维度）的小数就表示对应位置的 Token，
词向量维度常见的值有 200、300、768、1536 等。

```python
爱 [0.61048, 0.46032, 0.7194, 0.85409, 0.67275, 0.31967, 0.89993, ...]
不 [0.19444, 0.14302, 0.71669, 0.03330, 0.34856, 0.6991, 0.49111, ...]
对 [0.24061, 0.21402, 0.53269, 0.97005, 0.51619, 0.07808, 0.9278,...]
古琴 [0.21798, 0,62035, 0.09939, 0.93283, 0.24022, 0.91339, 0.6569,...]
你 [0.392, 0.13321, 0.00597, 0.74754, 0.45524, 0.23674, 0.7825,...]
完 [0.26508, 0.1003, 0.40059, 0.09404, 0.20121, 0.32476, 0.48591,...]
我 [0.07928, 0.37101, 0.94462, 0,87359, 0.59773, 0.13289, 0.22909,...]
... ...
```

这时候可能会有疑问：“句子该怎么表示？”这个问题非常关键，其实在深度 NLP(deep NLP)早期，
往往是对句子的所有词向量直接取平均（或者求和），最终得到一个和每个词向量同样大小的向量——句子向量。

这项工作最早要追溯到 Yoshua Bengio 等人于 2003 年发表的论文 "A neural probabilistic language model"，他们在训练语言模型的同时，顺便得到了词向量这个副产品。
不过，最终开始在实际中大规模应用，则要追溯到 2013 年谷歌的 Tomas Mikolov 发布的 Word2Vec。
借助 Word2Vec，可以很容易地在大量语料中训练得到一个词向量模型。也是从那时开始，深度 NLP 逐渐崭露头角成为主流。

早期的词向量都是静态的，一旦训练完就固定不变了。随着 NLP 技术的不断发展，
词向量技术逐渐演变成基于语言模型的动态表示。也就是说，当上下文不一样时，
同一个词的向量表示将变得不同。而且，句子的表示也不再是先拿到词向量再构造句子向量，
而是在模型架构设计上做了考虑。当输入句子时，模型经过一定计算后，就可以直接获得句子向量；
而且语言模型不仅可以表示词和句子，还可以表示任意文本。
类似这种将任意文本（或其他非文本符号）表示成稠密向量的方法，统称 Embedding 表示技术。
Embedding 表示技术可以说是 NLP 领域（其实也包括图像、语音、推荐等领域）最基础的技术，
后面的深度学习模型都基于此。甚至可以稍微夸张点说，深度学习的发展就是 Embedding 表示技术的不断发展。

## 语言模型

语言模型（Language Model, LM）简单来说，就是利用自然语言构建的模型。
自然语言就是我们日常生活、学习和工作中常用的文字。语言模型就是利用自然语言文本构成的，
根据给定文本，输出对应文本的模型。

### 贪心搜索和集束搜索

语言模型具体是如何根据给定文本输出对应文本的呢？方法有很多种，比如我们写好一个模板：“XX 喜欢 YY”。
如果 XX 是 我，YY 是你，那就是“我喜欢你”，反过来就是“你喜欢我”。
我们这里重点要说的是 **概率语言模型**，它的核心是概率，准确来说是下一个 Token 的概率。
这种语言模型的过程就是通过已有的 Token 预测接下来的 Token。举个简单的例子，
比如你只告诉模型“我喜欢你”这句话，当你输入“我”的时候，它就已经知道你接下来要输入“喜欢”了。
为什么？因为它的“脑子”力就只有这 4 个字。

接下来升级一下。假设我们给了模型很多资料，多到现在网上所能找到的资料都给了它。
这时候你再输入“我”，此时它大概不会说“喜欢”了。为什么呢？因为见到了更多不同的文本，
它的“脑子”里已经不只有“我喜欢你”这 4 个字了。不过，如果我们考虑的是最大概率，也就是说，
每次都只选择下一个最大概率的 Token，那么对于同样的给定输入，
我们依然会得到相同的对应输出（可能还是“喜欢你”，也可能不是，具体要看给的语料）。
对于这样的结果，语言模型看起来比较“呆”。我们把这种方法叫作 **贪心搜索(greedy search)**，
因为它只往后看一个词，只考虑下一步最大概率的词！

为了让生成的结果更加多样和丰富，语言模型都会在这个地方执行一些策略。比如让模型每一步多看几个可能的词，
而不是就看概率最大的那个词。这样到下一步时，上一步最大概率的 Token，加上这一步的 Token，
路径概率（两步概率的乘积）可能就不是最大的了。上面介绍的这种叫作 **集束搜索(beam search)**，简单来说，
就是一步多看几个词，看最终句子（比如生成到句号、感叹号或其他停止符号）的概率。
看得越多，越不容易生成固定的文本。

### 简单语言模型--N-Gram

上面介绍的两种不同搜索方法（贪心搜索和集束搜索）也叫解码策略。
当时更多被研究的还是模型本身，我们经历了从简答模型到复杂模型，
再到巨大模型的变迁过程。

简单模型就是就是把一句话拆成一个个 Token，然后统计概率，
这类模型有个典型代表：**N-Gram** 模型，它也是最简单的语言模型。
这里的 `$N$` 表示每次用到的上下文 Token 数。N-Gram 模型中的 `$N$` 通常等于 2 或 3，
等于 2 的叫 Bi-Gram，等于 3 的叫 Tri-Gram。Bi-Gram 和 Tri-Gram 的区别是，
前者的下一个 Token 是根据上一个 Token 来的，而后者的下一个 Token 是根据前两个 Token 来的。
在 N-Gram 模型中，Token 的表示是离散的，实际上就是词表中的一个单词。这种表示方式比较简单，
再加上 `$N$` 不能太大，导致难以学到丰富的上下文指示。事实上，它并没有用到深度学习和神经网络，
只是一些统计出来的概率值。假设“人工智能/让”出现了 5 次，“人工智能/是”出现了 3 次，
将它们出现的频率除以所有的 Gram 数就是概率。

训练 N-Gram 模型的过程其实是统计频率的过程。如果给定“人工智能”，
N-Gram 模型就会找基于“人工智能”下个最大概率的词，
然后输出“人工智能让”。接下来就是给定“让”，继续往下走了。
当然，我们也可以用上面提到的不同解码策略往下走。

接下来，让每个 Token 成为一个 Embedding 向量。简单解释一下在这种情况下怎么预测下一个 Token。
其实还是计算概率，但这次和刚才的稍微有点不一样。在刚才离散的情况下，
用统计出来的对应 Gram 数除以 Gram 总数就是出现概率。但是稠密向量要稍微换个方式，
也就是说，给你一个 `$d$` 维的向量（某个给定的 Token），你最后要输出一个长度为 `$N$` 的向量，
`$N$` 是词表大小，其中的每一个值都是一个概率值，表示下一个 Token 出现的概率，概率值加起来为 1。
按照贪心搜索解码策略，下一个 Token 就是概率最大的那个。写成简答的计算表达式如下：

```python
# d 维，加起来和 1 没关系，大小是 1xd，表示给定的 Token
X = [0.001, 0.002, 0.0052, ..., 0.0341]
# N 个，加起来为 1，大小是 1xN，表示下一个 Token 就是每个 Token 出现的概率
Y = [0.1, 0.5, ..., 0.005, 0.3]
# W 是模型参数，也可以叫作模型
X · W = Y
```

上面的 `$W$` 就是模型参数，其实 `$X$` 也可以被看作模型参数（自动学习到的）。
因为我们知道了输入和输出的大小，所以中间其实可以经过任意的计算，
也就是说，`$W$` 可以包含很多运算。总之各种张量（三维以上数组）运算，
只要保证最后的输出形式不变就行。各种不同的计算方式就意味着各种不同的模型。

### 复杂语言模型--RNN

在深度学习早期，最著名的语言模型就是使用循环神经网络（recurrent neural network, RNN）训练的，
RNN 是一种比 N-Gram 模型复杂得多的模型。RNN 与其他神经网络的不同之处在于，RNN 的节点之间存在循环连接，
这使得它能够记住之前的信息，并将它们应用于当前的输入。这种记忆能力使得 RNN 在处理时间序列数据时特别有用，
例如预测未来的时间序列数据、进行自然语言的处理等。通俗地讲，RNN 就像具有记忆能力的人，
它可以根据之前的经验和知识对当前的情况做出反应，并预测未来的发展趋势。

### 最强表示架构--Transformer


### 生成语言模型--GPT


### 利器强化学习--RLHF





# 词汇表征

在谈词嵌入和词向量等词汇表征方法之前，我们先来看一下将 NLP 作为监督机器学习任务时该怎样进行描述。
假设以一句话为例：`"I want a glass of orange ____."`。我们要通过这句话的其他单词来预测划横线部分的单词。
这是一个典型的 NLP 问题，将其作为监督机器学习来看的话，模型的输入是上下文单词，输出是划横线的目标单词，
或者说是目标单词的概率，我们需要一个语言模型来构建关于输入和输出之间的映射关系。应用到深度学习上，
这个模型就是循环神经网络（RNN）。

在 NLP 里面，最细粒度的表示就是词语，词语可以组成句子，句子再构成段落、篇章和文档。
但是计算机并不认识这些词语，所以我们需要对以词汇为代表的自然语言进行数学上的表征。
简单来说，我们需要将词汇转化为计算机可识别的数值形式，这种转化和表征方式目前主要有两种，
一种是传统机器学习中的 One-Hot 编码方式，另一种则是基于神经网络的词嵌入（Embedding）技术。

## 文本向量化简介

文本向量化又称为 “词向量模型”、“向量空间模型”，即将文本表示成计算机可识别的实数向量，
根据粒度大小不同，可将文本特征表示分为字、词、句子、篇章、文档几个层次。

## 文本向量化方法

### 离散表示

文本向量化离散表示是一种基于规则和统计的向量化方式，
常用的方法包括 **词集模型(Set of Word)** 和 **词袋模型(Bag of Word)**，
都是基于词之间保持独立性、没有关联为前提，将所有文本中单词形成一个字典，
然后根据字典来统计单词出现频数，不同的是:

* 词集模型(Set of Word)：统计各词在句子中是否出现。只要单个文本中单词出现在字典中，
  就将其置为 1，不管出现多少次。比如：
    - One-Hot 编码
* 词袋模型(Bag Of Words, BOW)：统计各词在句子中出现的次数。只要单个文本中单词出现在字典中，
  就将其向量值加 1，出现多少次就加多少次。比如：
    - 基于统计的词袋模型
        - Bi-Gram
        - Tri-Gram
        - N-Gram
    - 基于频率的词袋模型
        - TF-IDF
        - Co-currence Matrix(共现矩阵)
        - Count Vector

文本向量化离散表示的基本的特点是忽略了文本信息中的语序信息和语境信息，仅将其反映为若干维度的独立概念，
这种情况有着因为模型本身原因而无法解决的问题，比如主语和宾语的顺序问题，
词集词袋模型天然无法理解诸如“我为你鼓掌”和“你为我鼓掌”两个语句之间的区别。

### 分布式表示

文本向量化离散表示虽然能够进行词语或者文本的向量表示，进而用模型进行情感分析或者是文本分类之类的任务。
但其不能表示词语间的相似程度或者词语间的类比关系。比如：`beautifule` 和 `pretty` 两个词语，
它们表达相近的意思，所以希望它们在整个文本的表示空间内挨得很近。

词汇分布式表示方法的主要思想是： **用周围的词表示该词**，分布式词向量表示主要是基于神经网络的词嵌入。
词汇分布式表示最早由 Hinton 在 1986 年提出，其基本思想是：通过训练将每个词映射成 K 维实数向量(K 一般为模型中的超参数)，
通过词之间的距离(如 余弦相似度、欧氏距离)来判断它们之间的语义相似度。一般认为，词向量、文本向量之间的夹角越小，
两个词相似度越高，词向量、文本向量之间夹角的关系用下面的余弦夹角进行表示。离散表示，如 One-Hot 表示无法表示余弦关系。

`$$\cos \theta = \frac{\overrightarrow{A} \cdot \overrightarrow{B}}{|\overrightarrow{A}| \cdot |\overrightarrow{B}|}$$`

Tomas Mikolov 2013 年在 ICLR 提出用于获取 Word Vector 的论文《Efficient estimation of word representations in vector space》，即 Word2Vec，
文中简单介绍了两种训练模型 CBOW、Skip-gram，以及两种加速方法 Hierarchical Softmax、Negative Sampling。
除了 Word2Vec 之外，还有其他的文本向量化的方法。比如：doc2vec、str2vec

## 文本向量化流程

1. 为词向量模型建立词汇表
    - NLP 相关任务中最常见的第一步是创建一个 **词表库**，并把每个词顺序编号
2. 文本向量化

# 词集和词袋模型

* 词集模型
    - One-Hot 编码
* 基于统计的词袋模型
    - 词袋模型
    - Bi-gram 和 N-gram
* 基于频率的词袋模型
    - TF-IDF
    - 共现矩阵
    - Count Vector

## 词集模型: One-Hot 表征

> One-Hot Representation

### One-Hot 算法简介

熟悉机器学习中分类变量的处理方法的同学对此一定很熟悉，无序的分类变量是不能直接硬编码为数字放入模型中的，
因为模型会自动认为其数值之间存在可比性，通常对于分类变量我们需要进行 One-Hot 编码。

那么如何应用 One-Hot 编码进行词汇表征呢？假设有一个包括 10000 个单词的词汇表，
现在需要用 One-Hot 编码来对每个单词进行编码。以上面那句 `"I want a glass of orange ____."` 为例，
假设 `I` 在词汇表中排在第 3876 个，那么 `I` 这个单词的 One-Hot 表示就是一个长度为 10000 的向量，
这个向量在第 3876 的位置上为 1 ，其余位置为 0，其余单词同理，每个单词都是茫茫 0 海中的一个 1。大致如下图所示:

![img](images/one_hot.png)

可见 One-Hot 词汇表征方法最后形成的结果是一种稀疏编码结果，在深度学习应用于 NLP 任务之前，
这种表征方法在传统的 NLP 模型中已经取得了很好的效果。

### One-Hot 的优缺点

优点：

* 简单快捷

缺点:

* 容易造成维数灾难(维度爆炸)问题。随着词典规模的增大，句子构成的词袋模型的维度变得越来越大，
  矩阵也变得超稀疏，这种维度的爆增，会大大耗费计算资源
    - 10000 个单词的词汇表不算多，但对于百万级、千万级的词汇表简直无法忍受
* 不能很好地判断词汇与词汇之间的相似性，且还未考虑到词出现的频率，因而无法区别词的重要性。
  One-Hot 的基本假设是词之间的语义和语法关系是相互独立的，仅仅从两个向量是无法看出两个词汇之间的关系的，
  这种独立性不适合词汇语义的运算
    - 比如上述句子，如果已经学习到了 `"I want a glass of orange juice."`，
      但如果换成了 `"I want a glass of apple ____."`，模型仍然不会猜出目标词是 `juice`。
      因为基于 One-Hot 的表征方法使得算法并不知道 `apple` 和 `orange` 之间的相似性，
      这主要是因为任意两个向量之间的内积都为零，很难区分两个单词之间的差别和联系

### One-Hot 算法示例

1. 文本语料

```python
John likes to watch moives, Mary likes too.
John also likes to watch football games.
```

2. 基于上述两个文档中出现的单词，构建如下词典(dictionary)

```python
{
    "John": 1, 
    "likes": 2,
    "to": 3,
    "watch": 4,
    "moives": 5,
    "also": 6,
    "football": 7,
    "games": 8,
    "Mary": 9,
    "too": 10,
}
```

3. 文本 One-Hot

```python
# John likes to watch movies, Mary likes too.

John:     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
likes:    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
to:       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
watch:    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
movies:   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
also:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
football: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
games:    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Mary:     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
too:      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

```python
# John also likes to watch football games.

John:     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
likes:    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
to:       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
watch:    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
movies:   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
also:     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
football: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
games:    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
Mary:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
too:      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### One-Hot 算法实现

```python
import os
import numpy as np
import pandas as pd
import jieba

def doc2onthot_matrix(file_path):
    """
    文本向量化 One-Hot Encoding
    """
    # 读取待编码的文件
    with open(file_path, encoding = "utf-8") as f:
       docs = f.readlines()
    # 将文件每行分词, 分词后的词语放入 words 中
    words = []
    for i in range(len(docs)):
       docs[i] = jieba.lcut(docs[i].strip("\n"))
       words += docs[i]
    # 找出分词后不重复的词语, 作为词袋, 
    # 是后续 One-Hot 编码的维度, 放入 vocab 中
    vocab = sorted(set(words), key = words.index)
    # 建立一个 M 行 V 列的全 0 矩阵, M 是文档样本数, 
    # 这里是行数, V 为不重复词语数, 即编码维度
    M = len(docs)
    V = len(vocab)
    onehot = np.zeros((M, V))
    for i, doc in enumerate(docs):
       for word in doc:
            if word in vocab:
               pos = vocab.index(word)
               onehot[i][pos] = 1
    onehot = pd.DataFrame(onehot, columns = vocab)
    
    return onehot


def main():
    data_dir = ""
    corpus = os.path.join(data_dir, "corpus.txt")
    onehot = doc2onthot_matrix(corpus)
    print(onehot)

if __name__ == "__main__":
    pass
```

## 词袋模型

> 词袋，Bag of Words(BOW)

### 词袋模型算法

对于句子、篇章，常用的离散表示方法是词袋模型，词袋模型以 One-Hot 为基础，忽略词表中词的顺序和语法关系，
通过记录词表中的每一个词在该文本中出现的频次来表示该词在文本中的重要程度，解决了 One-Hot 未能考虑词频的问题。

词袋模型是最早的以词语为基本单元的文本向量化方法。词袋模型，也称为计数向量表示(Count Vectors).
文档的向量表示可以直接使用单词的向量进行求和得到

### 词袋模型示例

1. 文本语料

```python
John likes to watch movies, Mary likes too.
John also likes to watch football games.
```

2. 基于上述两个文档中出现的单词，构建如下词典(dictionary)

```python
{
    "John": 1, 
    "likes": 2,
    "to": 3,
    "watch": 4,
    "movies": 5,
    "also": 6,
    "football": 7,
    "games": 8,
    "Mary": 9,
    "too": 10,
}
```

3. 上面词典中包含 10 个单词，每个单词有唯一的索引，那么每个文本可以使用一个 10 维的向量来表示:

```python
John likes to watch movies, Mary likes too.  ->  [1, 2, 1, 1, 1, 0, 0, 0, 1, 1]
John also likes to watch football games.     ->  [1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
```

| 文本 | John | likes | to | watch | movies | also | football | games | Mary | too | 
| ---------------------------------------------|-----|----|----|----|----|----|----|----|----|----|
| John likes to watch movies, Mary likes too.  | [1, | 2, | 1, | 1, | 1, | 0, | 0, | 0, | 1, | 1] |
| John also likes to watch football games.     | [1, | 1, | 1, | 1, | 0, | 1, | 1, | 1, | 0, | 0] |

横向来看，把每条文本表示成了一个向量；纵向来看，不同文档中单词的个数又可以构成某个单词的词向量，
如: `"John"` 纵向表示成 `[1, 1]`

### 词袋模型优缺点

优点：

* 方法简单，当语料充足时，处理简单的问题如文本分类，其效果比较好

缺点：

* 数据稀疏、维度大
* 无法保留词序信息
* 存在语义鸿沟的问题

### 词袋模型实现

```python
from sklearn import CountVectorizer

count_vect = CountVectorizer(analyzer = "word")

# 假定已经读进来 DataFrame, "text"列为文本列
count_vect.fit(trainDF["text"])

# 每行为一条文本, 此句代码基于所有语料库生成单词的词典
xtrain_count = count_vect.transform(train_x)
```

## Bi-gram 和 N-gram

### Bi-gram 和 N-gram 算法简介

与词袋模型原理类似，Bi-gram 将相邻两个词编上索引，N-gram 将相邻 N 个词编上索引。

### Bi-gram 和 N-gram 算法示例

1. 文本语料

```python
John likes to watch movies, Mary likes too.
John also likes to watch football games.
```

2. 基于上述两个文档中出现的单词，构建如下词典(dictionary)

```python
{
    "John likes": 1,
    "likes to": 2,
    "to watch": 3,
    "watch movies": 4,
    "Mary likes": 5,
    "likes too": 6,
    "John also": 7,
    "also likes": 8,
    "watch football": 9,
    "football games": 10,
}
```

3. 上面词典中包含 10 组单词，每组单词有唯一的索引，那么每个文本可以使用一个 10 维的向量来表示:

```python
John likes to watch movies. Mary likes too.  -> [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
John also likes to watch football games.     -> [0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
```

| 文本 | John | likes | to | watch | movies | also | football | games | Mary | too | 
| ---------------------------------------------|-----|--- |----|----|----|----|----|----|----|----|
| John likes to watch movies, Mary likes too.  | [1, | 1, | 1, | 1, | 1, | 1, | 0, | 0, | 0, | 0] |
| John also likes to watch football games.     | [0, | 1, | 1, | 0, | 0, | 0, | 1, | 1, | 1, | 1] |

### Bi-gram 和 N-gram 优缺点

优点：

* 考虑了词的顺序

缺点：

* 词向量急剧膨胀

### Bi-gram 和 N-gram 算法实现

```python
# TODO
```

## TF-IDF

### TF-IDF 算法简介

词袋模型、Bi-gram 和 N-gram 模型都是基于计数得到的，而 TF-IDF 则是基于频率统计得到的，是对词袋模型进行修正。
TF-IDF(词频-逆文档频率，Term Frequency-Inverse Document Frequency) 是一种加权方法，
TF-IDF 在词袋模型的基础上对词出现的频次赋予 TF-IDF 权值，统计各词在文档中的 TF-IDF 值(词袋模型 + IDF 值)，
TF-IDF 的分数代表了词语在当前文档和整个语料库中的相对重要性。TF-IDF 分数由两部分组成：

TF(Term Frequency)：词语频率(词频)

`$$\text{TF}(t) = \frac{\text{词语在当前文档出现的次数}}{\text{当前文档中词语的总数}}$$`

* TF 判断的是该字/词语是否是当前文档的重要词语，但是如果只用词语出现频率来判断其是否重要可能会出现一个问题，
  就是有些通用词可能也会出现很多次，如：`a`、`the`、`at`、`in` 等，当然一般我们会对文本进行预处理时去掉这些所谓的停用词(stopwords)，
  但是仍然会有很多通用词无法避免地出现在很多文档中，而其实它们不是那么重要

IDF(Inverse Document Frequency)：逆文档频率

`$$\text{IDF}(t) = log_{e} \Bigg(\frac{\text{文档总数}}{\text{出现该词语的文档总数}}\Bigg)$$`
   
* IDF 用于判断是否在很多文档中都出现了此词语，即很多文档或所有文档中都出现的就是通用词。
  出现该词语的文档越多，IDF 越小，其作用是抑制通用词的重要性

将上述求出的 TF 和 IDF 相乘得到的分数 TF-IDF，就是词语在当前文档和整个语料库中的相对重要性。
TF-IDF 与一个词在当前文档中出现次数成正比，与该词在整个语料库中的出现次数成反比

`$$\text{TF-IDF} = \text{TF}(t) \times \text{IDF}(t)$$`

### TF-IDF 算法示例

* TODO

### TF-IDF 算法优缺点

优点:

* 简单快速，结果比较符合实际情况

缺点:

* 单纯以"词频"衡量一个词的重要性，不够全面，有时重要的词可能出现次数并不多
* 无法体现词的位置信息，出现位置靠前的词与出现位置靠后的词，都被视为重要性相同，这是不正确的

### TF-IDF 算法实现

```python
from sklearn import TfidfVectorizer

# word level tf-idf
tfidf_vect = TfidfVectorizer(
    analyzer = "word",
    token_pattern = r"\w{1,}",
    max_features = 5000
)
tfidf_vect.fit(trainDF["text"])
xtrain_tfidf = tfidf_vect.transform(train_x)

# n-gram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(
    analyzer = "word",
    token_pattern = r"\w{1,}",
    ngram_ragne = (2, 3),
    max_features = 5000
)
tfidf_vect_ngram.fit(trainDF["text"])
xtrain_tfidf_vect_ngram = tfidf_vect_ngram.transform(train_x)
```

## 共现矩阵 

> 共现矩阵，Co-currence Matrix

### 共现矩阵算法简介

首先解释下共现，即共同出现，如一个词汇在一句话中共同出现或一篇文章中共同出现。这里给共同出现的距离一个规范——窗口，
如果窗口宽度是 2，那就是在当前词的前后各两个词的范围内共同出现。可以想象，其实是一个总长为 5 的窗口依次扫过所有文本，
同时出现在其中的词就说它们共现。另外，当前词与自身不存在共现，共现矩阵实际上是对角矩阵

实际应用中，用共现矩阵的一行(列)作为某个词的词向量，其向量维度还是会随着词汇字典大小呈线性增长，
而且存储共现矩阵可能需要消耗巨大的内存。共现矩阵一般配合 PCA 或者 SVD 将其进行降维，
比如：将 `$m \times n$` 的矩阵降维为 `$m \times r$`，其中 `$r \le n$`，即将词向量的长度进行缩减

### 共现矩阵算法示例

1. 文本语料

```python
John likes to watch movies.
John likes to play basketball.
```

2. 假设上面两句话设置窗口宽度为 1，则共现矩阵如下

| 共现矩阵      | John  | likes  | to  | watch  | moives  | play  |  basketball |
|--------------|-------|--------|-----|--------|---------|-------|-------------|
| John         | 0     | 2      | 0   | 0      | 0       | 0     | 0 |
| likes        | 2     | 0      | 2   | 0      | 0       | 0     | 0 |
| to           | 0     | 2      | 0   | 1      | 0       | 1     | 0 |
| watch        | 0     | 0      | 1   | 0      | 1       | 0     | 0 |
| moives       | 0     | 0      | 0   | 1      | 0       | 0     | 0 |
| play         | 0     | 0      | 1   | 0      | 0       | 0     | 1 |
| basketball   | 0     | 0      | 0   | 0      | 0       | 1     | 0 |

### 共现矩阵算法实现

```python
# TODO
```

## Count Vector

### Count Vector 算法简介

假设有语料 C，预料中包含 D 个文档：`$\{d_{1}, d_{2}, \cdots, d_{D}\}$`。
那么，语料 C 中的 N 个不重复词构成词汇字典。然后，
根据预料文档和词汇字典构建 Count Vector matrix(计数向量矩阵) `$M_{D \times N}$`，
计数向量矩阵 `$M$` 的第 `$i(i=1, 2, \cdots, D)$` 行包含了字典中每个词在文档 `$d_{i}$` 中的频率

### Count Vector 算法示例

语料(`$D = 2$`)：

* `$d_1$`：`He is a lazy boy. She is also lazy.`
* `$d_2$`：`Neeraj is a lazy person.`

字典(`$N = 6$`)：

* `["He", "She", "lazy", "boy", "Neeraj", "person"]`

计数向量矩阵:

| Count Vector  | He  |  She | lazy | boy  | Neeraj | person |
|---------------|-----|------|------|------|--------|--------|
| `$d_1$`       | 1   | 1    | 2    | 1    | 0      | 0      |
| `$d_2$`       | 0   | 0    | 1    | 0    | 1      | 1      |

### Count Vector 算法实现

# Word Embedding

词嵌入(Word Embedding)的基本想法就是将词汇表中的每个单词表示为一个普通的向量，这个向量不像 One-Hot 向量那样都是 0 或者 1，
也没有 One-Hot 向量那样长，大概就是很普通的向量，比如长这样：`$[-0.91, 2, 1.8, -0.82, 0.65, \ldots]$`。
这样的一种词汇表示方式就像是将词嵌入到了一种数学空间里面，所以叫做词嵌入。那么如何进行词嵌入呢？
或者说如何才能将词汇表征成很普通的向量形式？这需要通过神经网络进行训练，
训练得到的网络权重形成的向量就是最终需要的东西，这种向量也叫词向量，Word2Vec 就是其中的典型技术。


## 神经网络语言模型

> 神经网络语言模型，NNLM

### NNLM 模型简介

2003 年提出了神经网络语言模型(Neural Network Language Model，NNLM)，
其用前 `$n-1$` 个词预测第 `$n$` 个词的概率，并用神经网络搭建模型。

NNLM 模型的基本结构：

![img](images/NNLM.png)

NNLM 模型目标函数：

`$$\begin{align}
L(\theta) 
&= P(\omega_{t} = i | context) \\
&= f(i, \omega_{t-1}, \omega_{t-2}, \cdots, \omega_{t- n + 1}) \\
&= g(i, C(\omega_{t-1}), C(\omega_{t-2}), \cdots, C(\omega_{t- n + 1})) \\
&= \sum_{t} log P(\omega_{t} = i|\omega_{t-1}, \omega_{t-2}, \cdots, \omega_{t- n + 1}) 
\end{align}$$`

NNLM 模型使用非对称的前向窗口，长度为 `$n-1$` ，滑动窗口遍历整个语料库求和，使得目标概率最大化，其计算量正比于语料库的大小。
同时，预测所有词的概率综合应为 1：

`$$\sum_{\omega \in \{vocabulary\}} P(\omega|\omega_{t-n+1}, \cdots, \omega_{t-1}) = 1$$`

### NNLM 模型解释

样本的一组输入是第 `$n$` 个词的前 `$n-1$` 个词的 One-Hot 表示，目标是预测第 `$n$` 个词，
输出层的大小是语料库中所有词的数量，然后 sotfmax 回归，使用反向传播不断修正神经网络的权重来最大化第 `$n$`  个词的概率。
当神经网络学得到权重能够很好地预测第 `$n$` 个词的时候，输入层到映射层，即这层，其中的权重 Matrix C 被称为投影矩阵，
输入层各个词的 Ont-Hot 表示法只在其对应的索引位置为 1，其他全为 0，在与 Matrix C 矩阵相乘时相当于在对应列取出列向量投影到映射层

`$$Matrix C = (w_{1}, w_{2}, \cdots, w_{v}) = 
\begin{bmatrix}
(\omega_{1})_{1} & (\omega_{2})_{1} & \cdots & (\omega_{v})_{1} \\
(\omega_{1})_{2} & (\omega_{2})_{2} & \cdots & (\omega_{v})_{2} \\
\vdots           & \vdots           &        & \vdots           \\
(\omega_{1})_{D} & (\omega_{2})_{D} & \cdots & (\omega_{v})_{D} \\
\end{bmatrix}$$`

此时的向量​就是原词​的分布式表示，其是稠密向量而非原来 One-Hot 的稀疏向量了

在后面的隐藏层将这 `$n-1$` 个稠密的词向量进行拼接，如果每个词向量的维度为 `$D$`，则隐藏层的神经元个数为 `$(n-1)\times D$`，
然后接一个所有待预测词数量的全连接层，最后用 Softmax 进行预测

可以看到，在隐藏层和分类层中间的计算量应该是很大的，word2vec 算法从这个角度出发对模型进行了简化。
word2vec 不是单一的算法，而是两种算法的结合：连续词袋模型(CBOW)和跳字模型(Skip-gram) 

## word2vec

### word2vec 简介

word2vec 是谷歌于 2013 年提出的一种 NLP 工具，其特点就是将词汇进行向量化，这样就可以定量的分析和挖掘词汇之间的联系。
因而 word2vec 也是词嵌入表征的一种，只不过这种向量化表征需要经过神经网络训练得到，word2vec 作为现代 NLP 的核心思想和技术之一，
有着非常广泛的影响

从深度学习的角度看，假设将 NLP 的语言模型看作是一个监督学习问题：即给定上下文词 `$X$`，输出中间词 `$Y$`，
或者给定中间词 `$X$`，输出上下文词 `$Y$`。基于输入 `$X$` 和输出 `$Y$` 之间的映射便是语言模型。
这样的一个语言模型的目的便是检查 `$X$` 和 `$Y$` 放在一起是否符合自然语言法则，更通俗一点说就是 `$X$` 和 `$Y$` 搁一起是不是人话

word2vec 训练神经网络得到一个关于输入 `$X$` 和 输出 `$Y$` 之间的语言模型，关注重点并不是说要把这个模型训练的有多好，
而是要获取训练好的神经网络权重，这个权重就是要拿来对输入词汇 `$X$` 的向量化表示。一旦拿到了训练语料所有词汇的词向量，
接下来开展 NLP 研究工作就相对容易一些了。所以，基于监督学习的思想，word2vec 便是一种基于神经网络训练的自然语言模型

### word2vec 核心思想

`word2vec` 以及其他词向量模型，都基于同样的假设：

1. 衡量词语之间的相似性，在于相邻词汇是否相识，这是基于语言学的“距离象似性”原理
2. 词汇和它的上下文构成了一个象，当从语料库当中学习得到相识或者相近的象时，它们在语义上总是相识的

### word2vec 优点

高效，Mikolov 在论文中指出一个优化的单机版本一天可以训练上千亿个词

### word2vec 语言模型

word2vec 通常有两个版本的语言模型：

* 一种是给定上下文词，需要我们来预测中间目标词，这种模型叫做：连续词袋模型(Continuous Bag-of-Wods，CBOW)
* 另一种是给定一个词语，根据这个词预测它的上下文，这种模型叫做：跳元模型(Skip-gram)

CBOW 和 Skip-gram 模型的原理示意图：

![img](images/CBOW_Skip_gram.png)

关于 CBOW 和 skip-gram 模型的更多数学细节，比如 Huffman 树、损失函数的推导等问题，
从监督学习的角度来说，word2vec 本质上是一个基于神经网络的多分类问题，当输出词语非常多时，
我们则需要一些像分级 Softmax 和负采样之类的技巧来加速训练。但从自然语言处理的角度来说，
word2vec 关注的并不是神经网络模型本身，而是训练之后得到的词汇的向量化表征。
这种表征使得最后的词向量维度要远远小于词汇表大小，所以 word2vec 从本质上来说是一种降维操作。
我们把数以万计的词汇从高维空间中降维到低维空间中，大大方便了后续的 NLP 分析任务

### word2vec 工具

#### word2vec 版本

* Google `word2vec`
    - https://github.com/dav/word2vec
* Gensim `word2vec`
    - https://pypi.python.org/pypi/gensim
* C++ 11
    - https://github.com/jdeng/word2vec
* Java 
    - https://github.com/NLPchina/Word2VEC_java

#### Gensim word2vec 示例

任务：使用中文维基百科语料库作为训练库

数据预处理：大概等待 15min 左右，得到 280819 行文本，每行对应一个网页

```python
from gensim.corpora import WikiCorpus

space = " "
with open("wiki-zh-article.txt", "w", encoding = "utf8") as f:
    wiki = WikiCorpus(
        "zhwiki-latest-pages-articles.xml.bz2", 
        lemmatize = False, 
        dictionary = {}
    )
    for text in wiki.get_texts():
        f.write(space.join(text) + "\n")
print("Finished Saved.")
```

繁体字处理：

* 目的:            
    - 因为维基语料库里面包含了繁体字和简体字，为了不影响后续分词，所以统一转化为简体字        
* 工具            
    - [opencc](https://github.com/BYVoid/OpenCC)

    ```bash
    $ opencc -i corpus.txt -o wiki-corpus.txt -c t2s.json
    ```

分词：

* 常用的分词工具为：
    - jieba
    - ICTCLAS(中科院)
    - FudanNLP(复旦)
* `word2vec` 一般需要大规模语料库(GB 级别)，这些语料库需要进行一定的预处理，
  变为精准的分词，才能提升训练效果。常用大规模中文语料库:
    - [维基百科中文语料(5.7G xml)](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)
    - [搜狗实验室的搜狗 SouGouT(5TB 网页原版)](https://www.sogou.com/labs/resource/t.php)

#### 训练一个 word2vec 词向量模型

通常而言，训练一个词向量是一件非常昂贵的事情，我们一般会使用一些别人训练好的词向量模型来直接使用，
很少情况下需要自己训练词向量，但这并不妨碍我们尝试来训练一个 word2vec 词向量模型进行试验

如何训练一个 Skip-gram 模型，总体流程是

1. 先下载要训练的文本语料
2. 然后根据语料构造词汇表
3. 再根据词汇表和 skip-gram 模型特点生成 skip-gram 训练样本
4. 训练样本准备好之后即可定义 skip-gram 模型网络结构，损失函数和优化计算过程
5. 最后保存训练好的词向量即可




## CBOW

### CBOW 模型概述

![img](images/cbow_model.png)
![img](images/cbow_model2.png)

### CBOW 模型简介

CBOW 模型的应用场景是要根据上下文预测中间的词，所以输入便是上下文词，当然原始的单词是无法作为输入的，
这里的输入仍然是每个词汇的 One-Hot 向量，输出为给定词汇表中每个词作为目标词的概率

CBOW 在 NNLM 基础上有以下几点创新：

1. 取消了隐藏层，减少了计算量
2. 采用上下文划窗而不是前文划窗，即用上下文的词来预测当前词
3. 投影层不再使用各向量拼接的方式，而是简单的求和平均

CBOW 目标函数为：

`$$J = \sum_{\omega \in corpus} P(\omega | context(\omega))$$`

可以看到，上面提到的取消隐藏层，投影层求和平均都可以一定程度上减少计算量，但输出层的数量在那里，
比如语料库有 500W 个词，那么隐藏层就要对 500W 个神经元进行全连接计算，这依然需要庞大的计算量。
word2vec 算法又在这里进行了训练优化

### CBOW 模型解释

CBOW 模型基本结构：

![img](images/CBOW.png)

可见 CBOW 模型结构是一种普通的神经网络结构。主要包括输入层、中间隐藏层、最后的输出层


下面以输入、输出样本 `$(context(w), w)$` 为例对 CBOW 模型的三个网络层进行简单说明，
其中假设 `$context(w)$` 由 `$w$` 前后各 `$c$` 个词构成。数学细节如下:

![img](images/CBOW2.png)

* 输入层: 包含 `$context(w)$` 中 2c 个词的词向量 
  
  `$$v(context(w)_{1}), v(context(w)_{2}), \cdots, v(context(w)_{2c}) \in R^{m}$$`
  
  这里，`$m$` 的含义同上表示词向量的长度

* 投影层：将输入层的 `$2c$` 个向量做求和累加，即

  `$$x_{w}=\sum_{i=1}^{2c}v(context(w)_{i}) \in R^{m}$$`

* 输出层：输出层对应一颗二叉树，它是以语料中出现过的词当叶子节点，
  以各词在语料中出现的次数当权值构造出来的 Huffman 树。在这棵 Huffman 树中，
  叶子节点共 `$N(=|D|)$` 个，分别对应词典 `$D$` 中的词，非叶子节点 `$N-1$` 个(图中标成黄色的那些节点)。

普通的基于神经网络的语言模型输出层一般就是利用 Softmax 函数进行归一化计算，这种直接 Softmax 的做法主要问题在于计算速度，
尤其是我们采用了一个较大的词汇表的时候，对大的词汇表做求和运算，Softmax 的分母运算会非常慢，直接影响到了模型性能

除了分级 Softmax 输出之外，还有一种叫做负采样的训练技巧

## Word Embedding 模型

### 霍夫曼树

霍夫曼树是一棵特殊的二叉树，了解霍夫曼树之前先给出几个定义：

* 路径长度：在二叉树路径上的分支数目，其等于路径上结点数减 1
* 结点的权：给树的每个结点赋予一个非负的值
* 结点的带权路径长度：根结点到该结点之间的路径长度与该节点权的乘积
* 树的带权路径长度：所有叶子节点的带权路径长度之和

霍夫曼树的定义为：在权为 `$\omega_{1}, \omega_{2}, \cdots, \omega_{n}$` ​的​ `$n$` 个叶子结点所构成的所有二叉树中，
带权路径长度最小的二叉树称为最优二叉树或霍夫曼树

可以看出，结点的权越小，其离树的根结点越远

### 层级 Softmax

word2vec 算法利用霍夫曼树，将平铺型 Softmax 压缩成层级 Softmax，不再使用全连接。
具体做法是根据文本的词频统计，将词频赋给结点的权

在霍夫曼树中，叶子结点是待预测的所有词，在每个子结点处，用 Sigmoid 激活后得到往左走的概率 `$p$`，
往右走的概率为 `$1-p$`。最终训练的目标是最大化叶子结点处预测词的概率。

层级 Softmax 的实现有点复杂，暂时先搞清楚大致原理

### 负例采样

> 负例采样，Negative Sampling

负例采样的想法比较简单，假如有 `$m$` 个待预测的词，每次预测的一个正样本词，其他的 `$m-1$` 个词均为负样本。
一方面正负样本数差别太大；另一方面，负样本中可能有很多不常用，或者词预测时概率基本为 0 的样本，
我们不想在计算它们的概率上面消耗资源

比如现在待预测的词有 100W 个，正常情况下，我们分类的全连接层需要 100W 个神经元，我们可以根据词语的出现频率进行负例采样，
一个正样本加上采样出的比如说 999 个负样本，组成 1000 个新的分类全连接层

采样尽量保持了跟原来一样的分布，具体做法是将 `$[0, 1]$` 区间均分为 108 份，然后根据词出现在语料库中的次数赋予每个词不同的份额

`$$len(\omega) = \frac{counter(\omega)}{\sum_{\mu \in D} counter(\mu)}$$`

然后在 `$[0, 1]$` 区间掷筛子，落在哪个区间就采样哪个样本。实际上，
最终效果证明上式中取 `$counter(\omega)$` 的 `$\frac{3}{4}$` 次方效果最好，
所以在应用汇总也是这么做的

## Skip-gram

### Skip-gram 模型概述

![img](images/skip_gram_model.png)
![img](images/skip_gram_model2.png)

### Skip-gram 简介

Skip-gram 模型的应用场景是要根据中间词预测上下文词，所以输入 `$x$` 是任意单词，
输出 `$y$` 为给定词汇表中每个词作为上下文词的概率

### Skip-gram 模型解释

Skip-gram 模型基本结构：

![img](images/Skip-gram2.png)
<!-- ![img](images/Skip-gram.png) -->

从上面的结构图可见，Skip-gram 模型与 CBOW 模型翻转，也是也是一种普通的神经网络结构，
同样也包括输入层、中间隐藏层和最后的输出层。继续以输入输出样本 `$(context(w), w)$` 为例
对 Skip-gram 模型的三个网络层进行简单说明，其中假设 `$context(w)$` 由 `$w$` 前后各 `$c$` 个词构成。数学细节如下:

* 输入层：只含当前样本的中心词 `$w$` 的词向量 `$v(w) \in R^{m}$` 
* 投影层：这是个恒等投影，把 `$v(w)$` 投影到 `$v(w)$`，因此，这个投影层其实是多余的。
  这里之所以保留投影层主要是方便和 CBOW 模型的网络结构做对比
* 输出层：和 CBOW 模型一样，输出层也是一棵 Huffman 树

Skip-gram 模型的训练方法也是基于损失函数的梯度计算，目标函数:
 
`$$L = \sum_{w \in C} log \prod_{u \in context(w)} \prod_{j=2}^{l^{u}}\{[\sigma(v(w)^{T}\theta_{j-1}^{u})]^{1-d_{j}^{u}} \cdot [1-\sigma(v(w)^{T}\theta_{j-1}^{u})]^{d_{j}^{u}} \} \\
= \sum_{w \in C} \sum_{u \in context(w)} \sum_{j=2}^{l^{u}} {(1-d_{j}^{u}) \cdot log[\sigma(v(w)^{T}\theta_{j-1}^{u})] + d_{j}^{u} \cdot log[1 - \sigma(v(w)^{T}\theta_{u}^{j-1})]}$$`

## GloVe

> 全局词向量表示，Global Vectors for Word Representation, GloVe

除了 word2vec 之外，常用的通过训练神经网络的方法得到词向量的方法还包括 
GloVe 词向量、fasttext 词向量等等

### GloVe 词向量简介
   
GloVe 词向量直译为全局的词向量表示，跟 word2vec 词向量一样本质上是基于词共现矩阵来进行处理的。
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

### GloVe vs word2vec

![img](images/GloVe_vs_word2vec.png)

## fasttext

### fasttext 算法简介

fasttext 的模型与 CBOW 类似，实际上，fasttext 的确是由 CBOW 演变而来的。
CBOW 预测上下文的中间词，fasttext 预测文本标签。与 word2vec 算法的衍生物相同，
稠密词向量也是训练神经网路的过程中得到的

![img](images/fasttext.png)

1. fasttext 的输入是一段词的序列，即一篇文章或一句话，输出是这段词序列属于某个类别的概率，所以，fasttext 是用来做文本分类任务的
2. fasttext 中采用层级 Softmax 做分类，这与 CBOW 相同。fasttext 算法中还考虑了词的顺序问题，即采用 N-gram，与之前介绍的离散表示法相同，如:
    - 今天天气非常不错，Bi-gram 的表示就是:今天、天天、天气、气非、非常、常不、不错

fasttext 做文本分类对文本的存储方式有要求:

```
__label__1, It is a nice day.
__label__2, I am fine, thank you.
__label__3, I like play football.
```

其中:

* `__label__`:为实际类别的前缀，也可以自己定义

### fasttext 算法实现

GitHub：

* https://github.com/facebookresearch/fastText

示例:

```python
import fasttext

classifier = fasttext.supervised(
    input_file, 
    output, 
    label_prefix = "__label__"
)
result = classifier.test(test_file)
print(result.precision, result.recall)
```

其中:

* `input_file`：是已经按照上面的格式要求做好的训练集 txt
* `output`：后缀为 `.model`，是保存的二进制文件
* `label_prefix`：可以自定类别前缀

# 参考

* [NLP中的文本表示方法](https://zhuanlan.zhihu.com/p/42310942)
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) 
* [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) 
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) 
* [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 
* [词向量最强面经](https://mp.weixin.qq.com/s/llMhdDosePtzqujtlqCy1w)
* [白话文讲解Word2Vec](https://mp.weixin.qq.com/s/JtqepPGCFUiHIrgZnrcJ4g)
* [秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)
* [sklearn.feature_extraction](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction)
* [自然语言处理之word2vec](https://mp.weixin.qq.com/s?__biz=MzI4ODY2NjYzMQ==&mid=2247484776&idx=1&sn=484bfa9299696f6e233227e05d0fb78c&chksm=ec3ba000db4c2916b2f79d98ea8fd5ea3326e638458af2c8d65a182bdebeda11e311649d954d&scene=21#wechat_redirect) 
* [word2vec 中的数学原理详解](https://www.cnblogs.com/peghoty/p/3857839.html)
* [深度学习word2vec学习笔记](http://www.docin.com/p-2066429827.html)
* [Word Embedding, RNN/LSTM介绍](https://zhuanlan.zhihu.com/p/361802292)
