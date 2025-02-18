---
title: LLM 应用--相似匹配
author: wangzf
date: '2024-06-06'
slug: similarity-matching
categories:
  - llm
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

- [相似匹配-万物皆可 Embedding](#相似匹配-万物皆可-embedding)
    - [相似匹配基础](#相似匹配基础)
        - [Embedding 表示技术](#embedding-表示技术)
        - [更好的 Embedding 表示](#更好的-embedding-表示)
        - [进一步思考](#进一步思考)
    - [度量 Embedding 相似度](#度量-embedding-相似度)
- [相似匹配任务](#相似匹配任务)
    - [简单问答-以问题找问题](#简单问答-以问题找问题)
        - [QA 问题](#qa-问题)
        - [数据](#数据)
    - [任务基本流程](#任务基本流程)
</p></details><p></p>


# 相似匹配-万物皆可 Embedding

Embedding 可以用来表示一个词或一句话。Embedding 和大语言模型有什么关系？
为什么需要 Embedding？在哪里需要 Embedding？上述三个问题可以简单用一句话来概括回答：
**因为需要获取“相关”上下文**。具体来说，
NLP 领域的不少任务以及大语言模型的应用都需要一定的上下文知识，
而 Embedding 表示技术就是用来获取这些上下文的。
这一过程在 NLP 领域中也被叫做 **相似匹配**：把相关内容转成 Embedding 表示，
然后通过 Embedding 相似度来获取最相关内容作为上下文。

## 相似匹配基础

### Embedding 表示技术

对于自然语言，因为输入是一段文本，在中文里就是一个一个字，或一个一个词，业内把这个字或词叫做 Token。
如果要使用模型：

1. 拿到一段文本的第一件事，就是把这段文本 **Token 化**。当然，可以按字，也可以按词，
   或按你想要的其他方式，比如每两个字一组（Bi-Gram）或每两个词一组（Bi-Gram）。
   于是自然地就有了一个新的问题：应该怎么选择 Token 化方式？其实每种不同的方式都有优点和不足，
   英文一般用子词表示，中文以前常见的事字或词的方式，中文的大语言模型基本都使用 字+词 的方式。
   这种方式一方面能够更好地表示语义，另一方面对于没见过的词又可以用字的方式来表示，
   避免了在遇到不在词表中的词时导致的无法识别和处理的情况。例子如下：
    - 给定文本：`人工智能让世界变得更美好。`
    - 按字 Token 化：`人 工 智 能 让 世 界 变 得 更 美 好 。`
    - 按词 Token 化：`人工智能 让 世界 变得 更 美好 。`
    - 按字 Bi-Gram Token 化：`人/工 工/智 智/能 能/让 让/世 世/界 界/变 变/得 得/更 更/美 美/好 好/。`
    - 按词 Bi-Gram Token 化：`人工智能/让 让/世界 世界/变得 变得/更 更/美好 美好/。`
2. Token 化之后，第二件事就是要怎么表示这些 Token，计算机只能处理是数字，
   所以要想办法把这些 Token 变成计算机所能识别的数字才行。这里需要一个**词表**，
   将每个词映射成词表中对应位置的序号。例子如下：假设以字为粒度，那么词表就可以用一个文本文件来存储，内容如下：

   ```python
   人
   工
   智
   能
   让
   世
   界
   变
   得
   更
   美
   好
   。
   ```

   一行一个字，将每个字作为一个 Token，此时，0=我，1=们，...，以此类推，我们假设词表的大小为 N。
   这里有一点需要注意，就是词表的顺序无关紧要，不过一旦确定下来，训练好模型后就不能再随便调整了。
   这里所说的调整包括调整顺序、增加词、删除词、修改词等。如果只是调整顺序或删除词，
   则不需要重新训练模型，但需要手动将 Embedding 参数也相应地调整顺序或删除对应行。
   如果只是增、改词表，则需要重新训练模型，获取增改部分的 Embedding 参数。
3. 接下来就是将这些序号（Token ID）表示成稠密向量（Embedding 表示），背后的主要思想如下：
    - 把特征固定在某个维度 D，比如 256、300、768 等，这个不重要，总之不再是词表那么大的数字。
    - 利用自然语言文本的上下文关系学习一个由 D 个浮点数组成的稠密向量。
4. 接下来是 Embedding 的学习过程。首先初始化一个 Numpy 数组：
    
    ```python
    import numpy as np
    rng = np.random.default_rng(42)

    # 词表大小 N=16，维度 D=256
    table = rng.uniform(size = (16, 256))
    table.shape == (16, 256)
    ```
    假设词表大小为 16，维度大小为 256，初始化后，就得到了一个 16×256 大小的二维数组，
    其中的每一行浮点数就表示对应位置的 Token。接下来就是通过一定的算法和策略来调整（训练）里面的数字（更新参数）。当训练结束时，最终得到的数组就是 **词表的 Embedding 表示**，也就是 **词向量**。
    这种表示方法在深度学习早期（2014 年左右）比较流行，不过由于这个矩阵在训练好后就固定不变了，
    因此它在有些时候就不合适了。比如，“我喜欢苹果”这句话在不同的情况下可能是完全不同的意思，因为“苹果”既可以指水果，也可以指苹果手机。
5. 我们知道，句子才是语义的最小单位，相比 Token，我们其实更加关注和需要句子的表示。
   而且，如果我们能够很好地表示句子，则由于词也可以被看作一个很短的句子，表示起来自然也不在话下。
   我们还期望可以根据不同上下文动态地获得句子的表示。这中间经历了比较多的探索，
   但最终走向了在模型架构上做设计——输入一段文本，模型经过一定计算后，就可以直接获得对应的向量表示。

### 更好的 Embedding 表示

前面都将免模型当做黑盒，默认输入一段文本就会给出一个表示。但这中间其实也有不少细节，
具体来说，就是如何给出这个表示。下面介绍几种常见的方法，并探讨其中的机理。

1. 直观地看，可以借鉴词向量的思想，把这里的“词”换成“句”，模型训练完之后，就可以得到句子向量了。
   不过，稍微思考一下就会发现，其实这在本质上只是粒度更大的一种 Token 化方式，粒度太大时，
   有的问题就会更加突出。而且，这样得到的句子向量还有一个问题——无法通过句子向量获取其中的词向量，
   而在有些场景下又需要词向量。看来，此路难行。
2. 还有一种操作起来更简单的方式，就是直接对词向量取平均。无论一句话或一篇文档有多少个词，
   找到每个词的词向量，平均就好了，得到的向量大小和词向量一样。事实上，
   在深度学习 NLP 刚开始的几年，这种方式一直是主流，也出现了不少关于如何平均的方法，
   比如使用加权求和，权重可以根据词性、句法结构等设定为一个固定值。
3. 2014年，也就是谷歌公司发布 Word2Vec 后一年，差不多是同一批人提出一种表示文档的方法——Doc2Vec，
   其思想是在每句话的前面增加一个额外的將 Token 作为段落的向量表示，我们可以将它视为段落的主题。
   训练模型可采用和词向量类似的方式，但每次更新词向量参数时，需要额外更新这个段落 Token 向量。
   直观地看，就是把文档的语义都融入这个特殊的 Token 向量。不这种方法存在一个很严重的问题，
   那就是推理时，如果遇到训练数据集里没有的文档，就需要将这个文档的参数更新到模型里。这不仅不方便，而且效率也低。
4. 之后，随着深度学习进一步发展，涌现出一批模型，其中最为经典的就 TextCNN 和 RNN。
   TextCNN 的想法来自图像领域的卷积神经网络（convolutional neural network,CNN）。
   TextCNN 以若干大小固定窗口在文本上滑动，每个窗口从头滑到尾，就会得到一组浮点数特征，
   使用若干不同大小的窗口（一般取 2、3、4），就会得到若干不同的特征，
   将它们拼接起来就可以表示这段文本了。TextCNN 的表示能力其实不错，一直以来都作为基准模型使用，
   很多线上模型也用它。TextCNN 的主要问题是只利用了文本的局部特征，没有考虑全局语义。
   RNN 和它的几个变体都是时序模型，从前到后一个 Token 接一个 Token 处理。
   RNN 也有不错的表示能力，但它有两个比较明显的不足：一是比较慢，没法并行；
   二是当文本太长时效果不好。总的来说，这一时期词向量用得比较少，
   文本的表示主要通过模型架构来体现，Token 化的方式以词为主。
5. 2017 年，Transformer 横空出世，带来了迄今为止最强的特征表示方式—自注意力机制。
   模型开始慢慢变大，从原来的十万、百万级别逐渐增加到亿级别。文档表示方法并没有太多创新，
   但由于模型变大，表示效果有了明显提升。
   自此，NLP 进入预训练时代——基于 Transformer 训练一个模型，
   在做任务时都以该模型为起点，在对应数据上进行微调训练。具有代表性的成果是 BERT 和 GPT，
   前者用了 Transformer 的编码器，后者用了 Transformer 的解码器。
   BERT 在每个文档的前面添加了一个 `[CLS］Token` 来表示整句话的语义，但与 Doc2Vec 不同的是，
   模型在推理时不需要额外训练，而是根据当前输入，通过计算自动获得表示。也就是说，同样的输入，
   相比 Doc2Vec，BERT 因为其强大的表示能力，可以通过模型计算，不额外训练就能获得不错的文本表示。GPT 在第1章有相关介绍，这里不赞述。无论是哪个预训练模型，
   底层其实都是对每个 Token 进行计算（在计算时一般会用到其他 Token 信息）。
   所以，预训练模型一般可以获得每个 Token 位置的向量表示。于是，
   文档表示依然可以使用那种最常见的方式—取平均。当然，由于模型架构变得复杂，
   取平均的方式也变得更加灵活多样，比如用自注意力作为权重加权平均。

### 进一步思考

ChatGPT 的出现其实是语言模型的突破，并没有涉及 Embedding，
但是由于模型在处理超长文本的限制（主要是资源限制和超长距离的上下文依赖问题），
Embedding 成了一个重要组件。

如前所述，Embedding 已经转变成了模型架构的副产物，架构变强 `$\rightarrow$` Token 表示变强 `$\rightarrow$` 文档表示变强：

* 第一步目前没什么问题，Token 表示通过架构充分利用了各种信息，而且可以得到不同层级的抽象。
* 第二步有点单薄，要么是 `[CLS]Token`，要么变着法子取平均。这些方法在句子上可能问题不大，
  句子一般比较短，但在段落、篇章，甚至更长的文本上就不一定了。

还是以人类阅读进行类比（很多模型都是从人类获得启发，比如 CNN，self-attention 等）。
我们在看一句话时，会重点关注其中一些关键词，整体语义可能通过这些关键词就能表达一二。
我们在看一段话时，可能依然重点关注的是关键词、包含关键词的关键句等。但是，当我们看一篇文章时，
其中的关键词和关键可能就不那么突出了，我们可能会更加关注这篇文章整体在表达什么，
描述这样的表达可能并不会用到文本中的词或句。也就是说，我们人类处理句子和篇章的方式是不一样的。
但是现在，模型把它们当成同样的东西进行处理，而没有考虑中间量变引起的质变。通俗点说，
这是几粒沙子和沙堆的区别。我们的模型设计是否可以考虑这样的不同？

Embedding 在本质上就是一组稠密向量（不用过度关注它是怎么来的），
用来表示一段文本（可以是字、词、句、段等）。获取到这个表示后，我们就可以进一步做一些任务。
不妨思考一下，当给定任意句子并得到它的固定长度的语义表示时，可以干什么。

## 度量 Embedding 相似度

提起相似度，首先会想到的编辑距离相似度，它可以用来衡量字面量的相似度，也就是文本本身的相似度。
但如果是语义层面，一般会使用余弦(cosine)相似度，它可以评估两个向量在语义空间中的分布情况：

`$$cosine(\nu, \omega) = \frac{\nu \cdot \omega}{|\nu||\omega|}=\frac{\sum_{i=1}^{N}\nu_{i}\omega_{i}}{\sqrt{\sum_{i=1}^{N}\nu_{i}^{2}}\sqrt{\sum_{i=1}^{N}\omega_{i}^{2}}}$$`

其中：

* `$\nu$` 和 `$\omega$` 分别表示两个文本向量
* `$i$` 表示向量中第 `$i$` 个元素的值

在上一节中，我们得到了一段文本的向量表示；在这里，我们可以计算两个向量的相似度。
这意味着我们现在可以知道两段给定文本的相似度，或者说，给定一段文本，
可以从库里找到与它语义最为相似的若干段文本。这个逻辑会用在很多 NLP 应用上，
我们一般把这个过程叫做 **语义匹配**。

# 相似匹配任务

## 简单问答-以问题找问题

### QA 问题

QA 是问答的意思，Q 表示 Question，A 表示 Answer，QA 是 NLP 非常基础和常用的任务。
简单来说，就是当用户提出一个问题时，我们能从已有的问题库中找到一个最相似的问题，并把它的答案返回给用户。
这里有两个关键点：第一，事先需要有一个 QA 库；第二，当用户提问时，要能够在 QA 库中找到一个最相似的问题。

用 ChatGPT 或其他生成模型执行这类任务有点麻烦，尤其是当 QA 库非常庞大时，
以及当提供给用户的答案是固定的、不允许自由发挥时。生成方式做起来事倍功半，
而 Embedding 与生俱来地非常适合，因为这类任务地核心就是在一堆文本中找出与给定文本最相似地文本。
简单总结一下，QA 问题其实就是相似度计算问题。

### 数据

* Kaggle 提供地 Qoura 数据集 [All Kaggle Questions On Quora Dataset](https://www.kaggle.com/datasets/umairnasir14/all-kaggle-questions-on-qoura-dataset)
* 数据集包括：1166 行、4 列
* 字段：
    - Questions：问题
    - Followers：关注人数
    - Answered：是否被回答
    - Link：对应地链接地址。这里把最后一列的链接地址当作答案来构造 QA 数据对

```python
import pandas as pd

# dataset
df = pd.read_csv("dataset/kaggle related questions on Qoura - Questions.csv")
print(df.shape)
print(df.head())
```

```
(1166, 4)
                                           Questions  Followers  Answered  \
0  How do I start participating in Kaggle competi...       1200         1   
1                                    Is Kaggle dead?        181         1   
2       How should a beginner get started on Kaggle?        388         1   
3              What are some alternatives to Kaggle?        201         1   
4  What Kaggle competitions should a beginner sta...        273         1   

                                                Link  
0  /How-do-I-start-participating-in-Kaggle-compet...  
1                                    /Is-Kaggle-dead  
2       /How-should-a-beginner-get-started-on-Kaggle  
3              /What-are-some-alternatives-to-Kaggle  
4  /What-Kaggle-competitions-should-a-beginner-st...  
```

## 任务基本流程

* 第一步：对每个问题计算 Embedding
    - 可以借助 OpenAI 的 Embedding 接口 
* 第二步：存储 Embedding，同时存储每个问题对应的答案
* 第三步：从存储的地方检索最相似的问题

```python

```