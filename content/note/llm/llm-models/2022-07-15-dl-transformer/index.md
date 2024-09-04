---
title: Transformer
subtitle: Attention Is All You Need
author: 王哲峰
date: '2022-04-05'
slug: dl-transformer
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

- [Transformer 架构](#transformer-架构)
    - [Transformer 模型架构](#transformer-模型架构)
    - [Transformer 数学表示](#transformer-数学表示)
        - [Inputs](#inputs)
        - [Embedding](#embedding)
        - [Attention](#attention)
        - [Multi-head Attention](#multi-head-attention)
        - [Feed forward](#feed-forward)
        - [Linear](#linear)
        - [Softmax](#softmax)
    - [Transformer 架构解析](#transformer-架构解析)
        - [Encoder-Decoder](#encoder-decoder)
        - [Encoder](#encoder)
        - [Decoder](#decoder)
- [输入模块](#输入模块)
- [编码模块](#编码模块)
    - [从高处看 Self-Attention](#从高处看-self-attention)
    - [从细节看 Self-Attention](#从细节看-self-attention)
        - [Self-Attention 的向量计算](#self-attention-的向量计算)
        - [Self-Attention 的矩阵计算](#self-attention-的矩阵计算)
    - [Multi-head 机制](#multi-head-机制)
    - [Positional Encoding](#positional-encoding)
    - [残差项](#残差项)
- [解码模块](#解码模块)
    - [Encoder-Deocder Attention 层](#encoder-deocder-attention-层)
    - [Linear 和 Softmax 层](#linear-和-softmax-层)
- [训练概要](#训练概要)
- [损失函数](#损失函数)
- [Transformer 要点问题](#transformer-要点问题)
- [Transformer 架构原理通俗解释](#transformer-架构原理通俗解释)
    - [Tokenization](#tokenization)
    - [Modeling tokens](#modeling-tokens)
    - [Attention](#attention-1)
    - [Query-Key-Value vector](#query-key-value-vector)
    - [Context aware Embedding](#context-aware-embedding)
    - [pytorch code](#pytorch-code)
- [参考](#参考)
- [TODO](#todo)
</p></details><p></p>

# Transformer 架构

## Transformer 模型架构

Transformer 是 Google 在 2017 年提出的一个 **NLP 模型**，适用于**机器翻译**任务。
它不依赖于使用 CNN 和 RNN，而是基于 **注意力机制(Attention Mechanism)** 构建网络结构。

![img](images/transformer.png)

## Transformer 数学表示

### Inputs

`$$\operatorname{Inputs} = (x_{1}, x_{2}, \cdots, x_{n})$$`

### Embedding

`$$X = \begin{bmatrix}
x_{1, embedding} = [\cdots] \\
x_{2, embedding} = [\cdots] \\
\cdots \\
x_{n, embedding} = [\cdots] \\
\end{bmatrix}$$`

### Attention

`$$Q = X \cdot W^{Q}$$`
`$$K = X \cdot W^{K}$$`
`$$V = X \cdot W^{V}$$`
`$$\operatorname{Attention}(Q, K, V)=\operatorname{Softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) \cdot V$$`

### Multi-head Attention

`$$\operatorname{Multi-head}(Q, K, V) =\operatorname{Concat}\left(\operatorname{head}_{1}, \ldots, \text { head }_{\mathrm{h}}\right) \cdot W^{O}$$`

其中：

`$$head_{i} =\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)$$`

### Feed forward


### Linear


### Softmax


## Transformer 架构解析

如果把 Transformer 模型当作黑盒，在机器翻译任务里，模型接受一种语言的句子，输出另一种语言的翻译句子：

![img](images/tf1.png)

### Encoder-Decoder

打开中间这个 Transformer 结构，能看到一个 **编码模块** 和一个 **解码模块**，且它们互相连接：

![img](images/ed.png)

**编码模块** 有多个 **编码器** 堆叠而成 (论文堆了 6 个，数字 6 没有特别规定，可以自行修改进行实验)，
**解码模块**也堆叠了 6 个 **解码器**。Transformer 模型结构中左侧和右侧的 `$N_x$` 就是指编码器或解码器的个数，
这里就是 `$N_6$`：

![img](images/ed2.png)

### Encoder

**编码器**之间都是相同结构，但不共享权重。每个编码器有两层：

![img](images/ed3.png)

编码器的输入首先流入一个 **自注意力层(Self-Attention layer)**，当编码器对某个特定单词进行编码时，
该层会帮助编码器关注输入句子中的其它单词。后续会详细讲这块。

自注意力层(Self-Attention layer)的输出会被喂入 **前馈神经网络 (feed-forward neural network)**`，
每个输入位置上都有一个前馈神经网络，它们之间是相互独立的 (补充: 论文说前馈神经网络是 point-wise)。

### Decoder

**解码器** 也有编码器的这两层，但是在它们中间还有个 **注意力层**，
帮助解码器关注输入句子的相关部分（跟 seq2seq 模型里的关注机制类似）。

![img](images/ed4.png)

# 输入模块

> Embedding

上面已经看过模型的主要模块，现在来看下 **向量/张量** 是如何在这些模块中从 **输入** 到 **输出**。

以 NLP 常见应用的为例，用一个 Embedding 算法将每个输入词转化为一个向量：

![img](images/vec1.png)

Word Embedding 只发生在编码器的最底端，对于所有编码器来说，它们会接收一个 list，
list 内含有多个长度为 512 的词向量。但对其他编码器来说，它们的输入会是上一层编码器的输入。
list 的长度是可调整的超参，一般为训练集中最长句子的长度。

对输入序列进行词向量化后，每个词向量会依次流入下面编码器中的两个子层

![img](images/vec2.png)

现在，来看下 Transformer 的一个重要特性：句子中的每个对应位置上的词是按照它自有路径流入编码器的，
在 Self-Attention 层这些路径是相互依赖的。但 Feed-forward 层不存在依赖性，
因此，当 Self-Attention 层的输入流入 feed-forward 层，这些独立的路径可以并行。

接下来，将目光转向一个更短的句子，看下在编码器每个子层里发生了什么？

# 编码模块

正如之前提到的，编码器接收一个 list 的词向量(Word Vector)作为输入。
它将 list 里的向量传入 Self-Attention 层，然后喂入 feed-forward 层，
最后输出给下个编码器：

![img](images/encoding1.png)

注意：对于每个位置的词向量来说，Feed-forward 层都是相互独立，
Transformer 的作者也因此称之为 position-wise fully connected feed-forward network。
而 Self-Attention 层则是多个词向量共用的。

## 从高处看 Self-Attention

不要被 “Self-Attention” 这个词迷惑了，它不是我们熟悉的含义。直到阅读了《Attention is all you need》原文，
才弄明白了此概念。下面来看下它具体怎么工作的。

如果说，想翻译以下句子：

```
The animal didn't cross the street because it was too tired.
```

句子中的 `"it"` 是指代什么？指代 `"street"` 还是 `"animal"`？对人来说很简单，但对算法来说不简单。
当模型处理 `"it"` 时，Self-Attention 允许将 `"it"` 和 `"animal"` 关联起来。

**当模型处理每个词时 (即在输入序列的每个位置上)，Self-Attention 允许关注输入序列其它位置作为辅助信息，
帮助对当前词更好地编码。如果熟悉 RNN，想想隐藏层是如何使得 RNN 利用历史词/向量的表征去获取当前词的表征向量。
Transformer 的 Self-Attention 是将对相关词的理解融入到当前词的处理当中。**

比如：当在第 5 个编码器 (即堆叠在上面的编码器) 编码 `"it"` 时，部分关注会集中在 `"The animal"` 上，
然后将它的表征融合进 `"it"` 的编码中：

![img](images/self-attention1.png)

## 从细节看 Self-Attention

首先看下如何使用 **向量** 计算 Self-Attention，然后进一步看如何使用 **矩阵** 实施计算。

### Self-Attention 的向量计算

1. **第一步，在计算 Self-Attention 中，是对编码器的每个输入向量 (即每个词的向量) 创建三个向量。**
   即：
   `$$q = x \cdot W^{q}$$`
   `$$k = x \cdot W^{k}$$`
   `$$v = x \cdot W^{v}$$`
   对于每个词 **Embedding 向量**，会创建 **Query 向量**、**Key 向量** 和 **Value 向量**，
   将三个权重矩阵乘以词向量便可得到这三个向量，这三个权重矩阵会在训练过程中不断学习。

    ![img](images/self-attention2.png)

> * 那什么是 Query 向量、Key 向量和 Value 向量呢？对于计算和思考 Attention 来说，提取它们是有益的。
> 一旦你往下了解 Attention 是如何计算，你就会了解到每个向量是扮演着什么角色了。
> 
> * 注意：相比于词向量 (512维)，这三个向量（Query、Key、Value）的维度是偏小的 (64维)。
> 三向量不是一定要小，这是为了让 Multi-head Attention 的计算是连续的 (补充: 这里可能比较模糊，
> 其实是这样的，Self-Attention 其实是个 Multi-head 的形式，即多个 Attention 模块，
> 原论文是说有 8 个并行的 Attention 层，这样若总维度为 512，那会变为 512/8=64 维，
> 相当于全 512 维度输入 Single-head Attention 变为了 64 维输入 Multi-head Attention，如下图所示)。
> 
> ![img](images/self-attention3.png)

2. **计算 Self-Attention 的第二步是计算一个分数。** 
   即：
   `$$q \cdot k$$`
   举例，在对 `"Thinking"` 这第一个词计算 Self-Attention，
   当前词(`"Thinking"`)要对输入句子中其它词(`"Machines"`)对进行打分，
   这个分数决定了：当对某给定位置上的词进行编码时，应该给输入句子中其它位置上的词多少关注。
   Query 向量和 Key 向量点乘便可得到分数，所以如果对位置 1 的词计算 Self-Attention，
   `$q_{1}$` 点乘 `$k_{1}$` 便可得到第一个分数，第二个分数则是 `$q_{1}$` 和 `$k_{2}$` 的点乘。

    ![img](images/self-attention4.png)

3. **第三步是将分数除 `$\sqrt{d_{k}}$`。**
   即：
   `$$q \cdot k / \sqrt{d_{k}}$$`
   论文提到这里是对 Key 向量维度开方(`$\sqrt{d_{k}}$`)，即 `$\sqrt{64}=8$`，
   这能帮助拥有更稳定的梯度，也可以是其它可能值，
   但这个是默认的(补充: 作者是担心对于大的 Key 向量维度会导致点乘结果变得很大，
   将 Softmax 函数推向得到极小梯度的方向，因此才将分数除以 Key 向量维度开方值)。
4. **第四步是基于上面的分数求 Softmax 分数。**
   即：
   `$$s = Softmax(q \cdot k / \sqrt{d_{k}})$$`
   这个 Softmax 分数决定对当前位置上的词，句子上的各词该表达多少程度。
   明显在当前位置上的词获取最高的 Softmax 分数，但有时，
   与当前词有关的其它词如果能参与进来也是有帮助的。

    ![img](images/self-attention5.png)

5. **第五步是 Value 向量与 Softmax 分数相乘(以便相加)。**
   即：
   `$$s \cdot v = Softmax(q \cdot k / \sqrt{d_{k}}) \cdot v$$`
   这是为了保留想关注的词，掩盖掉不相干的词，例如：给他们乘上极小值 0.001。
6. **第六步是加总这些加权的 Value 向量。**
   即：
   `$$z = \sum s \cdot v = \sum Softmax(q \cdot k / \sqrt{d_{k}}) \cdot v$$`
   对于第一个词，这便生成了 Self-Attention 层在此位置上的输出。

    ![img](images/self-attention6.png)

上面就是 Self-Attention 层的计算过程，结果向量可以输入给 feed-forword 神经网络。
在真实应用中，是使用矩阵计算加快处理速度，所以看下单词级别的矩阵计算

### Self-Attention 的矩阵计算

1. 第一步是计算 Query、Key 和 Value 矩阵，我们是通过打包词向量 `$x$` 成矩阵 `$X$`，
   然后分别乘上三个可学习的权重矩阵 `$(W^{Q}, W^{K}, W^{V})$`。
   `$$Q = X \cdot W^{Q}$$`
   `$$K = X \cdot W^{K}$$`
   `$$V = X \cdot W^{V}$$` 
   在矩阵 `$X$` 中，每一行对应输入句子中每一个单词，
   再次看到词向量长度(512，图中的 4 个 box)和 Q/K/V 向量长度(64, 图中的 3 个 box)是不一样的。

    ![img](images/self-attention7.png)

2. 最后，因为要处理这些矩阵，我们能压缩第二步到第六步到一个方程式，
   从而计算出 Self-Attention 层的输出结果：
   - 第二步：
   `$$Q \cdot K^{T}$$`
   - 第三步：
   `$$Q \cdot K^{T} / \sqrt{d_{K}}$$`
   - 第四步：
   `$$S = Softmax(Q \cdot K^{T} / \sqrt{d_{k}})$$`
   - 第五步：
   `$$S \cdot V = Softmax(Q \cdot K^{T} / \sqrt{d_{K}}) \cdot V$$`
   - 第六步：
   `$$Z = \sum S \cdot V = \sum Softmax(Q \cdot K^{T} / \sqrt{d_{K}}) \cdot V$$`

    ![img](images/self-attention8.png)

## Multi-head 机制

论文在 Self-Attention 层前还加入了 “Multi-head” 的 Attention 机制，它从两方面提升了 Attention 层的表现：

1. Multi-head Attention 增强了模型关注不同位置的能力
   - 在以上例子中，`$z_{1}$` 没怎么包括其他词编码的信息 (例如，过于关注 `"it"` 并不能带来很多信息)，
     但实际上可能是由真实值所控制的 (例如，关注 `"it"` 指代的 `"The animal"` 会对模型更好)，
     所以如果翻译 `"The animal didn’t cross the street because it was too tired."`，
     Multi-head Attention 能帮助我们知道 `"it"` 指代的是哪个词，从而提升模型表现。
2. Multi-head 机制给 Attention 层带来多个 “表征子空间”
    - 晚点会看到，Multi-head Attention 不只是 1 个，
      而是多个 Q/K/V 矩阵 (Transformer 用了 8 个关注头，所以对于每个编码器-解码器，有 8 组)，
      每组都是随机初始化。然后，训练之后，每组会将输入向量 (或者是来自更低部编码器-解码器的向量) 映射到不同的表征空间

    ![img](images/m-attention1.png)

    - 在 Multi-head Attention 下，对每个头都有独立的 Q/K/V 权重矩阵，因此每个头会生成不同的 Q/K/V 矩阵。
      正如之前所做，将 `$X$` 和 `$W^{Q}$`/ `$W^{K}$` / `$W^{V}$` 矩阵相乘便可得到 Q/K/V 矩阵。
      如果按上面方式去计算，8 次与不同权重矩阵相乘会得到 8 个不同的 Z 矩阵

    ![img](images/m-attention2.png)

    - 这有点困难了，因为 Feed-Forward 层不期望获取 8 个矩阵，它希望得到 1 个矩阵 (即一个词 1 个向量)，
      所以需要使用一个方法将这 8 个矩阵变为 1 个矩阵。怎么做呢？直接合并矩阵，
      然后乘上一个额外的权重矩阵 `$W^{O}$` 就好了

    ![img](images/m-attention3.png)

这差不多就是 Multi-head Self-Attention 层的全部了，来把这些矩阵放在一块看看：

![img](images/m-attention4.png)

现在，已经接触了 Attention 的 head 了，重温下之前的例子，去看下在编码 `"it"` 时，
不同 Attention head 是怎么 Attenion：

![img](images/m-attention5.png)

当我们编码词 `"it"` 时，一个 Attention head 更加关注 `"The animal"`，
另一个更关注 `"tired"`。在某种意义上，对于 `"it"` 的表征加入了 `"animal"` 和 `"tired"` 的考虑。
如果我们加入所有关注头，就可能会较难解释：

![img](images/m-attention6.png)

## Positional Encoding 

> 使用位置编码表征序列顺序

说了那么久，有件事没讲，怎么去将输入序列的词顺序考虑进模型呢？
(补充：需要考虑词顺序是因为 Transformer 没有循环网络和卷积网络，
因此需要告诉模型词的相对/绝对位置)。

为了解决这个问题，Transformer 给每个输入 Embedding 加上一个向量，该向量服从模型学习的特定模式，
这决定了词位置，或者序列中词之间的距离。当 Embedding 向量被映射到 Q/K/V 向量和点乘 Attention 时，
对 Embedding 向量加上位置向量有利于提供有意义的距离信息。

![img](images/position_encoding.png)

假设 Embedding 向量维度是 4，那么真实位置编码会如下所示：

![img](images/position_encoding2.png)

> 这个指定模式是怎样的？如下图所示，每行对应的是一个词向量的位置编码，
> 所以第一行是要加到输入序列中第一个词 Embedding 的向量。
> 每行包含 512 个值，每个值范围在 -1 到 1 之间。对值进行着色可视化：
> 
> ![img](images/position_encoding3.png)
> 
> 位置编码的公式在[《Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)》](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) 有提及，你也能看在 [`get_timing_signal_1d()`](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py) 里位置编码的代码。
> 这不是位置编码的唯一方法，但它能处理不可见长度的序列 (例如训练好的模型被要求去翻译一个超过训练集句子长度的句子)。
> 
> 以上展示的位置编码在 Tensor2Tensor (论文的开源代码) 实现里是合并 sine 和 cosine，
> 但是论文不在论文展示的又不一样，论文是交叉使用两种 signals (即偶数位置使用 sine，奇数位置使用 cosine)，
> 下图便是论文生成方式得到的：
> 
> ![img](images/position_encoding4.png)

这里作者提到了两种方法：

* 方法 1：用不同频率的 sine 和 cosine 函数直接计算
* 方法 2：学习出一份 Positional Embedding（参考文献）

经过实验发现两者的结果一样，所以最后选择了第一种方法，位置编码的公式如下：

`$$PE_{(pos, 2i)} = sin\Bigg(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\Bigg)$$`
`$$PE_{(pos, 2i + 1)} = cos\Bigg(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\Bigg)$$`

其中：

* `$pos$` 是位置
* `$i$` 是维度
* `$2i$` 代表偶数维度
* `$2i+1$` 代表奇数维度
* `$d_{model}$` 为 512

作者提到，方法 1 的好处有两点：

1. 任意位置的 `$PE_{pos+k}$` 都可以被 `$PE_{pos}$` 的线性函数表示，三角函数特性复习下：

`$$cos(\alpha + \beta) = cos(\alpha)cos(\beta) - sin(\alpha)sin(\beta)$$`
`$$sin(\alpha + \beta) = sin(\alpha)cos(\beta) + cos(\alpha)sin(\beta)$$`

2. 如果是学习到的 Positional Embedding，(个人认为，没看论文)会像词向量一样受限于词典大小。
   也就是只能学习到 “位置 2 对应的向量是 (1,1,1,2)” 这样的表示。所以用三角公式明显不受序列长度的限制，
   也就是可以对比所遇到序列的更长的序列进行表示。

## 残差项

在继续深入下去之前，编码器结构有个细节需要注意：
编码器每个子层 (Self-Attention, Feed Forward) 都有一个残差连接项，
并跟着 Normalization 层：

![img](images/res1.png)

如果将向量和 Self-Attention 层的 Layer-Norm 操作可视化，会如下所示：

![img](images/res2.png)

在解码器中也一样，如果 Transformer 堆叠了两层的编码器和解码器，它会如下所示：

![img](images/res3.png)

# 解码模块

现在，讲了编码器大部分的内容，也基本了解解码器的成分，但看下它们是怎样一起工作的。

## Encoder-Deocder Attention 层

编码器先处理输入序列，最上层编码器的输出被转换成一组关注向量 `$K$` 和 `$V$`，
它们被输入进 Encoder-Decoder Attention 层内的每个解码器里，帮助解码器关注输入序列中合适的部分：

![img](images/decoder.gif)

接下来不断重复解码过程，直到生成出特殊符号，标志 Transformer 解码器完成了输出。
每步的输出被喂入下一时间轮数下的底层解码器，然后不断向上运输解码结果。与对待编码器输入一样，
向量化和加位置编码给解码器的输入，以便解码器了解到每个词的位置信息

![img](images/decoder.png)

解码器中的 Self-Attention 层和编码器的有点不一样：在解码器中，Self-Attention 层只允许关注输出序列的历史位置，
这通过 Self-Attention 的 softmax 前，遮掩未来位置实现 (即设置它们为 -inf)。

Encoder-Decoder Attention 层工作就像 Multi-head Self-Attention，除了它是从前一层创建出 Query 矩阵，
然后接收编码器输出的 Key 和 Value 矩阵。

## Linear 和 Softmax 层

解码器们输出浮点向量，怎么将其转换为一个词呢？这是最后的 Linear 和 Softmax 层所做的工作。

* 线性层就是简单的全连接神经网络，将解码器的输出向量映射到一个很大很大的向量叫 logits 向量
    - 假设模型透过学习训练集，知道 1 万个独特的英文单词 (即模型的输出词汇)。
      这就需要 logits 向量有 1 万个格子，每个格子对应着一个独特词的分数
* Softmax 层将这些分数转化为概率 (0-1 的正数)，选择最高概率的格子，然后对应的单词作为当前时间步数下的输出

![img](images/linear_softmax.png)

# 训练概要

已经讲了 Transformer 的前向过程，现在来看下训练的内容。训练时，
未训练的模型会通过一样的前向过程，但由于是在有标签的训练集上训练，可以对比输出和真实值。

为了可视化，假设输出单词只包含 6 个词：

> `a`, `am`, `i`, `thanks`, `student`, `"<eos>"`

![img](images/word.png)

一旦定义了输出词表 (output vocabulary)，能用相同长度的向量去构建词表里的每个词，
这边是独热编码，举个例子，可以编码 `am` 成如下向量：

![img](images/word2.png)

# 损失函数

接下来，讨论下模型的损失函数，在训练阶段用于优化的度量指标，使得模型训练得更加准确。

假设我们在训练模型，训练模型实现将 `"merci"` 翻译成 `"thanks"`。
这意味着，想输出的概率分布指向词 `"thanks"`。但因为模型还未训练，它还不可以发生：

![img](images/loss1.png)

怎么对比两个概率分布？简单相减就好(对于细节，可以看交叉熵和 KL 散度相关资料)。
注意这只是个最简单的例子，更真实情况下，我们会用一个句子，例如 `je suis étudiant`，
期待输出 `i am a student`。在这里是希望模型能够成功输出概率分布：

* 每个概率分布被表征成一个长度为词表尺寸(`vocab_size`)的向量 (简单例子为 6，但现实是 3 万或 5 万)
* 第一个概率分布有最高的概率格子是在词 `"i"` 上
* 第二个概率分布有最高的概率格子是在词 `"am"` 上
* ...
* 直到第五个概率分布指向 `<end of sentence>` 符号，这里也有格子关联到它

![img](images/loss2.png)

在足够大的数据集上训练模型足够多的时间，期望生成的概率分布会如下所示：

![img](images/loss3.png)

现在，因为模型每步生成一个输出，可以假设模型选择最高概率的词，而丢弃剩余的词，
这种方式称为贪婪解码。另一种方式是便是 beam search 了，比方说第一步预测时，'I' 和 'a' 是两个 Top 概率词，
然后，如果以 'I' 和 'a' 分别作为第一个预测值，去进行下一步预测，
如果 'I' 作为第一词预测第二个词下的误差比 'a' 作为第一词预测第二个词下的误差小，
那么便保留 'I' 作为第一词，不断重复这个过程。在例子中，
beam_size 是 2(意味着任何时候，两个词 (未完成的翻译) 的假设都被保留在 memory 中)，
然后 top_beams 也是 2 个(意味着会返回 2 个翻译)，这些都是超参可以调整的

# Transformer 要点问题

1. Transformer 是如何解决长距离依赖的问题的？
    - Transformer 是通过引入 Scale-Dot-Product 注意力机制来融合序列上不同位置的信息，从而解决长距离依赖问题。
      以文本数据为例，在循环神经网络 LSTM 结构中，输入序列上相距很远的两个单词无法直接发生交互，
      只能通过隐藏层输出或者细胞状态按照时间步骤一个一个向后进行传递。
      对于两个在序列上相距非常远的单词，中间经过的其它单词让隐藏层输出和细胞状态混入了太多的信息，
      很难有效地捕捉这种长距离依赖特征。但是在 Scale-Dot-Product 注意力机制中，
      序列上的每个单词都会和其它所有单词做一次点积计算注意力得分，
      这种注意力机制中单词之间的交互是强制的不受距离影响的，所以可以解决长距离依赖问题
2. Transformer 在训练和测试阶段可以在时间(序列)维度上进行并行吗？
    - 在训练阶段，Encoder 和 Decoder 在时间(序列)维度都是并行的；
      在测试阶段，Encoder 在序列维度是并行的，Decoder 是串行的
    - 首先，Encoder 部分在训练阶段和预测阶段都可以并行比较好理解，
      无论在训练还是预测阶段，它干的事情都是把已知的完整输入编码成 memory，
      在序列维度可以并行
    - 对于 Decoder 部分有些微妙。在预测阶段 Decoder 肯定是不能并行的，因为 Decoder 实际上是一个自回归，
      它前面 `$k-1$` 位置的输出会变成第 `$k$` 位的输入的。前面没有计算完，后面是拿不到输入的，肯定不可以并行。
      那么训练阶段能否并行呢？虽然训练阶段知道了全部的解码结果，但是训练阶段要和预测阶段一致啊，
      前面的解码输出不能受到后面解码结果的影响啊。
      但 Transformer 通过在 Decoder 中巧妙地引入 Mask 技巧，使得在用 Attention 机制做序列特征融合的时候，
      每个单词对位于它之后的单词的注意力得分都为 0，这样就保证了前面的解码输出不会受到后面解码结果的影响，
      因此 Decoder 在训练阶段可以在序列维度做并行
3. Scaled-Dot Product Attention 为什么要除以 `$\sqrt{d_k}$`?
    - 为了避免 `$d_k$` 变得很大时 softmax 函数的梯度趋于 0。
      假设 Q 和 K 中的取出的两个向量 `$q$` 和 `$k$` 的每个元素值都是正态随机分布，
      数学上可以证明两个独立的正态随机变量的积依然是一个正态随机变量，
      那么两个向量做点积，会得到 `$d_k$` 个正态随机变量的和，
      数学上 `$d_k$` 个正态随机变量的和依然是一个正态随机变量，
      其方差是原来的 `$d_k$` 倍，标准差是原来的 `$\sqrt{d_k}$` 倍。
      如果不做 scale, 当 `$d_k$` 很大时，求得的 `$QK^T$` 元素的绝对值容易很大，
      导致落在 softmax 的极端区域(趋于 0 或者 1)，极端区域 softmax 函数的梯度值趋于 0，
      不利于模型学习。除以 `$\sqrt{d_k}$`，恰好做了归一，不受 `$d_k$` 变化影响
4. MultiHeadAttention 的参数数量和 head 数量有何关系?
    - MultiHeadAttention 的参数数量和 head 数量无关。
      多头注意力的参数来自对 QKV 的三个变换矩阵以及多头结果 concat 后的输出变换矩阵。
      假设嵌入向量的长度是 d_model, 一共有 h 个 head. 对每个 head，
      `$W_{i}^{Q},W_{i}^{K},W_{i}^{V}$` 这三个变换矩阵的尺寸都是 `$d_model \times (d_model/h)$`，
      所以 h 个 head 总的参数数量就是 `$3 \times d_model \times (d_model/h) \times h = 3 \times d_model \times d_model$`。
      它们的输出向量长度都变成 `$d_model/h$`，经过 Attention 作用后向量长度保持，
      h 个 head 的输出拼接到一起后向量长度还是 d_model，
      所以最后输出变换矩阵的尺寸是 `$d_model×d_model$`。
      因此，MultiHeadAttention 的参数数量为 `$4 \times d_model \times d_model$`，和 head 数量无关
5. Transformer 有什么缺点？
    - Transformer 主要的缺点有两个，一个是注意力机制相对序列长度的复杂度是 `$O(n^2)$`，第二个是对位置信息的
        - 第一，Transformer 在用 Attention 机制做序列特征融合的时候，
          每两个单词之间都要计算点积获得注意力得分，这个计算复杂度和序列的长度平方成正比，
          对于一些特别长的序列，可能存在着性能瓶颈，有一些针对这个问题的改进方案如 Linformer
        - 第二个是 Transformer 通过引入注意力机制两两位置做点乘来融合序列特征，
          而不是像循环神经网络那样由先到后地处理序列中的数据，导致丢失了单词之间的位置信息关系，
          通过在输入中引入正余弦函数构造的位置编码 PositionEncoding 一定程度上补充了位置信息，
          但还是不如循环神经网络那样自然和高效

# Transformer 架构原理通俗解释

## Tokenization

在开始之前，简要介绍一下 tokenization：

```
原始文本 → 分词 → 嵌入 → 模型
```

嵌入(Embedding)是使用一堆数字对每个 token（大约一个词）进行有意义的表示。
这个嵌入是提供给语言模型作为输入的。


![img](images/tf-attention1.jpg)
<!-- <image src="images/tf-attention1.jpg" width=50% align="center"> -->

## Modeling tokens

语言建模的核心思想是理解语言中的结构和模式。
通过对句子中的单词（tokens）建模，可以捕捉文本的上下文和含义。

![img](images/tf-attention2.jpg)
<!-- <image src="images/tf-attention2.jpg" width=50%> -->

## Attention

现在自我关注是一种帮助建立这些关系的通信机制，表达为概率分数。
每个 token 都会给自己最高分，并根据它们的相关性给其他 tokens 分数。
您可以将其视为一个有向图（Directed Graph）。

![img](images/tf-attention3.jpg)
<!-- <image src="images/tf-attention3.jpg" width=50%> -->

## Query-Key-Value vector

了解这些概率/注意力分数是如何获得的，必须理解 3 个关键术语：

* 查询向量
* 关键向量
* 价值向量

这些向量是通过将输入嵌入乘以三个可训练的权重矩阵而创建的。

![img](images/tf-attention4.jpg)
<!-- <image src="images/tf-attention4.jpg" width=50%> -->

## Context aware Embedding

现在更全面地了解输入嵌入是如何与键、查询和数值结合以获得实际的注意力分数的。
获取密钥、查询和值后，将它们合并以创建一组新的上下文感知嵌入。

![img](images/tf-attention5.jpg)
<!-- <image src="images/tf-attention5.jpg" width=50%> -->

## pytorch code

使用 PyTorch 实现自注意力，再也不会更简单了！这非常直观！

![img](images/tf-attention6.jpg)
<!-- <image src="images/tf-attention6.jpg" width=50%> -->

# 参考

* [Transformer 介绍](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247486160&idx=1&sn=2dfdedb2edbca76a0c7b110ca9952e98&chksm=fa0414bbcd739dad0ccd604f6dd5ed99e8ab7f713ecafc17dd056fc91ad85968844e70bbf398&scene=178&cur_album_id=1577157748566310916#rd)
* [《The Illustrated Transformer》](http://jalammar.github.io/illustrated-transformer/)

# TODO

* [Paper: Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [《Transformer: A Novel Neural Network Architecture for Language Understanding》](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
* [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html)
* [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg)
* [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
* [《Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)》](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* [Transformer 的一家](https://mp.weixin.qq.com/s/ArzUQHQ-imSpWRPt6XG9FQ)
* [详解 Transformer ](https://zhuanlan.zhihu.com/p/48508221)
* [深度学习中的注意力机制](https://zhuanlan.zhihu.com/p/37601161)
* [Self-Attention 和 Transformer](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/#background)
* [transformer_positional_encoding_graph.ipynb](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)
* [TransformerX](https://github.com/wangzhefeng/TransformerX)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/#training-data-and-batching)
* [Transformer](https://mp.weixin.qq.com/s/biFNBi8Efs5uDYEocwmqbg)
* HuggingFace
    - [Transformer Doc](https://huggingface.co/docs/transformers/quicktour)
    - [transformers library](https://github.com/huggingface/transformers)
    - [Transformers 教程：pipeline一键预测](https://mp.weixin.qq.com/s/1dtk5gCa7C-wyVQ9vIuRYw)
