---
title: LSTM 生成文本
author: wangzf
date: '2022-07-15'
slug: dl-gen-lstm-textgen
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

- [LSTM 生成文本](#lstm-生成文本)
  - [生成式循环网络简史](#生成式循环网络简史)
  - [如何生成序列数据](#如何生成序列数据)
- [采样策略的重要性](#采样策略的重要性)
- [实现字符级的 LSTM 文本生成](#实现字符级的-lstm-文本生成)
- [总结](#总结)
</p></details><p></p>

# LSTM 生成文本

本节将会探讨如何将循环神经网络用于生成序列数据。将以文本生成为例, 但同样的
技术也可以推广到任何类型的序列数据, 可以将其应用于音符序列来生成新音乐, 
也可以应用于笔画数据的时间序列, 以此类推

序列数据生成绝不仅限于艺术内容生成。它已经成功应用与语音合成和聊天机器人的对话生成。
Google 于 2016 年发布的 Smart Reply (智能回复)功能, 能够对电子邮件或短信自动生成
一组快速回复, 采用的也是相似的技术

## 生成式循环网络简史

截至 2014 年年底, 还没什么人见过 LSTM 这一缩写, 即使在机器学习领域也不常见。
用循环网络生成序列数据的成功应用在 2016 年才开始出现在主流领域。但是, 这些技术都
有着相当长的历史, 最早的是 1997 年开发的 LSTM 算法。这一算法早期用于逐字符的生成文本

- HOCHREITER S,SCHMIDHUBER J.Long short-term memery [J]. Neural Computation, 1997, 9(8):1735-1780.
   - [LSTM paper](https://www.bioinf.jku.at/publications/older/2604.pdf) 
   - [LSTM slide](http://people.idsia.ch/~juergen/lstm2003tutorial.pdf) 


## 如何生成序列数据

- 如何生成序列数据
   - 用深度学习生成序列数据的通用方法, 就是使用前面的标记作为输入, 
     训练一个网络(通常是循环神经网络或卷积神经网络)来预测序列中接下来的一个或多个标记。
   - 例如, 给定输入 `the cat is on the ma`, 训练网络来预测目标 `t`, 即下一个字符。
- 语言模型 
   - **标记(token)** 通常是单词或字符, 给定前面的标记, 
     能够对下一个标记的概率进行建模的任何网络都叫做 **语言模型(language model)**
   - 语言模型能够捕捉到语言的 **潜在空间(laten space)**, 即语言的统计结构
   - 一旦训练好了这样一个语言模型, 就可以从中采样(sample, 即生成新序列)。
     向模型中输入一个初始文本字符串(即条件数据(conditioning data)), 
     要求模型生成下一个字符或下一个单词(甚至可以同时生成多个标记), 
     然后将生成的输出添加到输入数据中, 并多次重复这一过程。
     这个循环可以生成任意长度的序列, 这些序列反映了模型训练数据的结构, 
     它们与人类书写的句子几乎相同
- LSTM 字符级神经语言模型   
   - 将会用到一个 LSTM 层, 向其输入从文本语料中提取的 N 个字符组成的字符串, 
     然后训练模型来生成第 N + 1个字符。模型的输出是对所有可能的字符做 softmax, 
     得到下一个字符的的概率分布。
     这个 LSTM 叫作字符级的神经语言模型(character-level neural language model)

# 采样策略的重要性

生成文本时, 如何选择下一个字符至关重要:

- 贪婪采样(greedy sampling)
    - 始终选择可能性最大的下一个字符, 这种方法会得到重复的、可预测的字符串, 看起来不像是连贯的语言
- 随机采样(stochastic sampling)
    - 在采样过程中引入随机性, 即从下一个字符的概率分布中进行采样
    - 从模型的 softmax 输出中进行概率采样是一种很巧妙的方法, 它甚至可以在某些时候采样到不常见的字符, 
      从而生成看起来更加有趣的句子, 而且有时会得到训练数据中没有的、听起来像是真实存在的新单词, 
      从而表现出创造性。但这种方法有一个问题, 就是它在采样的过程中无法控制随机性的大小

***
**Note:**

* - 为什么需要有一定的随机性？
        - 极端例子
            - 考虑一个极端的例子, 纯随机采样, 即从均匀概率分布中抽取下一个字符, 其中每个字符的概率相同。
               这种方案具有最大的随机性, 换句话说, 这种概率分布具有最大的熵。当然, 它不会生成任何有趣的内容
            - 再看一个极端的例子, 贪婪采样, 贪婪采样也不会生成任何有趣的内容, 它没有任何随机性, 
               即相应的概率分布具有最小的熵
        - 从真实概率分布(即模型 softmax 函数输出的分布)中进行采样, 是这两个极端之间的一个中间点。
          但是, 还有许多其他中间点具有更大或更小的熵
            - 更小的熵可以让生成的序列具有更加可预测的结构(因此可能看起来更真实)
            - 更大的熵会得到更加出人意料且更有创造性的序列
        - 从生成式模型中进行采样时, 在生成过程中探索不同的随机性大小总是好的做法。我们人类是生成数据是否有趣的最终判断者, 
          所以有趣是非常主观的, 无法提前知道最佳熵的位置
***

为了在采样的过程中控制随机性的大小, 可以引入一个叫做 **softmax 温度(softmax temperature)** 的参数, 
用于表示采样概率分布的熵, 即表示所选择的下一个字符会有多么出人意料或多么可预测。给定一个 `temperature` 值, 
按照下列方法对原始概率分布(即模型的 softmax 输出)进行加权, 计算得到一个新的概率值

```python
import numpy as np

def reweight_distribution(original_distribution, temperature = 0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)
```

# 实现字符级的 LSTM 文本生成

任务:训练一个语言模型, 这个模型是针对尼采的一些已被翻译为英文的作品的写作风格和主题的模型

# 总结

- 可以生成离散的序列数据, 其方法是:给定前面的标记, 训练一个模型来预测接下来的一个或多个标记
- 对于文本来说, 这种模型叫做语言模型。它可以是单词级的, 也可以是字符级的
- 对于一个标记进行采样, 需要在坚持模型的判断与引入随机性之间寻找平衡
- 处理这个问题的一种方法是使用 softmax 温度。一定要尝试多种不同的温度, 以找到合适的那一个

