---
title: Text
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-type-text
categories:
  - feature engine
tags:
  - ml
---


# 文本数据特征提取及处理

- 特征提取
    - 词袋模型(bag-of-words)
        - 单词数量的统计列表
- 特征缩放
    - tf-idf(term frequency-inverse document frequency)
- 主题模型(topic model)
- 词嵌入模型(word embeding)
    - Word2Vec
        - CBOW(Continues Bags of Words)
        - Skip-gram

对于文本数据, 可以从一个单词数量的统计列表开始, 称为词袋(bag-of-words).
对于像文本分类这样的简单任务来说, 单词数量统计通常就够用了. 
这种技术还可以用于信息提取, 它的目标是提取出一组与查询文本相关的文档. 
这两种任务都可以凭借单词级别的特征圆满完成, 
因为特定词是否存在于文档中这个指标可以很好的标识文档的主题内容. 

## 词袋

1. 在词袋特征化中, 一篇文本文档被转换为一个计数向量, 这个计数向量包含 `词汇表` 中所有可能出现的单词
2. 词袋将一个文本文档转换为一个扁平向量之所以说这个向量是"扁平的”, 是因为它不包含原始文本中的任何结构, 
   原始文本是一个单词序列, 但词袋中没有任何序列, 它只记录每个单词在文本中出现的次数. 
   因此向量中单词的顺序根本不重要, 只要它在数据集的所有文档之间保持一致即可
3. 词袋也不表示任何单词层次, 在词袋中, 每个单词都是平等的元素
4. 在词袋表示中, 重要的是特征空间中的数据分布.在词袋向量中, 
   每个单词都是向量的一个维度. 如果词汇表中有 n 个单词, 
   那么一篇文档就是 n 维空间中的一个点
5. 词袋的缺点是, 将句子分解为单词会破坏语义


## n-gram

1. n 元词袋(bag-of-n-grams)是词袋的一种自然扩展, n-gram(n元词)是由n个标记(token)组成的序列. 1-gram
   就是一个单词(word), 又称为一元词(unigram). 经过分词(tokenization)之后, 计数机制会将单独标记转换为单词计数, 或将有重叠的序列作为
   n-gram 进行计数. 
2. n-gram 能够更多地保留文本的初始序列结构, 因此 n
   元词袋表示法可以表达更丰富的信息
3. 然而, 并不是没有代价, 理论上, 有k个不同的单词, 就会有 :math:`k^{2}`
   个不同的
   2-gram(二元词), 实际上, 没有这么多, 因为并不是每个单词都可以跟在另一个单词后面
4. n-gram(n > 1)一般来说也会比单词多得多, 这意味着 n
   元词袋是一个更大也更稀疏的特征空间, 也意味着 n
   元词袋需要更强的计算、存储、建模能力. n
   越大, 能表示的信息越丰富, 相应的成本也会越高

```python
import pandas
import json
from sklearn.feature_extraction.text import CountVectorizer

js = []
with open("yelp_academic_dataset_review.json") as f:
    for i in range(10000):
        js.append(json.loads(f.readline()))

review_df = pd.DataFrame(js)
```

## 主题模型

- 主题模型用于从文本库中发现有代表性的主题(得到每个主题上面词的分布特性), 并且能够计算出每篇文章的主题分布

## 词嵌入模型



# 特征过滤

## 使用过滤获取清洁特征

### 停用词

- 停用词
    - 中文停用词
        - 网上找
    - 英文停用词
        - NLTK
- 分词
    - 中文分词
    - 英文分词
        - 不能省略撇号
        - 单词为小写

### 基于频率的过滤

