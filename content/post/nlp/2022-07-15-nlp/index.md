---
title: NLP
author: 王哲峰
date: '2022-04-05'
slug: nlp
categories:
  - nlp
tags:
  - tool
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

- [NLP 简介](#nlp-简介)
  - [NLP 介绍](#nlp-介绍)
  - [NLP 难度](#nlp-难度)
  - [NLP 结构](#nlp-结构)
  - [NLP 模型](#nlp-模型)
- [NLP 的研究任务](#nlp-的研究任务)
- [NLP 相关知识的构成](#nlp-相关知识的构成)
  - [基本术语](#基本术语)
  - [知识结构](#知识结构)
- [NLP 的三个层面](#nlp-的三个层面)
- [NLP 相关技术](#nlp-相关技术)
- [NLP 常用语料库](#nlp-常用语料库)
- [参考 \& TODO](#参考--todo)
</p></details><p></p>

# NLP 简介

## NLP 介绍

自然语言处理(Nature Language Processing, NLP)是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言。
NLP 包括多方面和步骤，基本有认知、理解、生成等部分

自然语言认知和理解是让计算机把输入的语言变成有意思的符号和关系，然后根据目的再进行处理。
所以简单来说，NLP 就是让计算机理解人类语言。为了达到这样的目的，我们需要在理论上基于
数学和统计理论建立各种自然语言，然后通过计算机来实现这些语言模型。因为人类语言的多样性和复杂性，
所以总体而言 NLP 是一门机具挑战的学科和领域

## NLP 难度

NLP 是个非常庞杂的领域了，学习和应用起来都颇有难度。难度体现在语言场景、学习算法和语料三个方面：
 
* 语言场景是指人类语言的多样性、复杂性和歧义性
* 学习算法指的是 NLP 的数理模型一般都较为难懂，比如隐马尔科夫模型(HMM)、
  条件随机场(CRF)以及基于 RNN 的深度学习模型 
* 语料的困难性指的是如何获取高质量的训练语料

## NLP 结构

从自然语言的角度出发，NLP 结构如下: 

- **自然语言理解(Natural Language Understanding，NLU)**
    - **音系学**: 指代语言中发音的系统化组织
    - **词态学**: 研究单词构成以及相互之间的关系
    - **句法学**: 给定文本的那本分是语法正确的
    - **语义学**: 给定文本的含义是什么
    - **语用学**: 文本的目的是什么
- **自然语言生成(Natural Language Generation，NLG)**
    - **自然语言文本**

> * (1) 自然语言理解涉及语言、语境和各种语言形式的学科
> * (2) 自然语言生成则从结构化数据中以读取的方式自动生成文本。该过程主要包含三个阶段: 
>     - 文本规划(完成机构化数据中的基础内容规划)
>     - 语句规划(从结构化数据中组合语句来表达信息流)
>     - 实现(产生语法通顺的语句来表达文本)

## NLP 模型

NLP 的模型基本包括两类：

* 一类是基于概率图方法的模型，主要包括贝叶斯网络、马尔科夫链、隐马尔科夫模型、EM 算法、CRF 条件随机场、
  最大熵模型等模型，在深度学习兴起之前，NLP 基本理论模型基础都是靠概率图模型撑起来的 
* 另一类则是基于 RNN 的深度学习方法，比如词嵌入(Word Embedding)、词向量(Word Vector)、word2vec、基于 RNN 的机器翻译等等

# NLP 的研究任务

* **机器翻译**: 计算机具备将一种语言翻译成另一种语言的能力
* **情感分析**: 计算机能够判断用户评论是否积极
* **智能问答**: 计算机能够正确回答输入的问题
* **文摘生成**: 计算机能够准确归纳、总结并产生文本摘要
* **文本分类**: 计算机能够采集各种文章，进行主题分析，从而进行自动分类
* **舆论分析**: 计算机能够判断目前舆论的导向
* **知识图谱**: 知识点相互连接而成的语义网路

# NLP 相关知识的构成

## 基本术语

- **分词(segment)**
    - 分词常用的方法是基于字典的最长串匹配，但是歧义分词很难
- **词性标注(part-of-speech tagging)**
    - 词性一般是指动词(noun)、名词(verb)、形容词(adjective)等
    - 标注的目的是表征词的一种隐藏状态，隐藏状态构成的转移就构成了状态转义序列
- **命名实体识别(NER，Named Entity Recognition)**
    - 命名实体是指从文本中识别具有特定类别的实体，通常是名词
- **句法分析(syntax parsing)**
    - 句法分析是一种基于规则的专家系统
    - 句法分析的目的是解析句子中各个成分的依赖关系，所以，往往最终生成的结果是一棵句法分析树
    - 句法分析可以解决传统词袋模型不考虑上下文的问题
- **指代消解(anaphora resolution)**
    - 中文中代词出现的频率很高，它的作用是用来表征前文出现过的人名、地名等
- **情感识别(emotion recognition)**
    - 情感识别本质上是分类问题。通常可以基于词袋模型+分类器，或者现在流行的词向量模型+RNN
- **纠错(correction)**
    - 基于 N-Gram 进行纠错、通过字典树纠错、有限状态机纠错
- **问答系统(QA system)**
    - 问答系统往往需要语言识别、合成、自然语言理解、知识图谱等多项技术的配合才会实现得比较好

## 知识结构

- 句法语义分析
- 关键词抽取
- 文本挖掘
- 机器翻译
- 信息检索
- 问答系统
- 对话系统
- 文档分类
- 自动文摘
- 信息抽取
- 实体识别
- 舆情分析
- 机器写作
- 语音识别
- 语音合成
- ...

# NLP 的三个层面

- **词法分析**
    - 分词
    - 词性标注
- **句法分析**
    - 短语结构句法体系
    - 依存结构句法体系
    - 深层文法句法分析
- **语义分析**
    - 语义角色标注(semantic role labeling)

# NLP 相关技术

- 文本分类
   - 朴素贝叶斯
   - 逻辑回归
   - 支持向量机
- 文本聚类
   - K-means
- 特征提取
   - Bag-of-words
   - TF
   - IDF
   - TF-IDF
   - N-Gram
- 标注
- 搜索与排序
- 推荐系统
- 序列学习
   - 语音识别
   - 文本转语音
   - 机器翻译

# NLP 常用语料库

* **中文**
    - [中文维基百科](https://dumps.wikimedia.org/zhwiki/) 
    - [搜狗新闻语料库](http://download.labs.sogou.com/resource/ca.php) 
    - [IMDB 情感分析语料库](https://www.kaggle.com/tmdb/tmdb-moive-metadata) 
    - 豆瓣读书
    - 邮件相关
* **英文**

# 参考 & TODO

* [我的NLP学习之路](https://mp.weixin.qq.com/s/QuTjgi8mr0Wwv7POGXsuCg)
* [GitHub 上有哪些有趣的关于 NLP 的Python项目](https://mp.weixin.qq.com/s/3HL3NtpyjzVqrne4-ymTeg)
* [用文本分类模型轻松搞定复杂语义分析；NLP管道模型可以退下了](https://mp.weixin.qq.com/s?__biz=MzkzMzI4MjMyNA==&mid=2247510970&idx=1&sn=1aa447379230d12ce8dedcf8a4e02b9b&source=41#wechat_redirect)
* [NLP的学习思考(新手和进阶)](https://mp.weixin.qq.com/s/p_bkGP1ABj9NZ2bBYoBLyw)
* https://fasttext.cc/docs/en/supervised-tutorial.html
* https://medium.com/@ageitgey/text-classification-is-your-new-secret-weapon-7ca4fad15788
* https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e
* https://medium.com/swlh/autonlp-sentiment-analysis-in-5-lines-of-python-code-7b2cd2c1e8ab