---
title: 深度学习
subtitle: Deep Learning
author: 王哲峰
date: '2023-02-24'
slug: dl
categories:
  - deeplearning
tags:
  - note
---

# 业务方向

根据业务和技术方向的差异性，目前工业界的算法工程师主要可以分成：

* 广告算法工程师
* 推荐算法工程师
* 风控算法工程师
* CV 算法工程师
* NLP 算法工程师
* ...

除了一些多模态数据外(文本、图像、社交网络等)，广告算法工程师、推荐算法工程师、
风控算法工程师处理的数据类型主要还是是结构化数据。
风控领域目前工业界用到的模型还是树模型为主，广告和推荐领域目前工业界是深度模型为主。
由于数据结构的相似性，广告的 CTR 预估模型和推荐系统的精排模型，基本是通用的。
广告、推荐领域常用的神经网络模型为：

* FM
* DeepFM
* FiBiNET
* DeepCross
* DIN
* DIEN

# 深度学习领域

## 深度理解深度学习的一些文章

* 用信息论的知识去分析高维数据
    - [Discovering Structure in High-Dimensional Data Through Correlation Explanation](https://arxiv.org/abs/1406.1222)
* 用 group theory 去分析为什么深度模型会有效
    - [Learning the Irreducible Representations of Commutative Lie Groups](https://arxiv.org/abs/1402.4437)
* 分析 model compression 的问题
    - [Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184)
* 分析每一层 layer 学出来的 representation
    - [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
* 用复杂理论来分析 dropout
    - [Dropout Rademacher Complexity of Deep Neural Networks](https://arxiv.org/abs/1402.3811)
* 分析什么情况下深度模型会失败
    - [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)

# 参考

* [机器学习领域，如何选择研究方向](https://www.zhihu.com/question/28689201)

