---
title: AdaBoost
author: wangzf
date: '2022-08-01'
slug: ml-gbm-adaboost
categories:
  - machinelearning
tags:
  - machinelearning
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

- [AdaBoost 简介](#adaboost-简介)
- [AdaBoost 模型原理](#adaboost-模型原理)
  - [模型基本步骤](#模型基本步骤)
  - [模型迭代算法](#模型迭代算法)
  - [模型具体算法](#模型具体算法)
- [参考](#参考)
</p></details><p></p>

# AdaBoost 简介

AdaBoost 是 Boosting 流派中最具代表性的一种模型。AdaBoost，是英文 "Adaptive Boosting"(自适应增强)的缩写，
由 Yoav Freund 和 Robert Schapire 在 1995 年提出

AdaBoost 的自适应在于: 前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器。
同时，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数

最初提出的 AdaBoost 算法主要用来解决二分类问题。当然，AdaBoost 算法同样可以用来解决响应变量为连续的回归问题

许多学者都研究了 AdaBoost 产生能够准确分类的分类器的原因，
他们在数据实验中发现，当使用基于决策树的分类器作为"基本学习器"(base learner) 

`$$G_{m}(\mathbf{x}), m=1,2,\ldots,M$$` 

时会使得 AdaBoost 算法比单棵决策树分类模型拥有显著低的分类误差。
Breiman 就直接将使用树模型作为基分类器的 AdaBoost 算法称为：
"世界上最好的直接可以拿来使用的分类器 "(best off-the-shelf classifier in the world)。
并且，许多关 AdaBoost 的算法实验都表明如果算法中不断有基本学习器加进来，
算法的分类误差一直在减小，从而可以得到 AdaBoost 算法似乎不容易过拟合的性质

# AdaBoost 模型原理

## 模型基本步骤

首先，初始化所有的训练数据样本的权重为 `$\omega_{i}=1/N, i=1, 2, \ldots, N$`，`$N$` 为训练样本的数量。
并利用这个加权重的训练数据训练产生一个弱分类器

然后，在算法第 `$m, m=2, 3, \ldots, M$` 步迭代中，算法每次都会根据前一次迭代训练出的弱分类器在训练数据上的分类结果重新计算训练样本权重，
并在每次拟合之前将权重 `$\omega_{1}, \omega_{2}, \ldots, \omega_{N}$` 作用在每个训练数据观测值 `$(x_{i}, y_{i}), i = 1,2,\ldots,N$` 上，
然后不断将弱分类算法应用在这些加权之后的训练数据上，重新进行拟合分类。
那些在前一步迭代中被弱分类器 `$G_{m-1}(\mathbf{x})$` 误分类的的样本观测值的权重将会增大，
而那些被正确分类的观测值的权重将减小。因此，随着迭代过程的进行，那些很难正确分类的样本观测值受到的关注也越来越大，
即样本权重越来越大。因此在序列中每个分类器将被迫重点关注这些很难被之前分类器正确分类的训练数据观测值

最后，再将这些弱分类器的分类结果进行加权组合，得到最终的强分类器

## 模型迭代算法

1. 初始化训练数据的权值分布
    - 如果有 `$N$` 个样本，则每一个训练样本最开始时都被赋予相同的权值 `$1/N$`
2. 训练弱分类器
    - 具体训练过程中，如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它的权值就被降低；
      相反，如果某个样本点没有被准确地分类，那么它的权值就得到提高
    - 然后，权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去
3. 将各个训练得到的弱分类器组合成强分类器
    - 各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用；
      降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。
      换言之，误差率低的弱分类器在最终分类器中占的权重较大，否则较小

## 模型具体算法

1. 初始化每个样本观测值的权重 `$\omega_{i}^{[0]}=1/N$`，其中 `$i=1, \ldots, N$`，`$N$` 为训练样本的数量
2. 开始迭代，令 `$m=1$`，`$m$` 为迭代次数
    - 2.1 利用加权重 `$\omega_{i}^{[m-1]}$` 的训练数据拟合一个弱分类器 `$G_{m}(\cdot)$` 
    - 2.2 计算加权训练数据的分类错误率：
  
    `$$err^{[m]} = \frac{\sum_{i=1}^{N}\omega_{i}^{[m-1]}I(y_{i} \neq G_{m}(x_{i}))}{\sum_{i=1}^{N}\omega_{i}^{[m-1]}}.$$`

    - 2.3 根据上一步的分类错误率计算分类器在最终分类结果上的权重值：
        
    `$$\alpha^{[m]}=log(\frac{1-err^{[m]}}{err^{[m]}}).$$`

    - 2.4 根据上面计算出的样本的误分类率更新样本的权重：

    `$$\omega_{i}^{[m]} \leftarrow \omega_{i}^{[m-1]}e^{\alpha^{[m]}I(y_{i} \neq G_{m}(\mathbf{x}_{i}))}$$`

3. 重复上面的迭代，直到满足迭代停止条件 `$m=M$`，并且将所有在迭代中产生的弱分类器通过加权投票的方法进行聚合，最终得到的强分类器为：

`$$G(\mathbf{x}) = sign\Bigg(\sum^{M}_{m=1}\alpha^{[m]}G^{[m]}(\mathbf{x})\Bigg)$$`


# 参考

* [原始论文]()

