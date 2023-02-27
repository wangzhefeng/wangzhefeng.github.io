---
title: KNN
author: 王哲峰
date: '2023-02-26'
slug: ml-knn
categories: 
    - machinelearning
tags: 
    - model
---

<style>
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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
</style>

<details><summary>目录</summary><p>

- [KNN 介绍](#knn-介绍)
- [KNN 算法](#knn-算法)
  - [输入](#输入)
  - [输出](#输出)
- [KNN 模型](#knn-模型)
  - [模型](#模型)
  - [距离度量](#距离度量)
  - [k 值的选择](#k-值的选择)
  - [分类决策规则](#分类决策规则)
- [KNN 的实现](#knn-的实现)
- [参考](#参考)
</p></details><p></p>

# KNN 介绍

`$k$` 近邻法(k-nearest neighbor, k-NN) 1968 年由 Cover 和 Hart 提出。
KNN 是一种基本分类与回归方法。这里只讨论分类问题中的 KNN。
KNN 的输入为实例的特征向量，对应于特征空间的点；输出位实例的类别，可以取多类

KNN 假设给定一个训练数据集，其中的实例类别已定。分类时，对新的实例，根据其 `$k$` 个最近邻的训练实例的类别，
通过多数表决等方法进行预测，因此，KNN 不具有显式的学习过程。KNN 实际上利用训练数据集对特征向量空间进行划分，
并作为其分类的“模型”

`$k$` 值的选择、距离度量及分类决策规则是 KNN 的三个基本要素

# KNN 算法

KNN 算法简单、直观：给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最近邻的 `$k$` 个实例，
这 `$k$` 个实例的多数属于某个类，就把该输入实例分为这个类

具体算法如下：

## 输入

训练数据集

`$$T=\{(x_{1}, y_{1}), (x_{2}, y_{2}), \ldots, (x_{N}, y_{N})\}$$`

其中：

* `$x_{i} \in \mathcal{X} \subseteq R^{n}$` 为实例的特征向量，
  `$y_{i} \in \mathcal{Y} = \{c_{1}, c_{2}, \ldots, c_{K}\}$` 为实例的类别，
  `$i= 1,2,\ldots, N$`
* 实例特征向量 `$x$`

## 输出

实例 `$x$` 所属的类 `$y$`

1. 根据给定的距离度量，在训练集 `$T$` 中找出与 `$x$` 最近邻的 `$k$` 个点，
   盖这 `$k$` 个点的 `$x$` 的邻域记作 `$N_{k}(x)$`
2. 在 `$N_{k}(x)$` 中根据分类决策规则（ 如多数表决）决定 `$x$` 的类别 `$y$`

`$$y=arg \underset{c_{j}}{max}\underset{x_{i} \in N_{k}(x)}{\sum}I(y_{i} = c_{j}), \quad i =1,2,\ldots,N; j=1,2,\ldots,K$$`

其中：

* `$I$` 为指示函数，即当 `$y_{i} = c_{j}$` 时 `$I$` 为 1，否则 `$I$` 为 0

KNN 的特殊情况是 `$k=1$` 的情形，称为最近邻法。对于输入的实例点（特征向量）`$x$`，
最近邻法将训练数据集中与 `$x$` 最邻点的类作为 `$x$` 的类

KNN 没有显式的学习过程

# KNN 模型

KNN 适用的模型实际上对应于对特征空间的划分。模型由三个基本要素——距离度量、`$k$` 值的选择和分类决策规则

## 模型

KNN 中，当训练集、距离度量（如欧氏距离）、`$k$` 值及分类决策规则（多数表决）确定后，
对于任何一个新的输入实例，它所属的类唯一地确定。这相当于根据上述要素将特征空间划分为一些子空间，
确定子空间里的每个点所属的类。这一事实从最近邻算法中可以看得很清楚

特征空间重，对每个训练实例点 `$x_{i}$`，距离该点比其他点更近的所有点组成一个区域，叫作单元（cell）。
每个训练实例点拥有一个单元，所有训练实例点的单元构成对特征空间的一个划分。
最近邻法将实例 `$x_{i}$` 的类 `$y_{i}$` 作为其单元中所有点的类标记（class label）。
这样，每个单元的实例点的类别是确定的

## 距离度量

特征空间中两个实例点的距离是两个实例点相似程度的反映。
KNN 模型的特征空间一般是 `$n$` 维实数向量空间 `$R^{n}$`。
使用的距离是欧氏距离，但也可以是其他距离，
如更一般的 `$L_{p}$` 距离（`$L_{p}$` distance）或 Minkowski 距离（Minkowski distance）

设特征空间 `$\mathcal{X}$` 是 `$n$` 维实数向量空间 `$R^{n}$`，
`$x_{i},x_{j} \in \mathcal{X}$`，`$x_{i}=(x_{i}^{(1)},x_{i}^{(2)}, \ldots, x_{i}^{(n)})^{T}$`，
`$x_{j}=(x_{j}^{(1)},x_{j}^{(2)}, \ldots, x_{j}^{(n)})^{T}$`

`$x_{i}$`，`$x_{j}$` 的 `$L_{p}$` 距离定义为：

`$$L_{p}(x_{i}, x_{j})=\Bigg(\sum_{l=1}^{n}|x_{i}^{(l)} - x_{j}^{(l)}|^{p}\Bigg)^{\frac{1}{p}}，p \geq 1$$`

当 `$p=2$` 时，称为欧氏距离（Euclidean distance），即

`$$L_{2}(x_{i}, x_{j}) = \Bigg(\sum_{l=1}^{n}|x_{i}^{(l)} - x_{j}^{(l)}|^{2}\Bigg)^{\frac{1}{2}}$$`

当 `$p=1$` 时，称为曼哈顿距离（Manhattan distance），即

`$$L_{1}(x_{i}, x_{j}) = \sum_{l=1}^{n}|x_{i}^{(l)} - x_{j}^{(l)}|$$`

当 `$p=\infty$` 时，它是各个坐标距离的最大值，即：

`$$L_{\infty}(x_{i}, x_{j}) = \underset{l}{max}|x_{i}^{(l)} - x_{j}^{(l)}|$$`

## k 值的选择

`$k$` 值的选择会对 KNN 的结果产生重大影响

如果选择较小的 `$k$` 值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差（approximation error）会减小，
只有与输入实例较近的（相似的）训练实例才会对预测结果起作用。但缺点是 “学习”的估计误差（estimation error）会增大，
预测结果会对近邻的实例点非常敏感。如果邻近的实例点恰巧是噪声，预测就会出错。换句话说，`$k$` 值的减小意味着整体模型变得复杂，
容易发生过拟合

如果选择较大的 `$k$`，就相当于用较大的邻域中的训练实例进行预测。其优点是可以减少学习的估计误差，但缺点是学习的近似误差会增大。
这时与输入实例较远的（不相似的）训练实例也会对预测起作用，使预测发生错误。`$k$` 值的增大就意味着整体的模型变得简单

如果 `$k=N$`，那么无论输入实例是什么，都将简单地预测它属于在训练实例中最多的类。这时，模型过于简单，
完全忽略训练实例中的大量有用信息，是不可取的

在应用中，`$k$` 值一般取一个比较小的数值。通常采用交叉验证法来选取最优的 `$k$` 值

## 分类决策规则

KNN 中的分类决策规则往往是多数表决的，即由输入实例的 `$k$` 个邻近的训练实例中的多数类决定输入实例的类

多数表决规则（majority voting rule）有如下解释：如果分类的损失函数为 `$0-1$` 损失函数，分类函数为

`$$f: R^{n} \rightarrow \{c_{1}, c_{2}, \ldots, c_{K}\}$$`

那么误分类的概率是

`$$P(Y \neq f(X))=1-P(Y = f(X))$$`

对给定的实例 `$x \in \mathcal{X}$`，其最近邻的 `$k$` 个训练实例点构成集合 `$N_{k}(x)$`。
如果涵盖 `$N_{k}(x)$` 的区域的类别是 `$c_{j}$`，那么误分类率是

`$$\frac{1}{k} \underset{x_{i} \in N_{k}(x)}{\sum}I(y_{i} \neq c_{j}) = 1- \frac{1}{k}\underset{x_{i} \in N_{k}(x)}{\sum}I(y_{i} = c_{j})$$`

要使误分类率最小即经验风险最小，就要 `$\underset{x_{i} \in N_{k}(x)}{\sum}I(y_{i} = c_{j})$` 最大，
所以多数表决规则等价于经验风险最小化

# KNN 的实现

> kd 树

实现 KNN 时，主要考虑的问题是如何对训练数据进行快速 `$k$` 近邻搜索。
这点在特征空间的维数大及训练数据容量大时尤其重要

KNN 最简单的实现方法是线性扫描（linear scan）。这时要计算输入实例与每一个训练实例的距离。
当训练集很大时，计算非常耗时，这种方法是不可行的

为了提高 KNN 搜索的效率，可以考虑使用特殊的结构存储训练数据，以减少计算距离的次数。
具体方法很多，比如：kd 树（kd tree）方法

# 参考

* [统计学习方法-李航]()

