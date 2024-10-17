---
title: 聚类评价指标
author: wangzf
date: '2022-11-22'
slug: cluster
categories:
  - machinelearning
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

- [评价指标简介](#评价指标简介)
- [外部指标](#外部指标)
  - [Rand Index](#rand-index)
  - [Adjusted Rand Index](#adjusted-rand-index)
  - [Jaccard Coefficient](#jaccard-coefficient)
  - [Fowlkes-Mallows Index](#fowlkes-mallows-index)
- [内部指标](#内部指标)
  - [Davies-Bouldin Index](#davies-bouldin-index)
  - [Dunn Index](#dunn-index)
  - [Silhouette Coefficient](#silhouette-coefficient)
  - [Calinski-Harabasz Index](#calinski-harabasz-index)
- [其他](#其他)
  - [Homogeneity 和 Completeness](#homogeneity-和-completeness)
  - [V-Measure](#v-measure)
  - [Mutual Information](#mutual-information)
  - [Adjusted Mutual Information](#adjusted-mutual-information)
  - [Normalized Mutual Information](#normalized-mutual-information)
  - [Contingency Matrix](#contingency-matrix)
  - [Pair Confusion Matrix](#pair-confusion-matrix)
- [参考](#参考)
</p></details><p></p>

# 评价指标简介

聚类性能度量亦称聚类"有效性指标"(validity index)

设置聚类性能度量的目的：

* 对聚类结果，通过某种性能度量来评估其好坏。若明确了最终将要使用的性能度量，
则可直接将其作为聚类过程的优化目标，从而更好地得到符合要求的聚类结果

什么样的聚类结果比较好：

* 簇内相似度(intra-cluster similarity)高
* 蔟间相似度(inter-cluster similarity)低

聚类性能度量指标类型：

* 外部指标(external index)
    - 将聚类结果与某个"参考模型"(reference model)进行比较
    - 度量指标的结果值均在 `$[0,1]$` 区间，值越大越好
* 内部指标(internal index)
    - 直接考察聚类结果而不利用任何参考模型

# 外部指标

对数据集 `$D = \{x_1, x_2, \ldots, x_n\}$`， 其中：`$n$` 为样本数量

假定：

* 通过聚类给出的簇划分为 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`，其中：`$k$` 为簇个数
* 参考模型给出的簇划分为 `$C^{*}=\{C_{1}^{*}, C_{2}^{*}\, \ldots, C_{k}^{*}\}$`，其中：`$k$` 为簇个数

相应地，令 `$\lambda=\{\lambda_{1}, \lambda_{2}, \ldots, \lambda_{n}\}$` 与 `$\lambda^{*}=\{\lambda_{1}^{*}, \lambda_{2}^{*}, \ldots, \lambda_{n}^{*}\}$` 分别表示与 `$C$` 和 `$C^{*}$` 对应的簇标记向量。
将样本两两配对考虑，定义：

`$$a=|SS|，SS=\{(x_{i}, x_{j}) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\}$$`
`$$b=|SD|，SD=\{(x_{i}, x_{j}) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\}$$`
`$$c=|DS|，DS=\{(x_{i}, x_{j}) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\}$$`
`$$d=|DD|，DD=\{(x_{i}, x_{j}) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\}$$`

其中: 

* 集合 `$SS$` 包含了在 `$C$` 中隶属于相同簇，且在 `$C^{*}$` 中也隶属于相同簇的样本对
* 集合 `$SD$` 包含了在 `$C$` 中隶属于相同簇，但在 `$C^{*}$` 中隶属于不同簇的样本对
* 集合 `$DS$` 包含了在 `$C$` 中隶属于不同簇，但在 `$C^{*}$` 中隶属于相同簇的样本对
* 集合 `$DD$` 包含了在 `$C$` 中隶属于不同簇，且在 `$C^{*}$` 中也隶属于不同簇的样本对

这样，由于每个样本对 `$(x_{i}, x_{j}), i<j$` 仅能出现在一个集合中，因此有：

`$$a+b+c+d=\frac{n(n-1)}{2}$$`

## Rand Index

> Rand Index，RI，Rand 指数

Rand Index 是一种衡量聚类算法性能的指标。它衡量的是聚类算法将数据点分配到聚类中的准确程度。
Rand Index 的范围为 `$[0, 1]$`，如果 Rand Index 为 1 表示两个聚类完全相同，接近 0 表示两个聚类有很大的不同

`$$RI = \frac{a + b}{C_{2}^{n}}=\frac{2(a+d)}{n(n-1)}$$`

其中：

* `$a$`：在真实标签中处于同一簇中的样本对数，在预测聚类中处于同一簇中的样本对数
* `$b$`：真实聚类和预测聚类中处于不同聚类的样本对的数目

## Adjusted Rand Index

Adjusted Rand Index（调整兰德指数）是一种用于衡量聚类算法性能的指标，
它是 Rand Index 的一种调整形式，可以用于评估将样本点分为多个簇的聚类算法。
它考虑了机会的概率，取值范围为 `$[-1,1]$`，其中值越接近 1 表示聚类结果越准确，
值越接近 0 表示聚类结果与随机结果相当，值越接近 -1 表示聚类结果与真实类别完全相反

`$$Adj\_RI = \frac{(RI - Expected RI)}{max(RI) - Expected RI}$$`

## Jaccard Coefficient

> Jaccard 系数(Jaccard Coefficient 简称 JC)

`$$JC=\frac{a}{a+b+c}$$`

## Fowlkes-Mallows Index

> FM 指数(Fowlkes and Mallows Index 简称 JMI)

Fowlkes-Mallows 分数是这个是最容易理解的，它主要基于数据的真实标签和聚类结果的交集、
联合集以及簇内和簇间点对数的比值来计算。由以下公式得到

`$$FMI = \sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$$`
`$$FMI = \frac{TP}{\sqrt{(TP + FP)(TP + FN)}}$$`

# 内部指标

根据聚类结果的簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$` ,定义: 

* 簇 `$C$` 内样本间的平均距离 

`$$avg(C)=\frac{2}{|C|(|C|-1)}\sum_{1<i<j<|C|}dist(x_{i}, x_{j})$$`

* 簇 `$C$` 内样本间的最远距离 

`$$diam(C)=max_{1<i<j<|C|}dist(x_{i}, x_{j})$$`

* 簇 `$C_{i}$` 与簇 `$C_{j}$` 最近样本间的距离 

`$$d_{min}(C_{i}, C_{j})=min_{1<i<j<|C|}dist(x_{i}, x_{j}$$`

* 簇 `$C_{i}$` 与簇 `$C_{j}$` 中心点间的距离 

`$$d_{cen}(C_{i}, C_{j})=dist(\mu_{i}, \mu_{j})$$`

其中: 

* `$dist(,)$` 是两个样本之间的距离
* `$\mu$` 是簇 `$C$` 的中心点 `$\mu=\frac{1}{|C|}\sum_{1<i<|C|}x_{i}$`

## Davies-Bouldin Index

> DB 指数(Davies-Bouldin Index 简称 DBI)

DB 指数主要基于簇内的紧密程度和簇间的分离程度来计算。DB 指数的取值范围是 `$[0, +\infty]$`，
值越小表示聚类结果越好

`$$DB = \frac{1}{k}\sum_{i=1}^{k}\underset{i \neq j}{\max}R_{ij}$$`

`$$R_{ij} = \frac{s_{i} + s_{j}}{d_{ij}}$$`

> `$$DBI=\frac{1}{k}\sum^{k}_{i=1}\underset{j \neq i}{max}\bigg(\frac{avg(C_{i})+avg(C_{j})}{d_{cen(\mu_{i}, \mu_{j})]}}\bigg)$$`

## Dunn Index

> Dunn 指数(Dunn Index 简称 DI)

`$$DI=\underset{1 \leqslant i \leqslant k}{min}\bigg\{\underset{j \neq i}{min}\bigg(\frac{d_{min}(C_{i}, C_{j})}{max_{1 eqslant l \leqslant k}diam(C_{l})}\bigg)\bigg\}$$`

* DI 的值越大越好

## Silhouette Coefficient

轮廓分数使用同一聚类中的点之间的距离，以及下一个临近聚类中的点与所有其他点之间的距离来评估模型的表现。
它主要基于样本点与其所属簇内和最近邻簇之间的距离、相似性和紧密度等因素来计算。数值越高，模型性能越好。
一般公式为

`$$s = \frac{b-a}{max(a, b)}$$`

其中：

* `$a$` 为样本与同类中所有其他点之间的平均距离
* `$b$` 为样本与下一个最近聚类中所有其他点之间的平均距离

## Calinski-Harabasz Index

Calinski-Harabasz Index（也称 Variance Ratio Criterion）是一种用于衡量聚类算法性能的指标，
它主要基于簇内和簇间方差之间的比值来计算。Calinski-Harabasz Index 的取值范围为 `$[0, +\infty)$`，
值越大表示聚类结果越好

`$$s = \frac{tr(B_{k})}{tr(W_{k})} \times \frac{n_{E} - k}{k-1}$$`

其中：`$tr(B)$` 和 `$tr(W)$` 分别表示簇间协方差矩阵和簇内协方差矩阵的迹

Calinski-Harabasz Index 也可以用于确定聚类的最优簇数。通过计算不同簇数下的 Calinski-Harabasz Index，
可以选择使得 Calinski-Harabasz Index 最大的簇数作为最终的聚类结果

# 其他

## Homogeneity 和 Completeness

Homogeneity Score 和 Completeness Score 是两个用于评估聚类结果的指标，
它们可以用于衡量聚类算法对数据的划分是否“同质”（即簇内只包含同一类别的样本点），
以及是否“完整”（即同一类别的样本点是否被划分到同一个簇中）。
该指标的取值范围也为 `$[0,1]$`，值越大表示聚类结果越好。需要注意的是，
该指标对簇的数量较为敏感，因此在使用时需要结合实际问题进行调参

同质性和完整性是给定公式的两个相关度量：

`$$Homogeneity = 1 - \frac{H(C|K)}{H(C)}$$`
`$$Completeness = 1 - \frac{H(K|C)}{H(K)}$$`

为了获得同质性和完整性，我们需要找到：

* 真实标签的熵(H)：`$H(K)$`
* 预测标签的熵(H)：`$H(C)$`
* 给定预测标签的真实标签的条件联合熵(CJH)：`$H(C|K)$`
* 给定真实标签的预测标签的条件联合熵(CJH)：`$H(K|C)$`

```python
import math
from collections import Counter
import numpy as np

def entropy(arr):
    # Find unique values and their counts:
    unique, counts = np.unique(arr, return_counts = True)
    # Get the probability for each cluster (unique value):
    p = counts / len(arr)
    # Apply entropy formula:
    entropy = -np.sum(p * np.log2(p))
    return entropy


def conditional_entropy(X, Y):
    """
    计算联合熵
    """
    # Build a 2D-numpy array with true clusters and predicted clusters
    XY = np.column_stack((X, Y))
    # Count the number of observations in X and Y with the same values
    xy_counts = Counter(map(tuple, XY))
    # Get the joint probability
    joint_prob = np.array(list(xy_counts.values())) / len(XY)
    # Get conditional probability
    y_counts = Counter(Y)
    conditional_prob = np.zeros_like(joint_prob)
    for i, (x, y) in enumerate(xy_counts.keys()):
        conditional_prob[i] = xy_counts[(x, y)] / y_counts[y]
    # Get conditional entropy
    conditional_entropy = -np.sum(joint_prob * np.log2(conditional_prob + 1e-10))
    return conditional_entropy

# 计算真实标签、预测标签的熵
entropy_y_true = entropy(y_true)
entropy_km_labels = entropy(km_labels)
print(f'Entropy for y_true: {entropy_y_true}')
print(f'Entropy for km_labels: {entropy_km_labels}')

# 计算联合熵
joint_entropy_y_true = conditional_entropy(y_true, km_labels)
joint_entropy_km_labels = conditional_entropy(km_labels, y_true)
print(f'Joint entropy for y_true given km_labels is: {joint_entropy_y_true}')
print(f'Joint entropy for km_labels given y_true is: {joint_entropy_km_labels}')

# 计算同质性和完整性值
homogeneity = 1 - (joint_entropy_y_true / entropy_y_true)
completeness = 1 - (joint_entropy_km_labels / entropy_km_labels)
print('homogeneity: ', homogeneity)
print('Completeness: ', completeness)
```

## V-Measure

V-Measure 是一种综合考虑同质性和完整性的评估指标，可以用于衡量聚类算法对数据的划分质量。
V-Measure 的取值范围为 `$[0,1]$`，值越大表示聚类结果越好

`$$\upsilon = \frac{(1 + \beta) \times homogeneity \times completeness}{\beta \times homogeneity + completeness}$$`

## Mutual Information

基于互信息的分数（Mutual Information）是一种用于衡量聚类算法性能的指标，
它衡量的是聚类结果与真实标签之间的相似性。基于互信息的分数可以用于评估将样本点分为多个簇的聚类算法

基于互信息的分数的取值范围为 `$[0, 1]$`，其中值越接近 1 表示聚类结果越准确，
值越接近 0 表示聚类结果与随机结果相当，值越小表示聚类结果与真实类别之间的差异越大。
基于互信息的分数是一种相对指标，它的取值受到真实类别数量的影响。
当真实类别数量很大时，基于互信息的分数可能会受到偏差

`$$MI(U, V) = \sum_{i=1}^{|U|}\sum_{j=1}^{|V|}\frac{|U_{i} \cap V_{j}|}{N}log\frac{N|U_{i} \cap V_{j}|}{|U_{i}||V_{j}|}$$`

## Adjusted Mutual Information

调整互信息分数(Adjusted Mutual Information Score)是一种用于衡量聚类算法性能的指标，
它是基于互信息的分数的一种调整形式，AMI 不受标签数值的影响，即使标签重新排序，也会产生相同的分数。
公式如下所示，其中 `$E$` 代表预期:

`$$AMI(U, V) = \frac{MI(U, V) - E(MI(U, V))}{avg(H(U), H(V)) - E(MI(U, V))}$$`

## Normalized Mutual Information

标准化互信息分数(Normalized Mutual Information Score)是基于互信息的分数的一种标准化形式，
可以用于评估将样本点分为多个簇的聚类算法。它知识将互信息分数进行了标准化，
在 0(表示没有互信息)和 1(表示完全相关)之间进行缩放。与基于互信息的分数相比，
标准化互信息分数更加稳健，不受真实类别数量的影响

`$$NMI = \frac{MI(U, V)}{\frac{-\sum_{}^{} U logU - \sum_{}^{} V log V}{2}}$$`

## Contingency Matrix

## Pair Confusion Matrix

# 参考

* [10个聚类算法的评价指标](https://mp.weixin.qq.com/s/mGZx80fEQIl7D9ib1Nvx6A)
