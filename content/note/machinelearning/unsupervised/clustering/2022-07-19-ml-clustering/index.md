---
title: Cluster
author: 王哲峰
date: '2022-07-19'
slug: ml-clustering
categories:
  - machinelearning
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
</style>


<details><summary>目录</summary><p>

- [聚类算法](#聚类算法)
- [聚类数据](#聚类数据)
- [聚类性能度量](#聚类性能度量)
  - [设置聚类性能度量的目的](#设置聚类性能度量的目的)
  - [什么样的聚类结果比较好](#什么样的聚类结果比较好)
  - [聚类性能度量指标](#聚类性能度量指标)
    - [外部指标](#外部指标)
    - [内部指标](#内部指标)
- [聚类距离计算](#聚类距离计算)
  - [闵可夫斯基距离(Minkowski distance)](#闵可夫斯基距离minkowski-distance)
  - [VDM(Value Difference Metric)](#vdmvalue-difference-metric)
  - [闵可夫斯基距离与VDM混合距离](#闵可夫斯基距离与vdm混合距离)
  - [加权闵可夫斯基距离](#加权闵可夫斯基距离)
</p></details><p></p>

# 聚类算法

* 聚类是从数据集中挖掘相似观测值集合的方法.
* 聚类试图将数据集中的样本划分为若干个通常是不相交的子集, 每个子集称为一个"簇"(cluster). 通过这样的划分, 
  每个簇可能对应于一些潜在的概念(类别). 
* 聚类过程仅能自动形成簇结构, 簇所对应的概念语义需由使用者自己来把握. 
* 聚类既能作为一个单独的过程用于寻找数据内在的分布结构, 也可以作为分类等其他学习任务的前驱过程. 


角度I:

* 基于原型的聚类(Prototype-based Clustering)
    - K 均值聚类(K-Means)
    - 学习向量量化聚类(Learning Vector Quantization)
    - 高斯混合模型聚类 (Gaussian Mixture Model)
* 基于密度的聚类 (Density-based Clustering)
    - DBSCAN (Density-Based Spatial Clustering of Application with Noise)
    - OPTICS (Ordering Points To Identify the Clustering Structure)
* 层次聚类 (Hierarchical Clustering)
* 基于模型的聚类 (Model-based Clustering)
    - 混合回归模型 (Mixture Regression Model)

[角度II](http://lchiffon.github.io/2014/12/28/cluster-analysis.html):

* 基于中心的聚类
    - K-Means 聚类
* 基于分布的聚类
    - GMM 聚类
* 基于密度的聚类
    - DBSCAN
    - OPTICS
* 基于连通性的聚类
    - 层次聚类
* 基于模型的聚类
    - Miture Regression Model
* 其他聚类方法
    - 谱聚类
    - Chameleon
    - Canopy
    - ...

# 聚类数据

假定样本集 `$D$` 包含 `$n$` 个无标记样本:

`$$D = \{x_1, x_2, \ldots, x_n\}$$`

每个样本是一个 `$p$` 维特征向量:

`$$x_i=(x_{i1}; x_{i2}; \ldots; x_{ip})$$`

聚类算法将样本集 `$D$` 划分为 `$k$` 个不相交的簇:

`$$\{C_l|l=1, 2, \ldots, k\}$$`

其中: 

* `$C_{l^{'}} \cap_{l^{'} \neq l} C_{l} = \emptyset$`
* `$D=\cup_{l=1}^{k}C_{l}$`

相应的, 用

`$$\lambda_{i} \in \{1, 2, \ldots, k\}$$`

表示样本 `$x_{i}$` 的"簇标记”(cluster label), 即

`$$x_{i} \in C_{\lambda_{i}}$$`

于是, 聚类的结果可用包含 `$n$` 个元素的簇标记向量表示

`$$\lambda=(\lambda_{1}, \lambda_{2}, \ldots, \lambda_{n})$$`

# 聚类性能度量

> 聚类性能度量亦称聚类"有效性指标"(validity index)

## 设置聚类性能度量的目的

对聚类结果, 通过某种性能度量来评估其好坏。若明确了最终将要使用的性能度量, 
则可直接将其作为聚类过程的优化目标, 从而更好地得到符合要求的聚类结果

## 什么样的聚类结果比较好

* "簇内相似度"(intra-cluster similarity)高
* "蔟间相似度"(inter-cluster similarity)低

## 聚类性能度量指标

* "外部指标"(external index): 将聚类结果与某个"参考模型"(reference model)进行比较
* "内部指标"(internal index): 直接考察聚类结果而不利用任何参考模型

### 外部指标

对数据集 `$D = \{x_1, x_2, \ldots, x_n\}$`, 
假定通过聚类给出的簇划分为 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`, 
参考模型给出的簇划分为 `$C^{*}=\{C_{1}^{*}, C_{2}^{*}\, \ldots, C_{k}^{*}\}$`. 

相应地, 令 `$\lambda$` 与 `$\lambda^{*}$` 分别表示与 `$C$` 和 `$C^{*}$` 对应的簇标记向量, 
将样本两两配对考虑, 定义: 

`$$a=|SS|, SS=\{(x_{i}, x_{j}) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\}$$`
`$$b=|SD|, SD=\{(x_{i}, x_{j}) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\}$$`
`$$c=|DS|, DS=\{(x_{i}, x_{j}) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\}$$`
`$$d=|DD|, DD=\{(x_{i}, x_{j}) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\}$$`

其中: 

* 集合 `$SS$` 包含了在 `$C$` 中隶属于相同簇且在 `$C^{*}$` 中也隶属于相同簇的样本对；
* 集合 `$SD$` 包含了在 `$C$` 中隶属于相同簇但在 `$C^{*}$` 中隶属于不同簇的样本对；
* 集合 `$DS$` 包含了在 `$C$` 中隶属于不同簇但在 `$C^{*}$` 中隶属于相同簇的样本对；
* 集合 `$DD$` 包含了在 `$C$` 中隶属于不同簇且在 `$C^{*}$` 中也隶属于不同簇的样本对；

这样, 由于每个样本对 `$(x_{i}, x_{j})(i<j)$` 仅能出现在一个集合中, 因此有: 

`$$a+b+c+d=n(n-1)/2$$`

**Jaccard 系数(Jaccard Coefficient 简称 JC)**

`$$JC=\frac{a}{a+b+c}$$`

**FM 指数(Fowlkes and Mallows Index 简称 JMI)**

`$$FMI=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$$`

**Rand 指数(Rand Index 简称 RI)**

`$$RI=\frac{2(a+d)}{n(n-1)}$$`

**说明:**

上述性能度量指标的结果值均在 `$[0,1]$` 区间, 值越大越好. 

### 内部指标

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

**DB指数(Davies-Bouldin Index 简称DBI)**

`$$DBI=\frac{1}{k}\sum^{k}_{i=1}\underset{j \neq i}{max}\bigg(\frac{avg(C_{i})+avg(C_{j})}{d_{cen(\mu_{i}, \mu_{j})]}}\bigg)$$`

**Dunn指数(Dunn Index 简称DI)**

`$$DI=\underset{1 \leqslant i \leqslant k}{min}\bigg\{\underset{j \neq i}{min}\bigg(\frac{d_{min}(C_{i}, C_{j})}{max_{1 eqslant l \leqslant k}diam(C_{l})}\bigg)\bigg\}$$`

说明:

* DBI 的值越小越好
* DI 的值越大越好

# 聚类距离计算

距离度量(distance measure)函数 `$dist(,)$` 需满足的基本性质:

* **非负性**: `$dist(x_{i}, x_{j}) \geqslant 0$`
* **同一性**: `$dist(x_{i}, x_{j})=0$` 当且仅当 `$x_{i}=x_{j}$`
* **对称性**: `$dist(x_{i}, x_{j})=dist(x_{j}, x_{i})$`
* **直递性**: `$dist(x_{i}, x_{j}) \leqslant dist(x_{i}, x_{k}) + dist(x_{k}, x_{j})$` (可不满足)

变量属性:

* 连续属性: 闵可夫斯基距离
* 离散属性
    - 有序属性: 闵可夫斯基距离
    - 无序属性: VDM(Value Difference Metric)
* 混合属性: 闵可夫斯基距离与VDM混合距离

## 闵可夫斯基距离(Minkowski distance)

样本: 

`$$x_{i}=(x_{i1}, x_{i2}, \ldots, x_{ip})$$`
`$$x_{j}=(x_{j1}, x_{j2}, \ldots, x_{jp})$$`

* `$q \geqslant 1$`: **闵可夫斯基距离(Minkowski distance)**

`$$dist_{mk}(x_{i}, x_{j})=\bigg(\sum_{u=1}^{p}|x_{iu}-x_{ju}|^{q}\bigg)^{\frac{1}{q}}, \quad q \geqslant 1$$`

* `$q=1$` :**曼哈顿距离(Manhattan distance):**

`$$dist_{man}(x_{i}, x_{j})=\|x_{i}-x_{j}\|_{1}=\sum^{p}_{u=1}|x_{iu}-x_{ju}|$$`

* `$q=2$` :**欧氏距离(Euclidean distance):**

`$$dist_{ed}(x_{i}, x_{j})=\|x_{i}-x_{j}\|_{2}=\sqrt{\sum^{p}_{u=1}|x_{iu}-x_{ju}|^{2}}$$`

## VDM(Value Difference Metric)

令 `$m_{u,a}$` 表示在属性 `$u$` 上取值为 `$a$` 的样本数, 
`$m_{u, a, i}$` 表示在第 `$i$` 个样本簇中在属性 `$u$` 上取值为 `$a$` 的样本数, 
`$k$` 为样本簇数, 则属性 `$u$` 上两个离散值 `$a$` 与 $b$ 之间的VDM距离为: 

`$$VDM_{q}(a, b)=\sum^{k}_{i=1}\bigg|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\bigg|^{q}$$`

## 闵可夫斯基距离与VDM混合距离

假设有 `$p_{c}$` 个有序属性, `$p-p_{c}$` 个无序属性, 有序属性排列在无序属性之前: 

`$$MinkovDM_{q}(x_{i}, x_{j})=\bigg(\sum^{p_{c}}_{u=1}|x_{i,u}-x_{j,u}|^{q}+\sum^{p}_{u=p_{c}+1}VDM_{q}(x_{i,u},x_{j,u})\bigg)^{\frac{1}{q}}$$`

## 加权闵可夫斯基距离

当样本在空间中不同属性的重要性不同时: 

`$$dist_{wmk}(x_{i}, x_{j})=(w_{1}\cdot|x_{i1}-x_{j1}|^{q}+w_{2}\cdot|x_{i2}-x_{j2}|^{q}+\ldots+w_{n}\cdot|x_{in}-x_{jn}|^{q})^{\frac{1}{q}}$$`

其中: 

* 权重 `$w_{i}\geqslant 0(i=1, 2, \ldots, p)$` 表示不同属性的重要性, 
  通常 `$\sum_{i=1}^{n}w_{i}=1$`.

