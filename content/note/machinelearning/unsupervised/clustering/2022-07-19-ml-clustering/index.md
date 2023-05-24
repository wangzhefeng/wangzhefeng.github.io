---
title: 聚类算法概览
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [聚类算法](#聚类算法)
  - [基于原型的聚类](#基于原型的聚类)
  - [基于密度的聚类](#基于密度的聚类)
  - [层次聚类算法](#层次聚类算法)
  - [基于模型的聚类](#基于模型的聚类)
  - [其他聚类算法](#其他聚类算法)
- [聚类数据](#聚类数据)
- [聚类性能度量](#聚类性能度量)
- [聚类距离计算](#聚类距离计算)
  - [闵可夫斯基距离](#闵可夫斯基距离)
  - [VDM](#vdm)
  - [闵可夫斯基距离与 VDM 混合距离](#闵可夫斯基距离与-vdm-混合距离)
  - [加权闵可夫斯基距离](#加权闵可夫斯基距离)
- [聚类算法库](#聚类算法库)
</p></details><p></p>

# 聚类算法

聚类是从数据集中挖掘相似观测值集合的方法。聚类试图将数据集中的样本划分为若干个通常是不相交的子集，
每个子集称为一个"簇"(cluster)。通过这样的划分，每个簇可能对应于一些潜在的概念(类别)。
聚类过程仅能自动形成簇结构，簇所对应的概念语义需由使用者自己来把握。
聚类既能作为一个单独的过程用于寻找数据内在的分布结构，也可以作为分类等其他学习任务的前驱过程 

角度 I:

* 基于原型、质心、中心的聚类 (Prototype-based Clustering)
    - K-Means 聚类 (K-Means)
    - 学习向量量化聚类 (Learning Vector Quantization)
* 基于分布的聚类
    - 高斯混合模型聚类 (Gaussian Mixture Model)
* 基于密度的聚类 (Density-based Clustering)
    - DBSCAN (Density-Based Spatial Clustering of Application with Noise)
    - OPTICS (Ordering Points To Identify the Clustering Structure)
* 基于连通性的聚类、层次聚类 (Hierarchical Clustering)
* 基于模型的聚类 (Model-based Clustering)
    - 混合回归模型 (Mixture Regression Model)
* 基于图论的聚类
    - 亲和力传播 (Affinity propagation)
* 其他聚类方法
    - 谱聚类
    - Chameleon
    - Canopy
    - ...

聚类算法比较：

| 方法 | 参数 | 可扩展性 | 用例 | 几何 |
|---------------------|----|----|----|----|
| K-Means                  | | | | |
| Affinity propagation      | | | | |
| Mean-shift               | | | | |
| Spectral clustering      | | | | |
| Ward hierarchical        | | | | |
| Agglomerative clustering | | | | |
| DBSCAN                   | | | | |
| OPTICS                   | | | | |
| Gaussian                 | | | | |
| BIRCH                    | | | | |
| Bisecting K-Means        | | | | |

## 基于原型的聚类

基于原型的聚类(Prototype-based Clustering) 假设聚类结构能通过一组原形刻画。
通常情况下，算法先对原型进行初始化，然后对原型进行迭代更新求解，
采用不同的原型表示，不同的求解方式，将产生不同的算法

常见的基于原型的聚类有：

* [K 均值聚类(K-Means)]()
* [学习向量量化聚类(Learning Vector Quantization)]()
* [高斯混合聚类(Mixture-of-Gaussian)]()

## 基于密度的聚类

基于密度的聚类(Density-based Clustering) 假设聚类结构能通过样本分布的紧密程度确定。
密度聚类算法从样本密度的角度来考察样本之间的可连续性，并基于可连续样本不断扩展聚类簇以获得最终的聚类结果

常见的基于密度的聚类有：

* [DBSCAN(Density-Based Spatial Clustering of Application with Noise)](https://en.wikipedia.org/wiki/DBSCAN)
* [OPTICS(Ordering Points To Identify the Clustering Structure)](https://en.wikipedia.org/wiki/OPTICS_algorithm)

## 层次聚类算法

层次聚类(Hierarchical Clustering) 也称为基于连通性的聚类。
这种算法试图在不同层次对数据进行划分，从而形成树形的聚类结构

数据集的划分采用不同的策略会生成不同的层次聚类算法：

* "自底向上" 的聚合策略
    - [AGNES(Agglomerative Nesting)]()
* "自顶向下" 的分拆策略

## 基于模型的聚类

* 混合回归模型

## 其他聚类算法

- 谱聚类
- Chameleon
- Canopy
- ...

# 聚类数据

假定样本集 `$D$` 包含 `$n$` 个无标记样本:

`$$D = \{x_1, x_2, \ldots, x_n\}$$`

每个样本是一个 `$p$` 维特征向量:

`$$x_i=(x_{i1}; x_{i2}; \ldots; x_{ip}), i=1,2,\ldots, n$$`

聚类算法将样本集 `$D$` 划分为 `$k$` 个不相交的簇:

`$$\{C_l|l=1, 2, \ldots, k\}$$`

其中: 

* `$C_{l^{'}} \underset{l^{'} \neq l}{\cap} C_{l} = \emptyset$`
* `$D=\cup_{l=1}^{k}C_{l}$`

相应的, 用

`$$\lambda_{i} \in \{1, 2, \ldots, k\}$$`

表示样本 `$x_{i}$` 的"簇标记”(cluster label)，即

`$$x_{i} \in C_{\lambda_{i}}$$`

于是，聚类的结果可用包含 `$n$` 个元素的簇标记向量表示

`$$\lambda=(\lambda_{1}, \lambda_{2}, \ldots, \lambda_{n})$$`

# 聚类性能度量

# 聚类距离计算

距离度量(distance measure)函数 `$dist(,)$` 需满足的基本性质：

* 非负性：`$dist(x_{i}, x_{j}) \geqslant 0$`
* 同一性：`$dist(x_{i}, x_{j})=0$` 当且仅当 `$x_{i}=x_{j}$`
* 对称性：`$dist(x_{i}, x_{j})=dist(x_{j}, x_{i})$`
* 直递性：`$dist(x_{i}, x_{j}) \leqslant dist(x_{i}, x_{k}) + dist(x_{k}, x_{j})$` (可不满足)

变量特征：

* 连续特征
    - 闵可夫斯基距离
* 离散特征
    - 有序特征: 闵可夫斯基距离
    - 无序特征: VDM(Value Difference Metric)
* 混合特征
    - 闵可夫斯基距离与 VDM 混合距离

## 闵可夫斯基距离

> Minkowski distance

样本: 

`$$\begin{cases}
x_{i}=(x_{i1}, x_{i2}, \ldots, x_{ip}),i=1,2,\ldots, n \\
x_{j}=(x_{j1}, x_{j2}, \ldots, x_{jp}),i=1,2,\ldots, n
\end{cases}$$`


* `$q \geqslant 1$`：闵可夫斯基距离(Minkowski distance)

`$$dist_{mk}(x_{i}, x_{j})=||x_{i} - x_{j}||^{q} = \bigg(\sum_{u=1}^{p}|x_{iu}-x_{ju}|^{q}\bigg)^{\frac{1}{q}}$$`

* `$q=1$`：曼哈顿距离(Manhattan distance)

`$$dist_{man}(x_{i}, x_{j})=\|x_{i}-x_{j}\|_{1}=\sum^{p}_{u=1}|x_{iu}-x_{ju}|$$`

* `$q=2$`：欧氏距离(Euclidean distance)

`$$dist_{ed}(x_{i}, x_{j})=\|x_{i}-x_{j}\|_{2}=\sqrt{\sum^{p}_{u=1}|x_{iu}-x_{ju}|^{2}}$$`

## VDM

> VDM，Value Difference Metric，值差异指标

令 `$m_{u,a}$` 表示在特征 `$u$` 上取值为 `$a$` 的样本数，
`$m_{u, a, i}$` 表示在第 `$i$` 个样本簇中在特征 `$u$` 上取值为 `$a$` 的样本数，
`$k$` 为样本簇数，则特征 `$u$` 上两个离散值 `$a$` 与 `$b$` 之间的 VDM 距离为: 

`$$VDM_{q}(a, b)=\sum^{k}_{i=1}\bigg|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\bigg|^{q}$$`

## 闵可夫斯基距离与 VDM 混合距离

假设有 `$p_{c}$` 个有序特征，`$p-p_{c}$` 个无序特征，有序特征排列在无序特征之前：

`$$MinkovDM_{q}(x_{i}, x_{j})=\bigg(\sum^{p_{c}}_{u=1}|x_{i,u}-x_{j,u}|^{q}+\sum^{p}_{u=p_{c}+1}VDM_{q}(x_{i,u},x_{j,u})\bigg)^{\frac{1}{q}}$$`

## 加权闵可夫斯基距离

当样本在空间中不同特征的重要性不同时：

`$$dist_{wmk}(x_{i}, x_{j})=(w_{1}\cdot|x_{i1}-x_{j1}|^{q}+w_{2}\cdot|x_{i2}-x_{j2}|^{q}+\ldots+w_{n}\cdot|x_{in}-x_{jn}|^{q})^{\frac{1}{q}}$$`

其中：

* 权重 `$w_{i}\geqslant 0(i=1, 2, \ldots, p)$` 表示不同特征的重要性，
  通常 `$\sum_{i=1}^{n}w_{i}=1$`

# 聚类算法库

* sklearn
    - sklearn.cluster
    - sklearn.feature_extraction
    - sklearn.metrics.pairwise
