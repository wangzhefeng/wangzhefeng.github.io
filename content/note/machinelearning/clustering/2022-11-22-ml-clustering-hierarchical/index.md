---
title: 层次聚类
author: 王哲峰
date: '2022-11-22'
slug: ml-clustering-hierarchical
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

- [层次聚类](#层次聚类)
  - [计算两个样本点之间的距离](#计算两个样本点之间的距离)
  - [计算两个聚类簇之间的距离](#计算两个聚类簇之间的距离)
    - [全链接](#全链接)
    - [单链接](#单链接)
    - [平均链接](#平均链接)
      - [中心链接](#中心链接)
      - [Ward](#ward)
    - [Minimum energy clustering](#minimum-energy-clustering)
- [Agglomerative 聚类](#agglomerative-聚类)
  - [算法介绍](#算法介绍)
  - [算法伪代码](#算法伪代码)
  - [算法实现](#算法实现)
  - [聚类树状图](#聚类树状图)
  - [增加链接约束](#增加链接约束)
- [Bisecting K-Means](#bisecting-k-means)
  - [算法介绍](#算法介绍-1)
  - [算法实现](#算法实现-1)
</p></details><p></p>

# 层次聚类

层次聚类(Hierarchical)是一个通用的聚类算法家族，它通过连续合并或拆分嵌套聚类来构建嵌套聚类。
这种集群层次结构表示为树（或树状图）。树的根是收集所有样本的唯一簇，叶子是只有一个样本的簇

层次聚类算法的关键是如何计算两个样本点之间以及两个聚类簇之间的距离

## 计算两个样本点之间的距离

> 距离矩阵(Metric)

| names           | Eng.                       | formula                                      |
|-----------------|----------------------------| ---------------------------------------------|
| 欧氏距离         | Euclidean distance         | `$\|x-y\|_2 = \sqrt{\sum_i (x_i-y_i)^2}$`    |
| 平方欧氏距离      | Squared Euclidean distance | `$\|x-y\|_2^2 = \sum_i (x_i-y_i)^2$`         |
| 曼哈顿距离	    | Manhattan distance         | `$\|x-y\|_{1} = \sum_{1} \|x_{1} - y_{1}\|$` |  
| 马氏距离(统计距离) | Mahalanobis distance       | `$\sqrt{(x-y)^{T} \Sigma^{-1}(x-y)}$`        |
| 极大值距离	    | Maximum distance           | `$\|x-y\|_{\infty} = {max}_{i} \|x_i-y_i\|$` |
| Cosine 距离      | | |

## 计算两个聚类簇之间的距离

> 链接准则(Linkage criteria)

### 全链接

Maximum/Complete linkage clustering(最大链接/全链接) 最小化两个聚类簇样本点之间的最大距离

计算两个聚类簇样本点之间的最大距离：

`$$d_{max}(C_{i}, C_{j})=\underset{x\in C_{i},y\in C_{j}}{max}dist(x, y)$$`

### 单链接

Single-linkage clustering(单链接) 最小化两个聚类簇样本点之间的最小距离

计算两个聚类簇样本点之间的最小距离：

`$$d_{min}(C_{i}, C_{j})=\underset{x\in C_{i},y\in C_{j}}{min}dist(x, y)$$`

### 平均链接

Mean/Average-linkage clustering(平均链接 UPGMA) 最小化两个聚类簇样本点的所有距离的平均值

计算两个聚类簇样本点之间距离的均值：

`$$d_{avg}(C_{i}, C_{j})=\frac{1}{|C_{i}||C_{j}|}\sum_{x\in C_{i}}\sum_{y\in C_{j}}dist(x, y)$$`

#### 中心链接

Centroid-linkage clustering(中心链接 UPGMC)

* Find the centroid of each cluster and calculate the distance between centroids of two clusters.

`$$d_{cen}(C_{i}, C_{j})=dist(x, y),\quad x,y 分别是 C_{i},C_{j} 的中心$$`

#### Ward

Ward 最小化所有聚类簇内的平方差之和，是一种方差最小化的方法，
类似于 K-Means 目标函数，但是用凝聚层次的方法处理

### Minimum energy clustering

`$$d_{ene}(C_{i}, C_{j})=\frac{2}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m}\|x_{i}- y_{j}\|_{2} - \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\|x_{i}-x_{j}\|_{2}-\frac{1}{m^2}\sum_{i=1}^{m}\sum_{j=1}^{m}\|y_{i}-y_{j}\|_{2}$$`


# Agglomerative 聚类

## 算法介绍

AGNES(Agglomerative Nesting) 是一种采用自底向上聚合策略的层次聚类算法。
每个观察从其自己的集群开始，然后集群依次合并在一起。
链接准则(Linkage criteria)决定了用于合并策略的指标

算法的步骤为: 

1. 先将数据集中的每个样本当做是一个初始聚类簇
2. 然后在算法运行的每一步中找出距离最近的两个点(聚类簇)进行合并为一个聚类簇
3. 上述过程不断重复, 直至所有的样本点合并为一个聚类簇或达到预设的聚类簇个数. 
   最终算法会产生一棵树, 称为树状图(dendrogram), 树状图展示了数据点是如何合并的

当 Agglomerative Clustering 与连接矩阵(connectivity matrix)一起使用时也可以扩展到大量样本，
但是当样本之间没有添加连接约束时计算量很大：它在每一步都考虑所有可能的合并

## 算法伪代码

> **输入:**
> 
> * 样本集: `$D=\{x_{1}, x_{2}, \ldots, x_{n}\}$`
> * 聚类簇距离度量函数 `$d$`
> * 聚类簇数 `$k$`
> 
> **过程:**
> 
> 1. **for** `$j=1, 2, \ldots, n$` **do**
> 
>    `$C_{j}=\{x_{j}\}$`
> 
>    **end for**
> 
> 2. **for** `$i=1, 2, \ldots, n, j=1, 2, \ldots, n$` **do**
> 
>    `$M(i,j)=d(C_{i}, C_{j})$`
> 
>    `$M(i,j)=M(j,i)$`
> 
>    **end for**
> 
> 3. 设置当前聚类簇个数: `$q=n$`
> 4. **while** `$q>k$` **do**
> 
>    * 找出距离最近的两个聚类簇 `$C_{i^{*}}$` 和 `$C_{j^{*}}$`;
> 
>    * 合并 `$C_{i^{*}}$` 和 `$C_{j^{*}}$` : `$C_{i^{*}}=C_{i^{*}} \cup C_{j^{*}}$`
> 
>    * **for** `$j=j^{*}+1,j^{*}+2, \ldots, q$` **do**
>       - 将聚类簇 `$C_{j}$` 重编号为 `$C_{j-1}$`
>    * **end for**
> 
>    * 删除距离矩阵 `$M$` 的 第 `$j^{*}$` 行与第 `$j^{*}$` 列
> 
>    * **for** `$j=1, 2, \ldots, q-1$` **do**
>        - `$M(i^{*},j)=d(C_{i^{*}}, C_{j})$`
>        - `$M(j,i^{*})=M(i^{*},j)$`
>    * **end for**
> 
>    * `$q=q-1$`
> 
>    **end while**
> 
> **输出:**
> 
> * 簇划分 `$C=\{C_{1}, C_{2}, \ldots, C_{k}\}$`

## 算法实现



## 聚类树状图

## 增加链接约束


# Bisecting K-Means

Bisecting K-Means 是一种分裂层次聚类算法(divisive hierarchical clustering)

## 算法介绍

Bisecting KMeans 是 KMeans 迭代变体，使用分裂层次聚类。
Bisecting KMeans 不是一次创建所有质心，而是根据先前的聚类逐步选择质心：
一个簇被重复分成两个新的簇，直到达到目标簇数

常规 K-Means 算法倾向于创建不相关的集群，
但 Bisecting K-Means 的集群有序且创建了一个非常明显的层次结构

* 当聚类簇的个数比较大时，Bisecting KMeans 比 KMeans 更有效，
  因为，Bisecting KMeans 在每次样本分割时是处理一个样本子集，
  而 KMeans 总是处理所有样本数据
* 当聚类簇的个数比较小的时候(相比于数据样本数量)，
  Bisecting KMeans 比 Agglomerative 聚类更有效

Bisecting K-Mean 有两种可选的分裂策略：

* largest cluster: 选择聚类簇拥有最多的样本点
* biggest inertia: 选择聚类簇有最大的评估准则，平方误差和最大

## 算法实现








