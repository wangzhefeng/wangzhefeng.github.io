---
title: 时间序列聚类
author: 王哲峰
date: '2022-10-30'
slug: timeseries-clustering
categories:
  - timeseries
tags:
  - ml
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

- [聚类](#聚类)
- [时间序列聚类算法](#时间序列聚类算法)
  - [基于距离的机器学习聚类算法](#基于距离的机器学习聚类算法)
  - [基于相似性的机器学习聚类算法](#基于相似性的机器学习聚类算法)
</p></details><p></p>


# 聚类

聚类分析(cluster analysis)简称聚类(clustering)，它是数据挖掘领域最重要的研究分支之一，
也是最为常见和最有潜力的发展方向之一。聚类分析是根据事物自身的特性对被聚类对象进行类别划分的统计分析方法，
其目的是根据某种相似度度量对数据集进行划分，将没有类别的数据样本划分成若干个不同的子集，
这样的一个子集称为簇（cluster)，聚类使得同一个簇中的数据对象彼此相似，
不同簇中的数据对象彼此不同，即通常所说的“物以类聚”

时间序列的聚类在工业生产生活中非常常见，大到工业运维中面对海量KPI曲线的隐含关联关系的挖掘，
小到股票收益曲线中的增长模式归类，都要用到时序聚类的方法帮助我们发现数据样本中一些隐含的、深层的信息

# 时间序列聚类算法

根据对时间序列的距离度量方法，可以采用机器学习中很多聚类算法，进行时间序列的聚类分析

## 基于距离的机器学习聚类算法



## 基于相似性的机器学习聚类算法

