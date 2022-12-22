---
title: 模型评价指标
author: 王哲峰
date: '2022-11-22'
slug: model-metrics
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
</style>

<details><summary>目录</summary><p>

- [分类](#分类)
  - [precision \& recall \& F1](#precision--recall--f1)
- [回归](#回归)
  - [普通回归](#普通回归)
  - [时间序列预测](#时间序列预测)
    - [MAPE](#mape)
    - [WMAPE](#wmape)
    - [SMAPE](#smape)
- [排序](#排序)
- [聚类](#聚类)
  - [Rank Index](#rank-index)
  - [Mutual Information based scores](#mutual-information-based-scores)
  - [Homogeneity, completeness and V-measure](#homogeneity-completeness-and-v-measure)
  - [Fowlkes-Mallows scores](#fowlkes-mallows-scores)
  - [Silhouette Coefficient](#silhouette-coefficient)
  - [Calinski-Harabasz Index](#calinski-harabasz-index)
  - [Davies-BouIdin Index](#davies-bouidin-index)
  - [Contingency Matrix](#contingency-matrix)
  - [Pair Confusion Matrix](#pair-confusion-matrix)
- [参考](#参考)
</p></details><p></p>

# 分类

* 准确率(precision)
* 召回率(recall)
* F 值(F-Measure)

## precision & recall & F1

一般来说：

* precision 是检索出来的条目(文档、网页)有多少是准确的(准确的定义为所关心的正好被检索出来)
* recall 就是所有准确的条目有多少被检索出来了







# 回归

## 普通回归


## 时间序列预测

在指标方面，作为一个回归问题，可以使用 MAE，MSE 等方式来计算。
但这类 metric 受到具体预测数值区间范围不同，
展现出来的具体误差值区间也会波动很大。比如预测销量可能是几万到百万，
而预测车流量可能是几十到几百的范围，那么这两者预测问题的 MAE 可能就差距很大，
很难做多个任务间的横向比较。所以实际问题中，
经常会使用对数值量纲不敏感的一些 metric，尤其是 SMAPE 和 WMAPE 这两种

这类误差计算方法在各类不同的问题上都会落在 0~1 的区间范围内，
方便来进行跨序列的横向比较，十分方便。在实际项目中还会经常发现，
很多真实世界的时序预测目标，如销量，客流等，
都会形成一个类似 tweedie 或 poisson 分布的情况。
如果用 WMAPE 作为指标，模型优化目标基本可以等价为 MAE（优化目标为中位数），则整体的预测就会比平均值小（偏保守）。在很多业务问题中，
预测偏少跟预测偏多造成的影响是不同的，所以实际在做优化时，
可能还会考察整体的预测偏差（总量偏大或偏小），
进而使用一些非对称 loss 来进行具体的优化

### MAPE


### WMAPE

`$$WMAPE = \frac{\sum_{t=1}^{n}|A_{t} - F_{t}|}{\sum_{t=1}^{n}|A_{t}|}$$`



### SMAPE




# 排序








# 聚类

## Rank Index

## Mutual Information based scores

## Homogeneity, completeness and V-measure

## Fowlkes-Mallows scores

## Silhouette Coefficient

## Calinski-Harabasz Index

## Davies-BouIdin Index

## Contingency Matrix

## Pair Confusion Matrix



# 参考

* [准确率(Precision)、召回率(Recall)、F值(F-Measure)的简要说明](https://blog.csdn.net/huacha__/article/details/80329707?spm=1001.2014.3001.5502)