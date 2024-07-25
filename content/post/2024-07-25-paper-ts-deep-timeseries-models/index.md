---
title: Deep Time Series Models 综述
author: 王哲峰
date: '2024-07-25'
slug: paper-ts-deep-timeseries-models
categories:
  - timeseries
  - 论文阅读
tags:
  - paper
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

- [论文信息](#论文信息)
- [背景介绍](#背景介绍)
- [时间序列及时间序列分析任务](#时间序列及时间序列分析任务)
  - [时间序列](#时间序列)
  - [时间序列任务](#时间序列任务)
- [基本模块](#基本模块)
  - [时间序列平稳化](#时间序列平稳化)
  - [时间序列分解](#时间序列分解)
    - [Seasonal-Trend 分解](#seasonal-trend-分解)
    - [Basis Expansion](#basis-expansion)
    - [Matrix Factorization](#matrix-factorization)
  - [傅里叶分析](#傅里叶分析)
- [模型架构](#模型架构)
- [参考](#参考)
</p></details><p></p>

# 论文信息

* 论文名称：Deep Time Series Models: A Comprehensive Survey and Benchmark
* 论文地址：[https://arxiv.org/abs/2407.13278](https://arxiv.org/abs/2407.13278)
* 论文代码：[https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

# 背景介绍

# 时间序列及时间序列分析任务

## 时间序列


## 时间序列任务

# 基本模块

## 时间序列平稳化

传统时间序列平稳化方法：

* 差分(differencing)
* 对数变换(log-transformation)

深度学习方法：

* deep adaptive input normalization(DAIN) layer：根据时间序列的原始分布自适应地平稳化时间序列
* RevIN：
* Non-Stationary Transformer
* SAN

平稳化地一般模式：

对于一个时间序列 `$\mathbf{X} = \{\mathbf{X}_{1}, \mathbf{X}_{2}, \cdots, \mathbf{X}_{T}\} \in \mathbb{R}^{T \times C}$`，`$T$` 为时间戳数量，`$C$` 为变量数。
对上述时间序列进行平稳化的固定框架如下：

`$$\mu_{\mathbf{X}} = \frac{1}{T}\sum_{i=1}^{T}\mathbf{X}_{i}, \sigma_{\mathbf{X}}^{2}=\sum_{i=1}^{T}\frac{1}{T}(\mathbf{X}_{i} - \mu_{\mathbf{X}})^{2}$$`

`$$\mathbf{X}' = \frac{(\mathbf{X} - \mu_{\mathbf{X}})}{\sqrt{\sigma_{\mathbf{X}}^{2} + \epsilon}}$$`

`$$\mathbf{Y}' = Model(\mathbf{X}')$$`

`$$\hat{\mathbf{Y}} = \sigma_{\mathbf{X}}^{2}(\mathbf{Y}' + \mu_{\mathbf{X}})$$`

其中：

* `$\epsilon$` 是一个较小的数字（数值稳定性）
* 

## 时间序列分解

### Seasonal-Trend 分解



### Basis Expansion


### Matrix Factorization



## 傅里叶分析



# 模型架构


# 参考

