---
title: 统计学知识
author: wangzf
date: '2022-05-07'
slug: statistic-basic
categories:
  - 数学、统计学
tags:
  - note
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

- [概述](#概述)
- [大数定律与中心极限定理](#大数定律与中心极限定理)
  - [大数定律](#大数定律)
    - [切比雪夫(Chebyishev)不等式](#切比雪夫chebyishev不等式)
    - [切比雪夫定理的特殊情况](#切比雪夫定理的特殊情况)
    - [伯努利大数定理](#伯努利大数定理)
    - [辛钦大数定理](#辛钦大数定理)
  - [中心极限定理](#中心极限定理)
    - [同分布的中心极限定理](#同分布的中心极限定理)
    - [德莫佛－拉普拉斯定理](#德莫佛拉普拉斯定理)
- [统计推断理论](#统计推断理论)
- [抽样分布](#抽样分布)
- [参数估计](#参数估计)
  - [点估计](#点估计)
  - [区间估计](#区间估计)
- [偏度、峰度](#偏度峰度)
  - [偏度](#偏度)
  - [峰度](#峰度)
- [回归分析](#回归分析)
  - [回归分析简介](#回归分析简介)
  - [回归分析理论](#回归分析理论)
- [参考](#参考)
</p></details><p></p>

# 概述

整理一下统计学中常用的概念、方法论. 作为一个统计学出身的人, 遇到这些问题时希望不要被难倒

内容大致包含：

* 大数定律、中心极限定理
* 贝叶斯公式、贝叶斯定理
* 参数估计
    - 点估计、区间估计
* 最大似然估计与EM算法
* 假设检验
    - A/B test
* 方差分析
* 回归分析
* 主成分分析
* 因子分析
* 聚类分析
* 统计显著性

# 大数定律与中心极限定理


在统计学中, 大数定律又称大数法则、大数率, 是描述相当多次数重复实验的结果的定律; 
根据这个定律, 样本数量越多, 则其算术平均值就有越高的概率接近期望值. 

## 大数定律

若 `$\xi_1, \xi_2,...,\xi_n,...$` 是随机变量序列, 令

`$$\eta_{n} = \frac{\xi_1+\xi_2+...+\xi_n}{n}$$`

若存在常数序列 `$a_1,a_2,...,a_n,...$` 对任何的正数 `$\epsilon$`, 恒有

`$$\lim\limits_{n \to \infty}P(|\eta_n-a_n|<\epsilon)=1$$`

则称序列 `${\epsilon_n}$` 服从 **大数定律**(或**大数法则**). 

### 切比雪夫(Chebyishev)不等式


### 切比雪夫定理的特殊情况

### 伯努利大数定理

### 辛钦大数定理

## 中心极限定理

对于独立随机变量序列 `$\xi_1, \xi_2,...,\xi_n,...$`, 
假定 `$E(\xi_n)$` 和 `$D(\xi_n)$` 都存在, 令

`$$\zeta_n=\frac{\sum_{i=1}^{n}\xi_i-\sum_{i=1}^{n}E(\xi_i)}{\sqrt{\sum_{i=1}^{n}}}$$`

若

`$$\lim\limits_{n \to \infty}P(\zeta_n < x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{\frac{-t^2}{2}}dt$$`
   
则称序列 `${\xi_n}$` 服从 **中心极限定理(Central Limit Theorem)**. 

### 同分布的中心极限定理

### 德莫佛－拉普拉斯定理

# 统计推断理论

* 抽样分布
* 参数估计
    - 点估计
    - 区间估计
* 假设检验
    - 参数假设检验问题
    - 非参数假设检验问题

# 抽样分布

# 参数估计

## 点估计

## 区间估计


# 偏度、峰度

## 偏度

* 偏度(skewness)又称偏态、偏态系数, 是描述数据分布偏斜方向和程度的度量, 
  其是衡量数据分布非对称程度的数字特征. 对于随机变量 `$X$`, 其偏度是样本的三阶标准化矩:

`$$Skew(x) = E[(\frac{(X-\mu)^{3}}{\sigma})] = \frac{E(X^{3})-3\mu \sigma^{2} - \mu^{3}}{\sigma^{3}}$$`

* 偏度的衡量是相对于正态分布来说, 正态分布的偏度为0. 因此说:
    - 若数据分布是对称的, 偏度为0
    - 若偏度 > 0, 则可认为分布为右偏, 也叫正偏, 即分布有一条长尾在右
    - 若偏度 < 0, 则可认为分布为左偏, 也叫负偏, 即分布有一条长尾在左

## 峰度

峰度(Kurtosis)是描述数据分布陡峭或平滑的统计量, 通过对峰度的计算, 
能够判定数据分布相对于正态分布而言是更陡峭还是平缓. 对于随机变量 `$X$`, 
其峰度为样本的四阶标准中心矩
      
`$$Kurt(x) = E[(\frac{(X-\mu)^{4}}{\sigma})] = \frac{E[(X-\mu)^4]}{(E[[(X-\mu)^2]])^2}$$`

* 当峰度系数 > 0, 从形态上看, 它相比于正态分布要更陡峭或尾部更厚
* 峰度系数 < 0, 从形态上看, 则它相比于正态分布更平缓或尾部更薄
* 在实际环境当中, 如果一个分部是厚尾的, 这个分布往往比正态分布的尾部具有更大的"质量", 即含又更多的极端值
* 常用的几个分布中, 正态分布的峰度为 0, 均匀分布的峰度为 -1.2, 指数分布的峰度为 6

# 回归分析

## 回归分析简介


## 回归分析理论


# 参考

* [概率论与数理统计]()
* [高等数理统计]()
* [维基百科-大数定律](https://zh.wikipedia.org/wiki/%E5%A4%A7%E6%95%B0%E5%AE%9A%E5%BE%8B)

