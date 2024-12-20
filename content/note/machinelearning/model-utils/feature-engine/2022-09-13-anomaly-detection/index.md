---
title: 异常值检测及处理
subtitle: Anomaly Detection
author: wangzf
date: '2022-09-13'
slug: anomaly-detection
categories:
  - feature engine
tags:
  - machinelearning
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

- [异常值的定义](#异常值的定义)
- [异常值出现的原因](#异常值出现的原因)
- [异常值检测方法](#异常值检测方法)
- [异常值处理方法](#异常值处理方法)
- [Sigmoid 参数曲线](#sigmoid-参数曲线)
- [参考](#参考)
</p></details><p></p>

# 异常值的定义

在一个特征的观测值中，明显不同于其他数据或不合乎常理的观测值。

# 异常值出现的原因

* 人为错误
    - 数据输入、记录导致的错误
* 自然错误
    - 测量误差，比如仪器出现故障

# 异常值检测方法

* 简单可视化分析
    - 对特征值进行一个数据可视化，远远偏离大部分样本观测值的样本点认为是异常值。
* `$3 \sigma$` 原则
    - 当数据服从正态分布，根据正态分布的定义可知，
      一个观测值出现在距离平均值 `$3 \sigma$` 之外的概率是 `$P(|x-\mu| > 3\sigma) \leq 0.003$`，
      这属于极小概率事件，因此，当观测值距离平均值大于 `$3\sigma$`，则认为该观测值是异常值。
* 箱型图分析(数字异常值，Numeric Outlier)
    - 落在 `$(Q1 - 1.5 \times IQR)$` 和 `$(Q3 + 1.5 \times IQR)$` 之外的观测值认为是异常值。
* Z-score
   - 假设特征服从正态分布，异常值是正态分布尾部的观测值点，因此远离特征的平均值。
     距离的远近取决于特征归一化之后设定的阈值 `$Z_{thr}$`，对于特征中的观测值 `$x_i$`，
     如果 `$Z_i = \frac{x_i - \mu}{\sigma} > Z_{thr}$`，则认为 `$x_i$` 为异常值，
     `$Z_{thr}$` 一般设为：2.5，3.0，3.5。

# 异常值处理方法

* 直接删除含有缺失值的样本
   - 优点: 简单粗暴
   - 缺点: 造成样本量(信息)减少
* 将异常值当做缺失值，交给缺失值处理方法来处理
* 用特征的均值修正；

# Sigmoid 参数曲线

Sigmoid 参数曲线：

`$$f(x) = \frac{a}{b + e^{-(dx - c)}}$$`

损失函数：

`$$S(p) = \sum_{i=1}^{m}(y_{i} - f(x_{i}, p))$$`

# 参考

* [风电异常数据识别与清洗竞赛冠军方案分享](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490892&idx=1&sn=bdd9aea219596e172636cccee4c6dc84&chksm=c32904c3f45e8dd56024943db0e3dc49efcfe21b493d8ab14b8085df7c490d751a0af5b2e618&scene=21#wechat_redirect)
