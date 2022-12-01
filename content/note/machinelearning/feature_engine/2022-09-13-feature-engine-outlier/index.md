---
title: 异常值处理
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-outlier
categories:
  - feature engine
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

- [异常值出现的原因](#异常值出现的原因)
- [异常值检测方法](#异常值检测方法)
- [特征异常值处理方法](#特征异常值处理方法)
- [极值分析](#极值分析)
</p></details><p></p>

异常值的定义: 在一个特征的观测值中, 明显不同于其他数据或不合乎常理的观测值

# 异常值出现的原因

- 人为错误
   - 数据输入、记录导致的错误
- 自然错误
   - 测量误差, 比如仪器出现故障

# 异常值检测方法

- 简单可视化分析
    - 对特征值进行一个数据可视化, 远远偏离大部分样本观测值的样本点认为是异常值
- `$3 \sigma$` 原则
    - 当数据服从正态分布, 根据正态分布的定义可知, 
      一个观测值出现在距离平均值 `$3 \sigma$` 之外的概率是 `$P(|x-\mu| > 3\sigma)<=0.003$`, 
      这属于极小概率事件, 因此, 当观测值距离平均值大于 `$3\sigma$`, 则认为该观测值是异常值；
- 箱型图分析(数字异常值, Numeric Outlier)
    - 落在 `$(Q1 - 1.5 \* IQR)$` 和 `$(Q3 + 1.5 \* IQR)$` 之外的观测值认为是异常值
- Z-score
   - 假设特征服从正态分布, 异常值是正态分布尾部的观测值点, 因此远离特征的平均值. 
     距离的远近取决于特征归一化之后设定的阈值 `$Z_thr$`, 对于特征中的观测值 `$x_i$`, 
     如果 `$Z_i = \frac{x_i - \mu}{\sigma} > Z_thr$`, 则认为 `$x_i$` 为异常值, 
     `$Z_thr$` 一般设为, 2.5, 3.0, 3.5

# 特征异常值处理方法

- 直接删除含有缺失值的样本
   - 优点: 简单粗暴
   - 缺点: 造成样本量(信息)减少
- 将异常值当做缺失值, 交给缺失值处理方法来处理
   - 优点: 
   - 缺点: 
- 用特征的均值修正；
   - 优点: 
   - 缺点: 

# 极值分析

