---
title: DoWhy：因果推断框架 
author: wangzf
date: '2023-07-15'
slug: statistics-causal-analysis-dowhy
categories:
  - 数学、统计学
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

- [DoWhy 流程详解](#dowhy-流程详解)
  - [建模](#建模)
  - [识别](#识别)
  - [估计](#估计)
  - [反驳](#反驳)
- [DoWy 应用示例](#dowy-应用示例)
- [参考](#参考)
</p></details><p></p>

因果推断(causal inference)是基于观察数据进行反事实估计，分析 **干预** 与 **结果** 之间的因果关系的一门科学。
虽然在因果推断领域已经有许多的框架与方法，但大部分方法缺乏稳定的实现。
DoWhy 是微软发布的一个用于进行端到端因果推断的 Python 库，其特点在于：

* 提供了一种原则性的方法将给定的问题转化为一张因果图，保证所有假设的明确性
* 提供了一种面向多种常用因果推断方法的统一接口，并结合了两种主要的因果推断框架
* 自动化测试假设的正确性及估计的鲁棒性

如上所述，DoWhy 基于因果推断的两大框架构建：**图模型** 与 **潜在结果模型**。
具体来说，其使用基于图的准则与 do-积分来对假设进行建模并识别出非参数化的因果效应；
而在估计阶段则主要基于潜在结果框架中的方法进行估计。DoWhy 的整个因果推断过程可以划分为四大步骤：

1. 识别(identify)
    - 在假设（模型）下识别因果效应的表达式（因果估计量）
2. 建模(model)
    - 利用假设（先验知识）对因果推断问题建模
3. 估计(estimate)
    - 使用统计方法对表达式进行估计
4. 反驳(refute)
    - 使用各种鲁棒性检查来验证估计的正确性

下图总结了 DoWhy 的整体流程：

![img](images/dowhy.png)

下面将分别对这四个步骤及其所涉及的方法进行简要介绍。

# DoWhy 流程详解

## 建模

DoWhy 会为每个问题创建一个因果图模型，以保证因果假设的明确性。
该因果图不需要是完整的，你可以只提供部分图，来表示某些变量的先验知识（即指定其类型），
DoWhy 支持自动将剩余的变量视为潜在的混杂因子。

目前，DoWhy 支持如下形式的因果假设：

* 图（Graph）：提供 gml 或 dot 形式的因果图，具体可以是文件或字符串格式
* 命名变量集合（Named variable sets）：直接提供变量的类型，
  包括混杂因子（common causes/cofounders）、工具变量（instrumental variables）、
  结果修改变量（effect modifiers）、前门变量（front-door variables）等

## 识别

基于构建的因果图，DoWhy 会基于所有可能的方式来识别因果效应。具体来说，
会使用基于图的准则与 do-积分 来找出可以识别因果效应的表达式，支持的识别准则有：

* 后门准则（Back-door criterion）
* 前门准则（Front-door criterion）
* 工具变量（Instrumental Variables）
* 中介-直接或间接结果识别（Mediation-Direct and indirect effect identification）

## 估计

DoWhy 支持一系列基于上述识别准则的估计方法，
此外还提供了非参数置信空间与排列测试来检验得到的估计的统计显著性。
具体支持的估计方法列表如下：

* 基于估计干预分配的方法
    - 基于倾向的分层（Propensity-based Stratification）
    - 倾向得分匹配（Propensity Score Matching）
    - 逆向倾向加权（Inverse Propensity Weighting）
* 基于估计结果模型的方法
    - 线性回归（Linear Regression）
    - 广义线性模型（Generalized Linear Models）
* 基于工具变量等式的方法
    - 二元工具/Wald 估计器（Binary Instrument/Wald Estimator）
    - 两阶段最小二乘法（Two-stage least squares）
    - 非连续回归（Regression discontinuity）
* 基于前门准则和一般中介的方法
    - 两层线性回归（Two-stage linear regression）

此外，DoWhy 还支持调用外部的估计方法，例如 `EconML` 与 `CausalML`。

## 反驳

DoWhy 支持多种反驳方法来验证估计的正确性，具体列表如下：

* 添加随机混杂因子：添加一个随机变量作为混杂因子后估计因果效应是否会改变（期望结果：不会）
* 安慰剂干预：将真实干预变量替换为独立随机变量后因果效应是否会改变（期望结果：因果效应归零）
* 虚拟结果：将真实结果变量替换为独立随机变量后因果效应是否会改变（期望结果：因果效应归零）
* 模拟结果：将数据集替换为基于接近给定数据集数据生成过程的方式模拟生成的数据集后因果效应是否会改变（期望结果：与数据生成过程的效应参数相匹配）
* 添加未观测混杂因子：添加一个额外的与干预和结果相关的混杂因子后因果效应的敏感性（期望结果：不过度敏感）
* 数据子集验证：将给定数据集替换为一个随机子集后因果效应是否会改变（期望结果：不会）
* 自助验证：将给定数据集替换为同一数据集的自助样本后因果效应是否会改变（期望结果：不会）

# DoWy 应用示例



# 参考

* [因果推断框架 DoWhy 入门](https://zhuanlan.zhihu.com/p/321808640)
* [DoWhy GitHub](https://github.com/py-why/dowhy)