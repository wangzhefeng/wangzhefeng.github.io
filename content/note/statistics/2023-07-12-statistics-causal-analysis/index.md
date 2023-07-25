---
title: 统计因果分析
author: 王哲峰
date: '2023-07-12'
slug: statistics-causal-analysis
categories:
  - 数学、统计学
tags:
  - article
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

- [因果分析基础](#因果分析基础)
  - [因果分析的概念](#因果分析的概念)
  - [因果分析的要素](#因果分析的要素)
  - [因果分析的分类](#因果分析的分类)
  - [因果分析的方法](#因果分析的方法)
- [因果分析原理](#因果分析原理)
  - [AB test](#ab-test)
  - [鱼骨图分析](#鱼骨图分析)
- [因果分析应用](#因果分析应用)
- [因果分析总结](#因果分析总结)
  - [因果分析的本质](#因果分析的本质)
  - [因果关系和相关关系](#因果关系和相关关系)
  - [因果分析的发展趋势](#因果分析的发展趋势)
  - [总结](#总结)
- [参考](#参考)
</p></details><p></p>

因果分析(Causal Analysis)是分析变量彼此之间的因果关系。
因果推断(Casual Inference)是基于原因推结果，是因果分析的一部分。

因果分析是数据分析、数据科学中重要的方法，广泛应用于 A/B Test、异常分析、用户增长等领域。

# 因果分析基础

## 因果分析的概念

1. 因果
    - 原因和结果
2. 因果关系
    - 原因和结果的关系
3. 因果分析
    - 分析彼此之间的因果关系

## 因果分析的要素

> 因果分析的三要素是：原因、结果、关系

原因可能有多个，导致的结果也可能是多个：

![img](images/casual.png)

这里把问题聚焦，仅探讨抽象的因果关系，所以因果分析可以抽象如下：

* 原因 `$\Longrightarrow$` 结果
* 原因 `$\Longleftarrow$` 结果
* 原因 `$\Longleftrightarrow$` 结果

## 因果分析的分类

按照因果分析的三要素，把因果分析分为三类：

1. 第一类：由原因推结果，又称为因果推断(Causal Inference)
2. 第二类：由结果找原因
3. 第三类：原因和结果互推

## 因果分析的方法

| 类型                         | 方法                    |
|------------------------------|------------------------|
| 原因 `$\Longrightarrow$` 结果 | 1.随机实验(A/B 实验)    |
|                              | 2.倾向评分匹配(PSM)     |
|                              | 3.断点归因(断点回归、RDD) |
|                              | 4.双重差分(DID)         |
|                              | 5.Uplift               |
|                              | ...                    |
| 原因 `$\Longleftarrow$` 结果  | 1.鱼骨图分析            |
|                              | 2.五个“为什么”分析       |
|                              | ...                     |
| 原因 `$\Longleftrightarrow$` 结果 | 公式推导            |

# 因果分析原理

互联网领域，A/B Test、鱼骨图分析法是常用的因果分析方法，所以下面阐述这两种方法的原理

* A/B Test（原因 `$\Longrightarrow$` 结果）
  - 适用于验证单因素的因果关系
* 鱼骨图分析（原因 `$\Longleftarrow$` 结果）
  - 适用于头脑风暴，寻找多个可能的原因

## AB test

A/B Test 是一种随机对照实验，用于实验验证因果关系。A/B Test，是因果实验的代表，
是因果归因、数据归因的主要手段。

A/B 实验是一种单因素归因，适用于验证单因素的因果关系。

具体可参考：

* [A/B Test 的概述、原理、公式推导、Python实现和应用](https://zhuanlan.zhihu.com/p/346602966)

## 鱼骨图分析

鱼骨图分析(Cause and Effect Analysis Chart，也称因果分析法)是典型的由结果找原因的方法。

鱼骨图分析法是对一个问题，分类别、穷举性地列出所有影响因素，进行进一步分析。
其中鱼头是结果（问题），大鱼骨是原因的类别，小鱼骨是具体原因。

鱼骨图分析适用于头脑风暴，寻找多个可能的原因。

# 因果分析应用

因果分析的典型场景是：

| 类型                         | 方法                     |
|------------------------------|-------------------------|
| 原因 `$\Longrightarrow$` 结果 | 1.随机实验(A/B 实验)：策略调整     |
|                              | 4.双重差分(DID)          |
|                              | ...                     |
| 原因 `$\Longleftarrow$` 结果  | 1.[异常分析](https://zhuanlan.zhihu.com/p/418371189)               |
|                              | 2.流失分析               |
|                              | ...                      |

因果分析、推断工具：

![img](images/tools.png)

1. DoWhy
    - 开发者：微软
    - 简介：基于因果推理的统一语言，结合了因果图模型和潜在结果框架，支持因果假设的显式建模和测试
    - 官网：[GitHub](https://github.com/py-why/dowhy)
2. CDT，CausalDiscoveryToolbox
    - 开发者：Goudet Olivier
    - 简介：基于神经网络 CGNN 的图形化（networkx）因果推断。基于 NumPy，sklearn，PyTorch，bnlearn, 和 pcalg
    - [GitHub](https://github.com/GoudetOlivier/CausalDiscoveryToolbox)
3. causalml
    - 开发者：uber
    - 简介：基于机器学习的因果推断的综合包。主要功能包括：
        - 基于实验数据，计算 CATE 和 ITE
        - 基于决策树的算法
        - Meta-learner algorithms
        - 工具变量算法
        - 基于神经网络的算法
    - [GitHub](https://github.com/uber/causalml)
4. EconML
    - 开发者：微软
    - 简介：机器学习因果推断综合包，对标 causalml，安装相对容易一些。主要功能：
        - 主要用于计算 HTE 异质性处理效应
        - 提供模型的解释，以及系数的CI置信区间
        - 基于观测数据，进行因果推断
    - [GitHub](https://github.com/py-why/EconML)

# 因果分析总结

## 因果分析的本质

因果分析的本质就是论证因果关系的充分性、必要性

* 充分条件：原因 `$\Longrightarrow$` 结果
* 必要条件：原因 `$\Longleftarrow$` 结果

## 因果关系和相关关系

因果关系大多数情况下有相关关系，但相关关系不一定是因果关系

## 因果分析的发展趋势

* 原因 `$\Longrightarrow$` 结果
    - 元分析(多个随机试验的整合)
    - 因果推断(因果推断机器学习)
* 原因 `$\Longleftarrow$` 结果
    - 破界创新(逻辑不适合，奇点下移，找到新的底层假设)

## 总结

《精益数据分析》作者认为："发现相关性可以帮助你预测未来，而发现因果关系意味着你可以改变未来"，
所以重视因果关系，重视因果分析吧。

因果分析可以定位问题，挖掘商业价值，洞见机会，在互联网领域有广泛的应用，
比如因果推断、A/B Test、用户增长、异常分析和流失分析等。

但是商业是个复杂生态，因果分析是个很好的分析工具，还要结合具体的用户、业务、数据进行针对性的分析。

# 参考

* [因果分析的原理、方法论和应用](https://zhuanlan.zhihu.com/p/409609129)
* [破界创新：从结果到原因，再从原因到结果](https://zhuanlan.zhihu.com/p/539417288)
* [目前主流的 Python 因果推断包和教材](https://www.zhihu.com/tardis/zm/art/405226148?source_id=1003)
* [A Survey on Causal Inference](https://arxiv.org/pdf/2002.02770.pdf)
* [因果推断：从概念到实践](https://github.com/xieliaing/CausalInferenceIntro)
