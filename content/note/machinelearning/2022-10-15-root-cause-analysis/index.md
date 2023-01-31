---
title: 根因分析 RCA
author: 王哲峰
date: '2022-10-15'
slug: root-cause-analysis
categories:
  - data analysis
tags:
  - algorithm
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

- [KPI 分类](#kpi-分类)
    - [率值 KPI 的计算方法](#率值-kpi-的计算方法)
- [算法](#算法)
  - [Adtributor](#adtributor)
    - [数据](#数据)
    - [输出](#输出)
    - [步骤](#步骤)
    - [算法](#算法-1)
    - [示例](#示例)
    - [局限](#局限)
  - [iDice](#idice)
  - [HotSpot](#hotspot)
  - [Squeeze](#squeeze)
- [参考](#参考)
</p></details><p></p>


通过监控系统运行状态(状态监控)，运维人员需要分析其中的故障点(故障检测)，
并通过回溯排查问题的源头(根因分析)，进而对系统进行管理(控制策略及控制信号生成)，
以使系统运行恢复正常或保持平稳

异常检测能够自动发现问题，而根因定位能够在发现问题的基础上进一步深入定位问题发生的具体维度。
结合数据指标体系、异常检测及根因定位，能够帮助通过监控数据指标的变化，
及时发现业务异动并精准定位问题，在逻辑较为复杂的业务中，将会极大的节省人力，同时提升分析的精准度

KPI 指标(如网页访问量，交易量，失败量，响应时间等)与多维属性(如设备类型、交易渠道、用户画像等)是互联网、
金融等行业常见而重要的业务监测指标。当一个 KPI 的总体值发生异常时，想要解除异常，
定位出其根因所在的位置是关键一步。然而，这一步常常是充满挑战的，尤其当根因是多个维度属性值的组合时

# KPI 分类

从统计分析的角度来看， KPI 可以分为两类：

* 量值 KPI，具有可加性(additive)，如成功数、访问总量
* 率值 KPI，量值 KPI 推导得到，亦称为推导性(derived, ratio)KPI，如点击率、成功率

举例说明：例如用户位置维度下，北京、上海和广州的访问成功量分别为 70、90 和 100，
访问总量均为 100，成功率分别为 70%、90%、100%；三者的访问成功量相加得到用户位置维度的访问成功量为 260，
但三者的成功率不能直接相加得到用户位置维度的成功率为 260%。
因此对于量值 KPI 和率值 KPI 需要采用不同的方法进行根因分析

### 率值 KPI 的计算方法

1. EP 值

率值 KPI 的计算方式不同于量值 KPI，因为率值 KPI 不能像量值 KPI一样在不同维度和元素上进行加减操作，
率值 KPI 的波动变化也不能通过加减反映在各个维度和元素上。
此时，偏导数和有限差分便派上用场了。衡量对于这种形式函数的波动变化情况，计算公式如下：

`$$EP_{ij} = A_{ij}(m_{1}) - F_{ij}(m_{1}) \times F(m_{2}) \\ - \frac{A_{ij}(m_{2}) - F_{ij}(m_{2}) \times F(m_{1})}{F(m_{2})}   \times (F(m_{2}) + A_{ij}(m_{2}) - F_{ij}(m2))$$`


2. S 值

根据相对熵理论，对于需要首先计算 `$f(.)$` 和 `$g(.)$` 函数的联合概率分布，
然后计算联合概率分布函数的相对熵，计算十分复杂。
论文作出近似假设，认为 `$f$` 和 `$g$` 函数之间相互独立，
则的联合概率分布相对熵就是 `$f(.)$` 和 `$g(.)$` 的概率分布函数相对熵之和。
因此，率值 KPI 的 `$S$` 值等于组成率值 KPI 定义公式的分子 KPI 和分母 KPI 的 `$S$` 值之和。
省略下标 `$ij$`，`$S$` 值计算公式如下：

`$$S=\sum(Sf, Sg)$$`


# 算法

## Adtributor

2014年，微软研究院提出了一种基于 Adtributor 算法的多维时间序列异常根因分析方法。
同时，在 AIOps 技术研讨交流会暨 2019 国际 AIOps 挑战赛中，
获奖的前五支团队无一例外地引用了该算法

经过长期的案例学习发现，一个异常的根因由多个维度共同导致的情况非常罕见。
因此本文将问题限制在只在一个维度的元素组合中寻找异常的根因

### 数据

多维时间序列数据，包含：时间戳TimeStamp、维度D、元素E、指标KPI。数据表结构如下：

![img](images/adtributor_input.png)

### 输出

* 对于每一个维度，元素集合要能够尽可能地解释 KPI 异常波动
* 对于每一个维度，元素集合符合奥卡姆剃刀原则、形式上尽可能简洁
* 在所有维度中，找出最意外的、真实情况与期望值相差最大的维度和元素

### 步骤

Adtributor 多维根因分析流程如图所示，主要包括四个步骤：

![img](images/adtributor_flow.png)

1. 数据收集：收集 KPI 的多维时间序列数据，对于缺失值、死值等进行初步预处理，提升数据质量
2. 异常检测：采用 ARMA 时间序列模型或 Isolation Forest 对 KPI 进行实时预测，
   将预测值 F 和真实值 A 对比，判断 KPI 是否发生异常；
   预测值 F 和真实值 A 将用于 Adtributor 根因分析
3. 根因分析：Adtributor 对异常 KPI 的所有维度和元素计算 EP 值和 S 值，
   并与 TEP、TEEP 阈值比对分析，从而筛选和定位出异常根因

### 算法

![img](images/adtributor_algorithm.png)

### 示例

假设目前有这样一个数据集:

![img](images/adtributor_example.png)

假设在某个时刻，广告收益的预测值为 100，实际只有 50，触发异常告警，并且执行根因分析。
这里假设和收益相关只有三个维度分别是数据中心(DC)、广告商(AD)、设备类型(DT)。
上图是该时刻三个维度的实际值和预测值的情况。算法需要定位是哪些维度的哪些元素导致了这次收益下降

1. 计算某维度下某因素的真实值和预测值占总收入的差异性(Surprise)

1.1 计算 `$p_{ij}$`，`$q_{ij}$`

其中: 

* `$p_{ij}$` 表示 i 维度下 j 因素对应的预测收入占总预测收入的比率，具体公式为

`$$p_{ij} = \frac{F_{ij}(m)}{F(m)}$$`

`$$p_{ij} = \frac{F_{ij}(m)}{F(m)}$$`



其中:
* m 是某种度量指标如广告收入，i 表示维度，j 表维度下具体因素。
F(m)表示预测总收入如 100 万， 表示维度 i 下 j 因素对应的预测收入，
如数据中心 X 对的预测收入为 94 万，则对应的  ； 
A(m) 表示实际总收入50万，则数据中心X的实际收入为47万，则对应的 

B: 计算 的差异度，论文中利用js散度进行计算， 直接的差异越大，则js散度值越大

JS 散度： 

2. 计算某维度下某因素波动占总体波动的比率 (Explanatory power)

其中波动表示真实值和异常值的差异性，具体公式入下：


如终端设备 PC 的 EP 值为：



Tablet 的 EP 值为：


3. 排序，得出归因结果

通过计算维度的惊奇性(维度内所有元素惊奇性之和)对维度进行排序，
确定根因所在的维度(例如省份)。在维度内部计算每个元素的解释力，
当元素的解释力之和超过阈值时，这些元素就被认为是根因



### 局限

1. 上述是针对基本类型的 KPI 的计算公式(例如 PV、交易量)，
   对于派生类型的 KPI(多个基本类型 KPI 计算得到，例如成功率)就不太适用了
2. 将根因限定在一维的假设不太符合我们的实际场景，同时用解释性和惊奇性的大小来衡量根因也不完全合理。
   因为其没有考虑到维度之间的相互影响以及外部根因的可能
3. Adtributor 的根因分析严重依赖于整体 KPI 的变化情况，
   对于整体变化不大，但是内部波动较为剧烈的数据表现不好

## iDice

Problem Identification for Emerging Issues

## HotSpot

Anomaly Localization for Additive KPIs with Multi-Dimensional Attributes

## Squeeze

Generic and Robust Localization of Multi-Dimensional Root Causes


必示业务明细多维定位算法


# 参考

* [根因分析思路方法总结](https://segmentfault.com/a/1190000041824375)
* [Adtributor](Revenue Debugging in Advertising Systems(PAPER).pdf)
* [HotSpot](Anomaly-Localization-for-Additive-KPIs-with-Multi-Dimensional-Attributes.pdf)
* [iDice](Problem-Identification-for-Emerging-Issues.pdf)
* [Apriori](Detecting and Localizing End-to-End Performance.pdf)
* [Recursive](Adtributor.pdf)
* [Real Time Root Cause Analysis in Open Distro for Elasticsearch]()