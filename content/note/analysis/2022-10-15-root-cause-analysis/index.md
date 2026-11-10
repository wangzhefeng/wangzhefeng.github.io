---
title: 根因分析
author: wangzf
date: '2022-10-15'
slug: root-cause-analysis
categories:
  - data-analysis
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [KPI 分类](#kpi-分类)
    - [率值 KPI 的计算方法](#率值-kpi-的计算方法)
- [算法](#算法)
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
并通过回溯排查问题的源头(根因分析，RCA)，进而对系统进行管理(控制策略及控制信号生成)，
以使系统运行恢复正常或保持平稳

异常检测能够自动发现问题，而根因定位能够在发现问题的基础上进一步深入定位问题发生的具体维度。
结合数据指标体系、异常检测及根因定位，能够帮助通过监控数据指标的变化，
及时发现业务异动并精准定位问题，在逻辑较为复杂的业务中，将会极大的节省人力，同时提升分析的精准度

KPI 指标(如网页访问量，交易量，失败量，响应时间等)与多维属性(如设备类型、交易渠道、用户画像等)是互联网、
金融等行业常见而重要的业务监测指标。当一个 KPI 的总体值发生异常时，想要解除异常，
定位出其根因所在的位置是关键一步。然而，这一步常常是充满挑战的，尤其当根因是多个维度属性值的组合时

## KPI 分类

从统计分析的角度来看， KPI 可以分为两类：

* 量值 KPI，具有可加性(additive)，如成功数、访问总量
* 率值 KPI，量值 KPI 推导得到，亦称为推导性(derived, ratio)KPI，如点击率、成功率

举例说明：例如用户位置维度下，北京、上海和广州的访问成功量分别为 70、90 和 100，
访问总量均为 100，成功率分别为 70%、90%、100%；三者的访问成功量相加得到用户位置维度的访问成功量为 260，
但三者的成功率不能直接相加得到用户位置维度的成功率为 260%。
因此对于量值 KPI 和率值 KPI 需要采用不同的方法进行根因分析

#### 率值 KPI 的计算方法

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


## 算法

### iDice

Problem Identification for Emerging Issues

### HotSpot

Anomaly Localization for Additive KPIs with Multi-Dimensional Attributes

### Squeeze

Generic and Robust Localization of Multi-Dimensional Root Causes


必示业务明细多维定位算法


## 参考

* [根因分析思路方法总结](https://segmentfault.com/a/1190000041824375)
* [HotSpot](Anomaly-Localization-for-Additive-KPIs-with-Multi-Dimensional-Attributes.pdf)
* [iDice](Problem-Identification-for-Emerging-Issues.pdf)
* [Apriori](Detecting and Localizing End-to-End Performance.pdf)
* [Recursive](Adtributor.pdf)
* Real Time Root Cause Analysis in Open Distro for Elasticsearch
* [根因分析初探：一种报警聚类算法在业务系统的落地实施](https://tech.meituan.com/2019/02/28/root-clause-analysis.html)
* [AIOps 电信故障根因定位竞赛Top方案总结](https://mp.weixin.qq.com/s/lN1HHHn9QVpaZVr-iGqH9A)
* [异动分析-根因分析定位问题](https://zhuanlan.zhihu.com/p/345569713)
* [一文看懂5大代表性算法的原理和优势](https://blog.csdn.net/weixin_52705010/article/details/115404511)
* [故障诊断与根因分析论文一览](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247485139&idx=1&sn=8310a0a9dcd10bd0e0eb6d87deba8ece&chksm=cecef326f9b97a30205d102114c4e37da827931a1c27c1d3cfcba5bd29e9c1dd86fb78998ada&scene=21#wechat_redirect)
* [日志根因分析](https://github.com/testtctc/log)
