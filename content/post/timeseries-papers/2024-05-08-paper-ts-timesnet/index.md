---
title: 【paper】TimesNet
author: 王哲峰
date: '2024-05-08'
slug: paper-ts-timesnet
categories:
  - timeseries
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
- [设计思路](#设计思路)
- [](#)
- [参考](#参考)
</p></details><p></p>

# 论文信息

* 论文名称：TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
* 论文地址：[https://openreview.net/pdf?id=ju_Uqw384Oq](https://openreview.net/pdf?id=ju_Uqw384Oq)
* 论文代码：[https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
* 论文内容：
    1. 时间序列分析的背景概念
    2. 时间序列及时间序列分析任务 
    3. 深度学习模型中的基本模块
    4. 现有的深度学习时间序列模型架构
    5. 开源时间序列库：Time Series Library(TSLib)
    6. 未来的方向

# 背景介绍

不同于自然语言、视频等序列数据，时间序列中单个时刻仅保存了一些标量，
其关键信息更多地被蕴含在时序变化（Temporal Variation）中。
因此，建模时序变化是各类时序分析任务共同的核心问题。

本文即围绕时序变化建模展开，设计提出了时序基础模型 TimesNet，
在长时、短时预测、缺失值填补、异常检测、分类五大任务上实现了全面领先。

# 设计思路

近年来，深度模型被广泛用于时序分析任务中，例如循环神经网络（RNN）、
时序卷积网络（TCN）和变换器网络（Transformer）。
然而，前两类方法主要关注捕捉临近时刻之间的变化，在长期依赖上建模能力不足。
Transformer 虽然在建模长期依赖上具有天然优势，但是由于现实世界的时序变化极其复杂，
仅仅依靠离散时间点之间的注意力（Attention）难以挖掘出可靠的时序依赖。

# 






# 参考

* [ICLR2023 | TimesNet: 时序基础模型，预测、填补、分类等五大任务领先](https://zhuanlan.zhihu.com/p/606575441)
* [Paper](https://openreview.net/pdf?id=ju_Uqw384Oq)
