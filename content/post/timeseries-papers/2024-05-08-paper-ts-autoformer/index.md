---
title: 【Paper】 Autoformer
author: 王哲峰
date: '2024-05-08'
slug: paper-ts-autoformer
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
- [参考](#参考)
</p></details><p></p>

# 论文信息

* 论文名称：Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
* 论文地址：[https://arxiv.org/abs/2106.13008](https://arxiv.org/abs/2106.13008)
* 论文代码：[https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

# 背景介绍

本文探索了**长期时间序列预测**问题：待预测的序列长度远远大于输入长度，即基于有限的信息预测更长远的未来。
上述需求使得此预测问题极具挑战性，对于模型的预测能力及计算效率有着很强的要求。




# 设计思路



# 参考

* [Autoformer:基于深度分解架构和自相关机制的长期序列预测模型](https://zhuanlan.zhihu.com/p/385066440)
* [Paper](https://arxiv.org/abs/2106.13008)
