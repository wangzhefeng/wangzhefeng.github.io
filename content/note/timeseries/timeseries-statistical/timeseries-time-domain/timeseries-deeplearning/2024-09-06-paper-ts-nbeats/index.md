---
title: 【Paper】N-BEATS
author: 王哲峰
date: '2024-09-06'
slug: paper-ts-nbeats
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

- [论文简介](#论文简介)
- [历史研究和瓶颈](#历史研究和瓶颈)
- [论文贡献](#论文贡献)
- [问题定义](#问题定义)
  - [MAPE](#mape)
- [模型定义](#模型定义)
- [实验结果](#实验结果)
- [总结](#总结)
- [资料](#资料)
</p></details><p></p>

# 论文简介

* 题目：N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. 
* 代码：[https://github.com/ElementAI/N-BEATS](https://github.com/ElementAI/N-BEATS)

# 历史研究和瓶颈

在 M4 竞赛中，前排基本都用机器学习的方法，但赢家们还结合了传统统计模型，
例如选手 Smyl 的方法是：带残差/关注机制的神经空洞 LSTM + 经典 Holt-Winter 统计模型。
受到 M4 竞赛的启发，作者想探究在时间序列预测背景下的纯深度学习模型结构，
同时想让模型具有可解释性，让模型能够抽取可解释性因子后，给出预测结果。

# 论文贡献

N-BEATS 的贡献主要有两点：

* 深度神经结构：没用时序特别组成成分，用单纯的深度学习模型也能超过 M3 和 M4 竞赛中做的好的统计方法。
* 对时序可解释的深度学习模型：N-BEATS 模型能跟传统时序中“seasonality-trend-level” 的方法（序列分解）相近，
  输出的结果具有可解释性。

# 问题定义

先了解下离散时间下单一时序点预测的评估函数：MAPE、sMAPE、MASE 和 OWA。

## MAPE


# 模型定义


# 实验结果


# 总结


# 资料

