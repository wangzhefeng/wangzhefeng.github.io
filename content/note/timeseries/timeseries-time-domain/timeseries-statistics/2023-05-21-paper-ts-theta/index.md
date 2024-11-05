---
title: Theta
author: wangzf
date: '2023-05-21'
slug: paper-ts-theta
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
- [模型定义](#模型定义)
- [实验结果](#实验结果)
- [总结](#总结)
- [参考](#参考)
</p></details><p></p>

# Theta 介绍

Theta 模型基本上依赖于分解。我们知道时间序列可以分解为三个部分：趋势部分、季节性部分和残差。
因此，将序列分解为各个组成部分，对未来预测每个组成部分，并将每个部分的预测组合成最终预测，是一个合理的做法。
不幸的是，在实践中这并不奏效，尤其是因为很难分离残差并对其进行预测。
因此，Theta 模型是这个想法的演变，它依赖于将序列分解为长期部分和短期部分。
从正式的角度来看，Theta 模型基于修改时间序列的局部曲率的概念。
这种修改由一个称为 theta 的参数管理（因此称为 Theta 模型）。
这种修改应用于序列的二次差分，意味着它被差分两次。当 theta 在 0 和 1 之间时，序列“被减压”。
这意味着短期波动较小，我们强调长期影响。当 theta 达到 0 时，序列转化为一条直线回归线。




# 参考

* [Time Series Forecasting with Theta model](https://www.kaggle.com/code/kkhandekar/time-series-forecasting-with-theta-model)
* [The Theta Model](https://www.statsmodels.org/dev/examples/notebooks/generated/theta-model.html)
* [AutoTheta Model](https://nixtlaverse.nixtla.io/statsforecast/docs/models/autotheta.html#optimised-and-standard-theta-models)