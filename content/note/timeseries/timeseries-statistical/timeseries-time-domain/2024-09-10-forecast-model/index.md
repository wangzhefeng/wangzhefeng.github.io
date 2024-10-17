---
title:  时间序列预测模型
author: wangzf
date: '2024-09-10'
slug: forecast-model
categories:
  - timeseries
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

- [预测方法](#预测方法)
  - [均值预测法](#均值预测法)
  - [朴素预测法](#朴素预测法)
  - [移动平均法](#移动平均法)
  - [指数平滑法](#指数平滑法)
  - [模型预测法](#模型预测法)
- [预测模型](#预测模型)
  - [预测模型类型](#预测模型类型)
  - [预测模型构建](#预测模型构建)
  - [预测模型介绍](#预测模型介绍)
  - [预测模型参数](#预测模型参数)
</p></details><p></p>

# 预测方法

时间序列分析的一个目标是借助历史数据反映出的客观规律，对序列的未来观测值进行预测。下面简单介绍常用的预测方法，
用 `$\hat{y}_{T+h|T}$` 表示基于 `$T$` 时刻的信息构造的对 `$h$` 时刻后(即 `$T+h$` 时刻)的 `$y$` 的预测值。

## 均值预测法

均值(average 或 mean)预测是指采用历史数据的平均值作为未来观测值的预测值。
其中，简单均值(simple average)预测是均值预测的一种，是指采用历史数据的等权重平均值预测，
即：

`$$\hat{y}_{T+h|T} = \frac{1}{T}\sum_{t=1}^{T}y_{t}$$`

其中，`$h$` 为预测的步长。

## 朴素预测法

朴素(naive)预测是指采用当前的观测值作为未来观测值的预测值，即：

`$$\hat{y}_{T+h|T} = y_{T}$$`

在序列具有随机游走特征时，该预测是最优预测。另外，
该预测也是对复杂数据进行预测时常用的预测方法。

## 移动平均法

> 不是 `$\text{MA}(q)$` 模型

移动平均法是指采用滑动窗口对窗口内的数据采用等权重进行预测，即：

`$$\begin{align}
\hat{y}_{T+h|T}&=\frac{1}{k}\sum_{t=T-k+1}^{T}y_{t}  \\
&=\frac{1}{k}(y_{T-(k-1)} + y_{T-(k-2)} + \ldots + y_{T-0})
\end{align}$$`

其中，`$k$` 为滑窗窗口长度。

该预测的预测值表达式类似于简单的自回归模型（`$\text{MA}(p)$`）表达式，
不同之处在于自回归模型表达式的系数一般基于数据来估计。
该预测也是处理具有多个时间序列数据特征(如周期性、时间趋势性)的数据时常用的方法。

## 指数平滑法

指数平滑(Exponential Smoothing)法是结合了简单均值预测法、朴素预测法和移动平均法的一种预测方法。

* 简单均值预测法的缺陷是为所有历史数据赋予相同的权重
* 朴素预测法则将所有权重赋给最近的观测值
* 移动平均预测法只为最近的一系列数据赋予相同的非零权重，完全忽略较早时间的观测数据

指数平滑法的表达式为：

`$$\begin{align}
\hat{y}_{T+1|T} &=\alpha y_{T} + \alpha(1-\alpha)y_{T-1}+\alpha(1-\alpha)^{2}y_{T-2}+\ldots \\
&=\alpha y_{T} + (1-\alpha)\hat{y}_{T|T-1}
\end{align}$$`

其中，`$\alpha$` 可以通过最小二乘法来选取。

该预测方法对所有历史观测值进行加权平滑，平滑的权重的大小随着时间的推移呈指数衰减，因而得名指数平滑法。易见，
下一期的预测值 `$\hat{y}_{T+1|T}$` 可以表示为当期观测值 `$y_{T}$` 和当期预测值 `$\hat{y}_{T|T-1}$` 的加权平均。

## 模型预测法

上面的预测方法都不需要任何时间序列模型的假设。如果通过建模能够找到适合数据的时间序列模型，
则可以基于模型构造预测值。

模型(model)预测法则为基于模型构造预测的预测方法。
例如，对于符合自回归模型 `$y_{t} = 0.5y_{t-1}+\varepsilon_{t}$` 的序列，
可以与该模型在最小平方损失函数(MSE)下推出其最优预测值为 `$\hat{y}_{T+1|T}=0.5y_{T}$`。

在基于模型的预测构建过程中，最为关键的问题就是如何通过历史数据确定预测需要构建的计量模型。

# 预测模型

时间序列预测技术是指基于历史数据和时间变化规律，通过数学模型和算法对未来发展趋势进行预测的一种技术。
时间序列预测技术广泛应用于经济、金融、交通、气象等领域，以帮助人们做出更加准确的决策。

## 预测模型类型

时间序列从不同角度看有不同分类：

* 从**实现原理**角度，可以分为：
    - 传统统计学
    - 机器学习(非深度学习)
    - 深度学习
* 按**输入变量**区分，可以分为：
    - 自回归预测
    - 使用协变量预测
* 按**预测步长**区分，可以分为：
    - 单步预测
    - 多步预测
* 按**目标个数**区分，可以分为：
    - 一元预测
    - 多元预测
    - 多重预测
* 按**输出结果**区分，可以分为：
    - 点预测
    - 概率预测

这些分类是不同角度下的分类，同一种算法往往只能是分类中的一种，
例如传统的统计学模型只适合做自回归预测而不适合协变量预测。

## 预测模型构建

![img](images/timeseries.png)

## 预测模型介绍

| 模型(model)                       | 单变量/自回归               | 多变量/协变量               | 一元预测            | 多元预测                  | 多重预测             | 点预测              | 概率预测            |             |
|----------------------------------|---------------------------|---------------------------|--------------------|--------------------------|--------------------|--------------------|--------------------|-------------|
| Naive Baselines                  | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| AR                               | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| MA                               | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| ARMA                             | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| ARIMA                            | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| AutoARIMA                        | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| VARIMA                           | :white_check_mark:        | :white_check_mark:        | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| VAR                              | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| ExponentialSmoothing             | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| Theta                            | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| TBATS                            | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| FourTheta                        | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| Prophet                          | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| Fast Fourier Transform(FFT)      | :white_check_mark:        |                           | :white_check_mark: |                          |                    | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| LinearRegressionModel            | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: |                    | [模型介绍]() |
| RandomForest                     | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: |                    | [模型介绍]() |
| XGBoost                          | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: |                    | [模型介绍]() |
| LightGBM                         | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: |                    | [模型介绍]() |
| CatBoost                         | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: |                    | [模型介绍]() |
| LSTM(rnn)                        | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| GRU(rnn)                         | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| DeepAR(rnn)                      | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| BlockRNNModel(LSTM, GRU)         | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| NBeats                           | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| TCNModel                         | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| Transformer                      | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| Informer                         | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| Autoformer                       | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |
| TFT(Temporal Fusion Transformer) | :white_check_mark:        | :white_check_mark:        | :white_check_mark: | :white_check_mark:       | :white_check_mark: | :white_check_mark: | :white_check_mark: | [模型介绍]() |

## 预测模型参数

实际场景中，一般需要确定几个参数：

1. 历史窗口的大小(history length)
    - 即预测未来时，要参考过去多少时间的信息作为输入。太少可能信息量不充分，
      太多则会引入早期不相关的信息(比如疫情前的信息可能目前就不太适用了)。
2. 预测点间隔的大小(predict gap)
    - 即预测未来时，是从 T+1 开始预测，还是 T+2，T+3，这与现实的业务场景有关。
      例如像补货场景，预测 T+1 的销量，可能已经来不及下单补货了，
      所以需要扩大这个提前量，做 T+3 甚至更多提前时间的预测。
3. 预测窗口的大小(predict length)
    - 即需要连续预测多长的未来值。比如从 T+1 开始一直到 T+14 都需要预测输出。
      这一点也跟实际的业务应用场景有关。