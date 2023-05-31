---
title: 时间序列预测
author: 王哲峰
date: '2022-04-25'
slug: forecast
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

- [为什么时间序列预测很难](#为什么时间序列预测很难)
  - [时间序列预测很难](#时间序列预测很难)
  - [样本量与模型精度](#样本量与模型精度)
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
- [模型参数](#模型参数)
- [时间序列模型构建](#时间序列模型构建)
- [参考](#参考)
</p></details><p></p>

# 为什么时间序列预测很难

机器学习和深度学习已越来越多应用在时序预测中。ARIMA 或指数平滑等经典预测方法正在被 XGBoost、
高斯过程或深度学习等机器学习回归算法所取代

尽管时序模型越来越复杂，但人们对时序模型的性能表示怀疑。有研究表明，
复杂的时序模型并不一定会比时序分解模型有效（Makridakis, 2018）

## 时间序列预测很难

时间序列是按时间排序的值，但时序预测具有很大的挑战性。从模型难度和精度角度考虑，
时序模型的比常规的回归和分类任务更难

原因1：序列是非平稳的

* 平稳性是时间序列的核心概念，如果时间序列的趋势（例如平均水平）不随时间变化，则该时间序列是平稳的。
  许多现有方法都假设时间序列是平稳的，但是趋势或季节性会打破平稳性

原因2：依赖外部数据

* 除了时间因素之外，时间序列通常还有额外的依赖性。时空数据是一个常见的例子，每个观察值都在两个维度上相关，
  因此数据具有自身的滞后（时间依赖性）和附近位置的滞后（空间依赖性）

原因3：噪音和缺失值

* 现实世界受到噪音和缺失值的困扰，设备故障可能会产生噪音和缺失值。
  传感器故障导致数据丢失，或者存在干扰，都会带来数据噪音。

原因4：样本量有限

* 时间序列往往都只包含少量的观察值，可能没有足够的数据来构建足够的模型。
  数据采集的频率影响了样本量，同时也会遇到数据冷启动问题

## 样本量与模型精度

时序模型往往无法进行完美预测，这可能和时序数据的样本量相关。在使用较大的训练集时，
具有大模型往往比参数较少的模型表现更好。在时序序列长度小于 1000 时，
深度模型往往并不会比时序分类模型更好

下面对比了模型精度与样本个数的关系，这里尝试了五种经典方法（ARIMA、ETS、TBATS、Theta 和 Naive）和五种机器学习方法（高斯过程、
M5、LASSO、随机森林和 MARS）。预测任务是来预测时间序列的下一个值

结果如下图所示，轴表示训练样本大小，即用于拟合预测模型的数据量。
轴表示所有时间序列中每个模型的平均误差，使用交叉验证计算得出。

![img](images/forecast.png)

当只有少数观测值可用时，基础方法表现出更好的性能。然而，随着样本量的增加，机器学习方法优于经典方法

进一步可以得出以下结论：

* 机器学习方法拥有很强的预测能力，前提是它们具有足够大的训练数据集
* 当只有少量观测值可用时，推荐首选 ARIMA 或指数平滑等经典方法
* 可以将指数平滑等经典方法与机器学习相结合可以提高预测准确性

# 预测方法

时间序列分析的一个目标是借助历史数据反映出的客观规律，对序列的未来观测值进行预测。下面简单介绍常用的预测方法。
用 `$\hat{y_{T+h|t}}$` 表示基于 `$T$` 时刻的信息构造的对 `$h$` 时刻后(即 `$T+h$`时刻)的 `$y$` 的预测值

## 均值预测法

均值(average 或 mean)预测是指采用历史数据的平均值作为未来观测值的预测值。
其中，简单均值(simple average)预测是均值预测的一种，是指采用历史数据的等权重平均值预测，
即：

`$$\begin{align}
\hat{y}_{T+h|T} &= \frac{1}{T}\sum_{t=1}^{T}y_{t} \\
&=\frac{1}{T}y_{1} + \frac{1}{T}y_{2} + \ldots + \frac{1}{T}y_{T}
\end{align}$$`

其中：

* `$h$` 为预测的步长

## 朴素预测法

朴素(naive)预测是指采用当前的观测值作为未来观测值的预测值，即：

`$$\hat{y}_{T+h|T} = y_{T}$$`

在序列具有随机游走特征时，该预测是最优预测。另外，该预测也是对复杂数据进行预测时常用的预测方法

## 移动平均法

> 不是 `$MA(q)$` 模型

移动平均法是指采用滑动窗口对窗口内的数据采用等权重进行预测，即：

`$$\begin{align}
\hat{y}_{T+h|T}&=\frac{1}{k}\sum_{t=T-k+1}^{T}y_{t}  \\
&=\frac{1}{k}y_{T-(k-1)} + \frac{1}{k}y_{T-(k-2)} + \ldots + \frac{1}{k}y_{T-0}
\end{align}$$`

其中：

* `$k$` 为滑窗窗口长度

该预测的预测值表达式类似于简单的自回归模型表达式，不同之处子域自回归模型表达式的系数一般基于数据来估计。
该预测也是处理具有多个时间序列数据特征(如周期性、时间趋势性)的数据时常用的方法

## 指数平滑法

指数平滑(Exponential Smoothing)法是结合了简单均值预测法、朴素预测法和移动平均法的一种预测方法

* 简单均值预测法的缺陷是为所有历史数据赋予相同的权重
* 朴素预测法则将所有权重赋给最近的观测值
* 移动平均预测法只为最近的一系列数据赋予相同的非零权重，完全忽略较早时间的观测数据

指数平滑法的表达式为：

`$$\begin{align}
\hat{y}_{T+1|T} &=\alpha y_{T} + \alpha(1-\alpha)y_{T-1}+\alpha(1-\alpha)^{2}y_{T-2}+\ldots \\
&=\alpha y_{T} + (1-\alpha)\hat{y}_{T|T-1}
\end{align}$$`

其中：

* `$\alpha$` 可以通过最小二乘法来选取

该预测方法对所有历史观测值进行加权平滑，平滑的权重的大小随着时间的推移呈指数衰减，因而得名指数平滑法。易见，
下一期的预测值 `$\hat{y}_{T+1|T}$` 可以表示为当期观测值 `$y_{T}$` 和当期预测值 `$\hat{y}_{T|T-1}$` 的加权平均

## 模型预测法

上面的预测方法都不需要任何时间序列模型的假设。如果通过建模能够找到适合数据的时间序列模型，
则可以基于模型构造预测值。模型(model)预测法则为基于模型构造预测的预测方法。
例如，对于符合自回归模型 `$y_{t} = 0.5y_{t-1}+\varepsilon_{t}$` 的序列，科技与该模型在最小平方损失函数(MSE)下推出其最优预测值为 `$\hat{y}_{T+1|T}=0.5y_{T}$`

在基于模型的预测构建过程中，最为关键的问题就是如何通过历史数据确定预测需要构建的计量模型

# 预测模型

时间序列预测技术是指基于历史数据和时间变化规律，通过数学模型和算法对未来发展趋势进行预测的一种技术。
时间序列预测技术广泛应用于经济、金融、交通、气象等领域，以帮助人们做出更加准确的决策

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

![img](images/timeseries_type.png)

这些分类是不同角度下的分类，同一种算法往往只能是分类中的一种，
例如例如传统的统计学模型只适合做自回归预测而不适合协变量预测

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

# 模型参数

实际场景中，一般需要确定几个参数：

1. 历史窗口的大小(history length)
    - 即预测未来时，要参考过去多少时间的信息作为输入。太少可能信息量不充分，
      太多则会引入早期不相关的信息(比如疫情前的信息可能目前就不太适用了)
2. 预测点间隔的大小(predict gap)
    - 即预测未来时，是从 T+1 开始预测，还是 T+2，T+3，这与现实的业务场景有关。
      例如像补货场景，预测 T+1 的销量，可能已经来不及下单补货了，
      所以需要扩大这个提前量，做 T+3 甚至更多提前时间的预测
3. 预测窗口的大小(predict length)
    - 即需要连续预测多长的未来值。比如从 T+1 开始一直到 T+14 都需要预测输出。
      这一点也跟实际的业务应用场景有关

# 时间序列模型构建

用机器学习算法构造时间序列预测模型，关键的思路在于，通过时间滑窗，人为地构造 “未来” Target，来给算法进行学习

![img](images/ml.png)

和之前一样，从时间的角度上来看，有历史数据和新数据。但这里，不能简单地把历史数据作为训练集、把新数据作为测试集

1. 首先，在历史数据上，通过截取不同时间窗口的数据来构造一组或几组数据
    - 比如，历史数据是 2017 年 1 月到 12 月每家店每天的销售数据，那么可以截取 3 组数据（见上图的深绿、浅绿部分）：
        - 2017 年 1 月到 10 月的数据
        - 2017 年 2 月到 11 月的数据
        - 2017 年 3 月到 12 月的数据
2. 然后，人为地给每组数据划分历史窗口（对应上图的深绿色部分）和未来窗口（对应上图的浅绿色部分）
    - 比如，对于 2017 年 1 月到 10 月的数据，把 1 月到 9 月作为历史窗口，10 月作为未来窗口，以此类推
3. 接着，分别给每组数据构建预测特征，包括历史特征（预测特征 A）和未来特征（预测特征 B）。
   而此时，每组数据还有预测 Target

这个时候，把得到的所有预测特征（例子里是三组预测特征）都合并起来作为训练集特征、
把所有预测 Target（例子里是三组预测 Target）合并起来作为训练集 Target，之后就可以构建机器学习模型了

有了训练集和训练模型，还差测试集。测试集的构建遵循之前的数据处理逻辑，拿历史数据构建历史特征，
拿新数据构建未来特征，然后把这些特征加入到从训练集上训练出的预测模型中去，即可得到任务需要的最终预测值。
这里需要注意，划多少个时间窗口因数据而异。此外，数据的历史窗口（图上深绿部分）和未来窗口（图上浅绿部分）可以是定长也可以是变长，
看具体情况


# 参考

* [为什么时序预测很难](https://mp.weixin.qq.com/s/K0VVbZBcFJB5ctKWeMHUgQ)
