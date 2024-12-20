---
title: 时间序列平滑及预测
author: wangzf
date: '2022-11-19'
slug: timeseries-smoothing
categories:
  - timeseries
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

- [时间序列平滑简介](#时间序列平滑简介)
- [移动平均](#移动平均)
  - [简单移动平均](#简单移动平均)
  - [加权移动平均](#加权移动平均)
- [指数平滑](#指数平滑)
    - [一次指数平滑](#一次指数平滑)
    - [二次指数平滑](#二次指数平滑)
    - [三次指数平滑](#三次指数平滑)
  - [平滑系数 `$\alpha$` 的取值](#平滑系数-alpha-的取值)
  - [平滑初值 `$S^{(1)}_{1}$` 的取值](#平滑初值-s1_1-的取值)
</p></details><p></p>

# 时间序列平滑简介

数据平滑通常是为了消除一些极端值或测试误差. 
即使有些极端值本身是真实的, 但是并没有反映出潜在的数据模式, 仍需处理

* 数据平滑方法
    - 差分运算
    - 移动平均
        - 简单移动平均
        - 加权移动平均
    - 指数加权移动平均/指数平滑
        - 一次指数平滑
        - 二次指数平滑
        - 三次指数平滑
    - 时间序列分解
    - 日历调增
    - 数学变换
* 数据平滑工具
    - `.diff()`：差分
    - `.shift()`
    - `.rolling()`
    - `.expanding()`
    - `.ewm()`
    - `.pct_change()`

# 移动平均

移动平均模型是最简单的时间序列建模方法, 即: 下一个值是所有一个时间窗口中值的平均值。
时间窗口越长, 预测值的趋势就越平滑

* 移动平均(Moving Average, MA)
    - 给定时间窗口内样本点相同权重
* 加权移动平均(Weighted Moving Average, WMA)
    - 给邻近点指定更高权重

## 简单移动平均

`$$m_{t} = \frac{1}{k}\sum_{i=1}^{k}y_{t-i}$$`

## 加权移动平均

`$$m_{t} = \sum_{i=1}^{k}\omega_{i}y_{t-i}$$`

# 指数平滑

指数平滑(Exponential Smoothing, ES) 也叫做指数加权移动平均(Exponential Weighted Moving Average, EWMA)，
由布朗提出、他认为时间序列的态势具有稳定性或规则性, 所以时间序列可被合理地顺势推延; 
他认为最近的过去态势, 在某种程度上会持续的未来, 所以将较大的权数放在最近的资料 

指数平滑是在 20 世纪 50 年代后期提出的, 并激发了一些十分成功的预测方法. 
使用指数平滑方法生成的预测是过去观测值的加权平均值, 并且随着过去观测值离预测值的距离的增大,
权重呈指数型衰减. 换句话说, 观测值越近, 相应的权重越高. 该框架能够快速生成可靠的预测结果, 
并且适用于广泛的时间序列, 这是一个巨大的优势并且对于工业应用来说非常重要

指数平滑使用了与移动平均相似的逻辑, 但是, 指数平滑对每个观测值分配了不同的递减权重, 
即: 离现在的时间距离越远, 时间序列观测值的重要性就越低。
所有的指数平滑法都要更新上一时间步长的计算结果, 并使用当前时间步长的数据中包含的新信息. 
通过"混合"新信息和旧信息来实现, 而相关的新旧信息的权重由一个可调整的参数来控制 

* 一次指数平滑(Simple Exponential Smoothing, SES)
    - 从最邻近到最早的数据点的权重呈现指数型下降的规律, 针对没有趋势和季节性的序列
* 二次指数平滑(Holt Exponential Smoothing, HES)
    - 通过引入一个额外的系数来解决指数平滑无法应用于具有趋势性数据的问题, 
      针对有趋势但没有季节性的序列
* 三次指数平滑(Holt-Winters Exponential Smoothing, HWES)
    - 通过再次引入一个新系数的方式同时解决了二次平滑无法解决具有季节性变化数据的不足

### 一次指数平滑

当时间序列数据没有明显的变化趋势, 就可以使用一次指数平滑. 实际就是对历史数据的加权平均, 
它可以用于任何一种没有明显函数规律但确实存在某种前后关联的时间序列的短期预测. 
但是一次指数平滑也具有一定的局限性: 

1. 预测值不能反映趋势变动、季节波动等有规律的变动
2. 这种方法多适用于短期预测, 而不适合作中长期的预测
3. 由于预测值是历史数据的均值, 因此与实际序列的变化相比有滞后现象

一次指数平滑的预测公式/递推公式如下:

`$$S_{i} = \alpha x_{i} + (1 - \alpha) S_{i-1}$$`

其中: 

* `$S_{i}$`: `$i$` 时刻的平滑值/预测值
* `$x_{i}$`: `$i$` 时刻的实际观测值
* `$S_{i-1}$`: `$i-1$` 时刻的平滑值/预测值
* `$\alpha \in [0, 1]$`: 平滑系数/衰减因子, 可以控制对历史数据的遗忘程度, 决定了之前观测值的权重下降的速度
  当 `$\alpha$` 接近 1 时表示只保留当前时间点的值。
  平滑因子越小, 时间序列就越平滑, 因为当平滑因子接近 0 时, 指数平滑接近移动平均模型

递推公式展开可以得到以下形式:

`$$\begin{aligned} 
S_{i} &= \alpha x_{i}+(1-\alpha) S_{i-1} \\ 
      &= \alpha x_{i}+(1-\alpha)\left[\alpha x_{i-1}+(1-\alpha) S_{i-2}\right] \\ 
      &= \alpha x_{i}+(1-\alpha)\left[\alpha x_{i-1}+(1-\alpha)\left[\alpha x_{i-2}+(1-\alpha) S_{i-3}\right]\right] \\ 
      &= \alpha\left[x_{i}+(1-\alpha) x_{i-1}+(1-\alpha)^{2} x_{i-2}+(1-\alpha)^{3} S_{i-3}\right] \\ 
      &= \ldots \\ 
      &= \alpha \sum_{j=0}^{i}(1-\alpha)^{j} x_{i-j} 
\end{aligned}$$`

从上式即可看出指数平滑考虑所有历史观测值对当前值的影响, 但影响程度随时间增长而减小, 
对应的时间序列预测公式:

`$$\hat{x}_{i+h} = S_{i}$$`

其中:

* `$S_{i}$` 表示最后一个时间点对应的值
* `$h=1$` 表示下个预测值

### 二次指数平滑

二次指数平滑是对一次指数平滑的再平滑. 它适用于具线性趋势的时间数列. 
虽然一次指数平均在产生新的数列的时候考虑了所有的历史数据, 
但是仅仅考虑其静态值, 即没有考虑时间序列当前的变化趋势. 
所以二次指数平滑一方面考虑了所有的历史数据, 另一方面也兼顾了时间序列的变化趋势. 

当时间序列中存在趋势时, 使用二次指数平滑, 它只是指数平滑的两次递归使用

二次指数平滑的数学表示:

`$$y=\alpha x_{t} + (1 - \alpha)(y_{t-1} + b_{t-1})$$`
`$$b_{t}=\beta (y_{t} - y_{t-1}) + (1 - \beta)b_{t-1}$$`

其中:

- `$\alpha \in [0, 1]$` 是一个平滑因子
- `$\beta \in [0, 1]$` 是趋势平滑因子

* 预测公式

`$$\left \{
\begin{array}{lcl}
S^{(1)}_{t} = \alpha x_{t-1} + (1 - \alpha) S^{(1)}_{t-1} \\ 
S^{(2)}_{t} = \alpha S^{(1)}_{t} + (1 - \alpha)S^{(2)}_{t-1} \\ 
\end{array} \right.$$`

`$$\left \{
\begin{array}{lcl}
a_{t} = 2S^{(1)}_{t} - S^{(2)}_{t} \\
b_{t} = \frac{\alpha}{1 - \alpha}(S^{(1)}_{t} - S^{(2)}_{t}) \\
\end{array} \right.$$`

`$$x^{*}_{t+T} = a_{t} + b_{t} T$$`

其中: 

* `$x^{*}_{t+T}$`: 未来 T 期的预测值
* `$\alpha$`: 平滑系数

### 三次指数平滑

若时间序列的变动呈现出二次曲线趋势, 则需要采用三次指数平滑法进行预测. 
实际上是在二次指数平滑的基础上再进行一次指数平滑. 它与二次指数平滑的区别
就是三次平滑还考虑到了季节性效应


三次指数平滑通过添加季节平滑因子扩展二次指数平滑

三次指数平滑的数学表示:

`$$y=\alpha \frac{x_{t}}{c_{t-L}} + (1 - \alpha)(y_{t-1} + b_{t-1})$$`
`$$b_{t}=\beta (y_{t} - y_{t-1}) + (1 - \beta)b_{t-1}$$`
`$$c_{t}=\gamma \frac{x_{t}}{y_{t}} + (1-\gamma)c_{t-L}$$`

其中:

- `$\alpha \in [0, 1]$` 是一个平滑因子
- `$\beta \in [0, 1]$` 是趋势平滑因子
- `$\gamma$` 是季节长度

* 预测公式

`$$\left \{
\begin{array}{lcl}
S^{(1)}_{t} = \alpha x_{t-1} + (1 - \alpha) S^{(1)}_{t-1} \\
S^{(2)}_{t} = \alpha S^{(1)}_{t} + (1 - \alpha)S^{(2)}_{t-1} \\
S^{(3)}_{t} = \alpha S^{(2)}_{t} + (1 - \alpha)S^{(3)}_{t-1} \\
\end{array} \right.$$`

`$$\left \{
\begin{array}{lcl}
a_{t} = 3S^{(1)}_{t} - 3S^{(2)}_{t} + S^{(3)}_{t} \\
b_{t} = \frac{\alpha}{2(1 - \alpha)^{2}}[(6 - 5 \alpha)S^{(1)}_{t} - 2(5 - 4\alpha)S^{(2)}_{t} + (4 - 3\alpha)S^{(3)}_{t}] \\
c_{t} = \frac{\alpha^{2}}{2(1 - \alpha)^{2}}(S^{(1)}_{t} - 2S^{(2)}_{t} + S^{(3)}_{t}) \\
\end{array} \right.$$`

`$$x^{*}_{t+T} = a_{t} + b_{t} T + c_{t} T^{2}$$`

其中: 

* `$x^{*}_{t+T}$`: 未来 T 期的预测值
* `$\alpha$`: 平滑系数

## 平滑系数 `$\alpha$` 的取值

1. 当时间序列呈现较稳定的水平趋势时, 应选较小的 `$\alpha$`, 一般可在 0.05~0.20 之间取值
2. 当时间序列有波动, 但长期趋势变化不大时, 可选稍大的 `$\alpha$` 值, 常在 0.1~0.4 之间取值
3. 当时间序列波动很大, 长期趋势变化幅度较大, 呈现明显且迅速的上升或下降趋势时, 宜选择较大的 `$\alpha$` 值, 
   如可在 0.6~0.8 间选值. 以使预测模型灵敏度高些, 能迅速跟上数据的变化. 
4. 当时间序列数据是上升(或下降)的发展趋势类型, `$\alpha$` 应取较大的值, 在 0.6~1 之间

## 平滑初值 `$S^{(1)}_{1}$` 的取值

不管什么指数平滑都会有个初值, 平滑初值 `$S^{(1)}_{1}$` 的取值遵循如下规则即可: 

* 如果时间序列长度小于 20, 一般取前三项的均值

`$$S^{(1)}_{1} = S^{(2)}_{1} = S^{(3)}_{1} = \frac{1}{n}(x_{1} + x_{2} + x_{3})$$`

* 如果时间序列长度大于 20, 看情况调整第一项的值, 一般取第一项就行, 
  因为数据多, `$y^{*}_{0}$` 的取值对最后的预测结果影响不大

