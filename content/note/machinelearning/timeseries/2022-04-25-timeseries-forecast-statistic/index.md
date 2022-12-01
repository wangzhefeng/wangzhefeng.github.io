---
title: 时间序列预测-统计模型
author: 王哲峰
date: '2022-04-25'
slug: timeseries-forecast-statistic
categories:
  - timeseries
tags:
  - ml
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

- [时间序列分析简介](#时间序列分析简介)
  - [数学计量基础](#数学计量基础)
  - [优缺点及意义](#优缺点及意义)
    - [优点](#优点)
    - [缺点](#缺点)
    - [意义](#意义)
  - [自回归](#自回归)
    - [一般回归问题](#一般回归问题)
    - [自回归问题](#自回归问题)
- [平稳时间序列分析](#平稳时间序列分析)
  - [AR 模型](#ar-模型)
    - [AR(`$p$`) 模型](#arp-模型)
    - [AR(`$p$`) 模型结构](#arp-模型结构)
    - [AR(`$p$`) 模型的统计性质](#arp-模型的统计性质)
    - [AR(`$p$`) 模型应用](#arp-模型应用)
    - [参考资料](#参考资料)
  - [MA 模型](#ma-模型)
    - [MA(`$q$`) 模型](#maq-模型)
    - [MA(`$q$`) 模型结构](#maq-模型结构)
    - [MA(`$q$`) 模型的统计性质](#maq-模型的统计性质)
    - [MA(`$q$`) 模型应用](#maq-模型应用)
  - [ARMA 模型](#arma-模型)
    - [ARMA(`$p$`, `$q$`) 模型](#armap-q-模型)
    - [ARMA(`$p$`, `$q$`) 模型结构](#armap-q-模型结构)
    - [ARMA(`$p$`, `$q$`) 模型的统计性质](#armap-q-模型的统计性质)
    - [ARMA(`$p$`, `$q$`) 模型应用](#armap-q-模型应用)
- [非平稳时间序列分析](#非平稳时间序列分析)
  - [时间序列分解](#时间序列分解)
    - [TODO](#todo)
  - [差分](#差分)
    - [差分运算](#差分运算)
    - [滞后算子](#滞后算子)
    - [线性差分方程](#线性差分方程)
  - [ARIMA 模型](#arima-模型)
    - [ARIMA(`$p$`, `$d$`, `$q$`) 模型](#arimap-d-q-模型)
    - [ARIMA(`$p$`, `$d$`, `$q$`) 模型结构](#arimap-d-q-模型结构)
    - [ARIMA(`$p$`, `$d$`, `$q$`) 模型的另一种形式](#arimap-d-q-模型的另一种形式)
    - [ARIMA(`$p$`, `$d$`, `$q$`) 模型的统计性质](#arimap-d-q-模型的统计性质)
    - [ARIMA 模型建模](#arima-模型建模)
    - [ARIMA(`$p$`, `$d$`, `$q$`) 模型应用](#arimap-d-q-模型应用)
- [多元时间序列分析](#多元时间序列分析)
  - [移动平均](#移动平均)
  - [指数平滑](#指数平滑)
    - [一次指数平滑](#一次指数平滑)
    - [二次指数平滑](#二次指数平滑)
    - [三次指数平滑](#三次指数平滑)
- [其他](#其他)
  - [SARIMA](#sarima)
    - [SARIMA() 模型](#sarima-模型)
  - [SARIMAX](#sarimax)
    - [TODO](#todo-1)
  - [SES](#ses)
    - [TODO](#todo-2)
  - [HWES 模型](#hwes-模型)
    - [TODO](#todo-3)
  - [VAR](#var)
    - [TODO](#todo-4)
  - [VARMA](#varma)
    - [TODO](#todo-5)
  - [VARMAX](#varmax)
    - [TODO](#todo-6)
- [参考](#参考)
</p></details><p></p>

# 时间序列分析简介

传统的参数预测方法可以分为两种：

* 一种拟合标准时间序列的参数方法，包括移动平均，指数平均等
* 另一种是考虑多因素组合的参数方法，即 AR, MA, ARMA 等模型

这类方法比较适用于小规模，单变量的预测，比如某门店的销量预测等。总的来说，基于此类方法的建模步骤是：

1. 首先，需要对观测值序列进行平稳性检测，如果不平稳，则对其进行差分运算直到差分后的数据平稳
2. 在数据平稳后则对其进行白噪声检验，白噪声是指零均值常方差的随机平稳序列
3. 如果是平稳非白噪声序列就计算 ACF(自相关系数)、PACF(偏自相关系数)，进行ARMA等模型识别，
   对已识别好的模型，确定模型参数，最后应用预测并进行误差分析

这类方法一般是统计或者金融出身的人用的比较多，对统计学或者随机过程知识的要求比较高。
而在数据挖掘的场景中比较难适用，因为需要大量的参数化建模。
比如有一个连锁门店的销售数据，要预测每个门店的未来销量，
用这类方法的话就需要对每个门店都建立模型，这样就很难操作了

传统时间序列预测模型, 通常指用于时间序列分析/预测的统计学模型, 
比如常用的有均值回归、ARIMA、指数平滑预测法等

## 数学计量基础

协方差告诉我们两个随机变量是如何一起移动的。但只用协方差衡量变量相关性存在一些问题：协方差是有量纲的，
它的大小受随机变量本身波动范围（或取值范围）所影响。因此，人们希望使用某个和协方差有关，
但是又是无量纲的测量来描述两个随机变量的相关性。最简单的做法就是用变量自身的波动对协方差进行标准化，相关系数便由此得来。
值得一提的是，相关系数仅仅刻画 X和Y之间的线性相关性；它不描述它们之间的（任何）非线性关系

## 优缺点及意义

### 优点

* 复杂度低、计算速度快

### 缺点

由于真实应用场景的复杂多样性(现实世界的时间序列往往受到各种不同因素的限制与影响, 而难以预测), 
比如受到营销计划、自然灾害等的影响

传统的单一统计学模型的准确率相对来说会比机器学习差一些, 
而机器学习模型或者更复杂的 ensemble 集成模型会有更好的效果

### 意义

传统时间序列预测模型也有其重要的意义, 比如:

* 可以作为预测的 baseline model, 为项目提供一个准确率的基准线, 来帮助评估其他模型的提升
* 由于传统时序模型由于其较好的可解释性，可以用来对数据进行前置清洗, 可以帮助剔除一些异常值, 
  比如因服务器故障或者业务线逻辑调整产生的异常值
* 可以作为集成模型中的一块, 参与时序集成模型的训练
* 可以提供一个预测结果的合理的范围, 使用这个合理的范围, 
  在黑盒模型最后输出结果时, 帮忙进行后置校准, 从而使预测系统更加稳定

## 自回归

时间序列预测的问题, 并不是普通的回归问题, 而是自回归

### 一般回归问题

一般的回归问题比如最简单的线性回归模型: 

`$$Y=a X_{1} + b X_{2} + \epsilon$$`

讨论的是因变量 `$Y$` 关于两个自变量 `$X_1$` 和 `$X_2$` 的关系, 
目的是找出最优的 `$a^{*}$` 和 `$b^{*}$`, 
来使预测值 `$y=a^{*} X_{1} + b^{*} X_{2}$` 逼近真实值 `$Y$`

### 自回归问题

自回归模型中, 自变量 `$X_1$` 和 `$X_2$` 都为 `$Y$` 本身, 
也就是说 

`$$Y(t)=a Y(t-1)+ b Y(t-2) + \epsilon$$`

其中 

* `$Y(t-1)$` 为 `$Y$` 在 `$t-1$` 时刻的值
* `$Y(t-2)$` 为 `$Y$` 在 `$t-2$` 时刻的值
  
换句话说, 现在的 `$Y$` 值由过去的 `$Y$` 值决定, 
因此自变量和因变量都为自身, 因此这种回归叫自回归

自回归模型都有着严格理论基础,讲究时间的平稳性, 
需要对时间序列进行分析才能判断是否能使用此类模型. 
这些模型对质量良好的时间序列有比较高的精度. 传统的自回归模型有: 


* 自回归模型(AR, Auto Regressive)
* 移动平均模型(MA, Moving Average)
* 自回归移动平均模型(ARMA, Auto-Regressive Moving Average)
* 指数平滑(Exponential Smoothing)
* 差分自回归移动平均模型(ARIMA)

# 平稳时间序列分析

一个时间序列经过预处理被识别为 **平稳非白噪声时间序列**, 就说明该序列是一个蕴含相关信息的平稳序列. 
在统计上, 通常是建立一个线性模型来拟合该序列的发展, 借此提取该序列中的有用信息. 

ARMA(Auto Regression Moving Average) 模型的全称是**自回归移动平均模型**, 
它是目前最常用的拟合平稳序列的模型.它又可以细分为 AR 模型, MA 模型, ARMA 模型三类

假设一个时间序列经过预处理被识别为**平稳非白噪声时间序列**, 就可以利用ARMA 模型对该序列建模. 建模的基本步骤为:

1. 求出该观察序列的 **样本自相关系数(ACF)** 和 **样本偏自相关系数(PACF)** 的值
2. 根据样本自相关系数和偏自相关系数的性质，选择阶数适当的 ARMA(`$p$`, `$q$`) 模型进行拟合
3. 估计模型中的位置参数的值
4. 检验模型的有效性
    - 如果拟合模型通不过检验, 转向步骤 2, 重新选择模型再拟合
5. 模型优化
    - 如果拟合模型通过检验, 仍然转向步骤 2, 充分考虑各种可能, 
      建立多个拟合模型, 从所有通过检验的拟合模型中选择最优模型
6. 利用拟合模型, 预测序列的将来走势

## AR 模型

### AR(`$p$`) 模型

AR(`$p$`), `$p$` 阶自回归模型, 该模型适用于无趋势(trend)和季节性(seasonal)因素的单变量时间序列

### AR(`$p$`) 模型结构

`$$
\left\{
\begin{array}{**lr**}
y_{t}=c + \phi_{1}y_{t-1} + \phi_{2}y_{t-2} + \cdots + \phi_{p}y_{t-p} + \epsilon_{t} & \\
\phi_{p} \neq 0 & \\
E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t & \\
E(y_{s}\epsilon_{t}) = 0, \forall s < t & 
\end{array}
\right.$$`

其中: 

* `$\epsilon_{t}$` 是白噪声序列
* `$c$` 是常数, 表示时间序列没有进行 0 均值化

### AR(`$p$`) 模型的统计性质

- 均值
- 方差
- 自协方差函数
- 自相关系数
- 偏自相关系数

### AR(`$p$`) 模型应用

```python
from statsmodels.tsa.ar_model import AR
from random import random

data = [x + random() for x in range(1, 100)]

model = AR(data)
model_fit = model.fit()

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

### 参考资料

* [Autoregressive models](https://otexts.com/fpp2/AR.html)


## MA 模型

### MA(`$q$`) 模型

MA(`$p$`), `$q$` 阶移动平均模型, 残差误差(residual erros)的线性函数, 
与计算时间序列的移动平均不同, 该模型适用于无趋势(trend)和季节性(seasonal)因素的单变量时间序列

### MA(`$q$`) 模型结构

`$$\left\{
\begin{array}{**lr**}
y_{t}=\mu + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + \cdots + \theta_{q}\epsilon_{t-q}& \\
\theta_{q} \neq 0 & \\
E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t &
\end{array}
\right.$$`

其中: 

- `$\epsilon_{t}$` 是白噪声序列
- `$\mu$` 是常数

### MA(`$q$`) 模型的统计性质

- 常数均值
- 常数方差
- 自协方差函数只与滞后阶数相关, 且 `$q$` 阶截尾
- 自相关系数 `$q$` 阶截尾

### MA(`$q$`) 模型应用

```python
from statsmodesl.tsa.arima_model import ARMA
from random import random

data = [x + random() for x in range(1, 100)]

model = ARMA(data, order = (0, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

## ARMA 模型

### ARMA(`$p$`, `$q$`) 模型

ARMA(`$p$`, `$q$`), 自回归移动平均模型, 是时间序列和残差误差的线性函数, 
是 AR(`$p$`) 和 MA(`$q$`) 模型的组合, 
该模型适用于无趋势(trend)和季节性(seasonal)因素的单变量时间序列

### ARMA(`$p$`, `$q$`) 模型结构

`$$\left\{
\begin{array}{**lr**}
x_{t}=\phi_{0} + \phi_{1}x_{t-1} + \phi_{2}x_{t-2} + \cdots + \phi_{p}x_{t-p} + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + \cdots + \theta_{q}\epsilon_{t-q} & \\
\phi_{p} \neq 0, \theta_{q} \neq 0& \\
E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t & \\
E(x_{s}\epsilon_{t}) = 0, \forall s < t & 
\end{array}
\right.$$`

ARMA 模型的另一种形式: 

`$$(1-\sum_{i=1}^{p}\phi_{i}B^{i})x_{t} = (1 - \sum_{i=1}^{q}\theta_{i}B^{i})\epsilon_{t}$$`
`$$\Phi(B)x_{t} = \Theta(B)\epsilon_{t}$$`

- 当 `$q = 0$` 时, ARMA(`$p$`, `$q$`) 模型就退化成了 AR(`$p$`) 模型.
- 当 `$p = 0$` 时, ARMA(`$p$`, `$q$`) 模型就退化成了 MA(`$q$`) 模型.
- 所以 AR(`$p$`) 和 MA(`$q$`) 实际上是 ARMA(`$p$`, `$p$`) 模型的特例, 它们统称为 ARMA 模型. 
  而 ARMA(`$p$`, `$p$`) 模型的统计性质也正是 AR(`$p$`) 模型和 MA(`$p$`) 模型统计性质的有机结合.

### ARMA(`$p$`, `$q$`) 模型的统计性质

- 均值
- 自协方差函数
- 自相关系数

### ARMA(`$p$`, `$q$`) 模型应用

```python
from statsmodels.tsa.arima_model import ARMA
from random import random

data = [random.() for x in range(1, 100)]

model = ARMA(data, order = (2, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

# 非平稳时间序列分析

在自然界中绝大部分序列都是非平稳的, 因而对非平稳序列的分析更普遍、更重要.
对非平稳时间序列分析方法可以分为**随机时间序列分析**和**确定性时间序列分析**

- 时间序列的分解
- 差分运算
- ARIMA 模型

差分运算具有强大的确定性信息提取能力, 许多非平稳序列差分后会显示出平稳序列的性质, 称这个非平稳序列为差分平稳序列,
对差分平稳序列可以使用 ARIMA(autoregression integrated moving average, 求和自回归移动平均)模型进行拟合

ARIMA 模型的实质就是差分运算和 ARMA 模型的组合, 说明任何非平稳序列如果能通过适当阶数的差分实现差分后平稳, 
就可以对差分后序列进行 ARMA 模型拟合, 而 ARMA 模型的分析方法非常成熟

## 时间序列分解

### TODO

## 差分

### 差分运算

### 滞后算子

### 线性差分方程


## ARIMA 模型

### ARIMA(`$p$`, `$d$`, `$q$`) 模型

ARIMA(Autoregressive Integrated Moving Average), 差分自回归移动平均模型, 
是差分后的时间序列和残差误差的线性函数

### ARIMA(`$p$`, `$d$`, `$q$`) 模型结构

`$$\left\{
\begin{array}{**lr**}
\Phi(B)\Delta^{d}x_{t} = \Theta(B)\epsilon_{t}& \\
E(\epsilon_{t}) =0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{s}\epsilon_{t}) = 0, s \neq t& \\
E(x_{s}\epsilon_{t}) = 0, \forall s < t&
\end{array}
\right.$$`

其中: 

- `${\epsilon_{t}}$` 为零均值白噪声序列
- `$\Delta^{d} = (1-B)^{d}$`
- `$\Phi(B) = 1-\sum_{i=1}^{p}\phi_{i}B^{i}$` 为平稳可逆 ARMA(`$p$`, `$q$`) 模型的自回归系数多项式
- `$\Theta(B) = 1 + \sum_{i=1}^{q}\theta_{i}B^{i}$` 为平稳可逆 ARMA(`$p$`, `$q$`) 模型的移动平滑系数多项式

ARIMA 之所以叫 **求和自回归移动平均** 是因为: `$d$` 阶差分后的序列可以表示为下面的表示形式, 
即差分后序列等于原序列的若干序列值的加权和, 而对它又可以拟合 ARMA 模型: 

`$$\Delta^{d}x_{t} = \sum_{i=0}^{d}(-1)C_{d}^{i}x_{t-i}$$`

其中: 

* `$C_{d}^{i} = \frac{d!}{i!(d-i)!}$`

### ARIMA(`$p$`, `$d$`, `$q$`) 模型的另一种形式

`$$\Delta^{d}x_{t} = \frac{\Theta(B)}{\Phi(B)}\epsilon_{t}$$`

其中: 

- 当 `$d=0$` 时 ARIMA(`$p$`, `$0$`, `$q$`) 模型就是 ARMA(`$p$`, `$q$`) 模型
- 当 `$p=0$` 时, ARIMA(`$0$`, `$d$`, `$q$`) 模型可以简记为 IMA(`$d$`, `$q$`) 模型
- 当 `$q=0$` 时, ARIMA(`$p$`, `$d$`, `$0$`) 模型可以简记为 ARI(`$p$`, `$d$`) 模型
- 当 `$d=1, p=q=0$` 时, ARIMA(`$0$`, `$1$`, `$0$`) 模型为 随机游走 (random walk) 模型:
  
`$$\left\{
\begin{array}{**lr**}
x_{t} = x_{t-1} + \epsilon_{t}& \\
E(\epsilon_{t}) =0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{s}\epsilon_{t}) = 0, s \neq t& \\
E(x_{s}\epsilon_{t}) = 0, \forall s < t&
\end{array}
\right.$$`

### ARIMA(`$p$`, `$d$`, `$q$`) 模型的统计性质

1. 平稳性
2. 方差齐性

### ARIMA 模型建模

1. 获得时间序列观察值
2. 平稳性检验
    - 不平稳: 差分运算 => 平稳性检验
    - 平稳: 下一步
3. 白噪声检验
    - 不通过: 拟合 ARMA 模型 => 白噪声检验
    - 通过: 分析结束

### ARIMA(`$p$`, `$d$`, `$q$`) 模型应用

```python
from statsmodels.tsa.arima_model import ARIMA
from random import random

data = [x + random() for x in range(1, 100)]

model = ARIMA(data, order = (1, 1, 1))
model_fit = model.fit(disp = True)

y_hat = model_fit.predict(len(data), len(data), typ = "levels")
print(y_hat)
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from datetime import timedelta
from matplotlib.pyplot import rcParams
rcParams["figure.figsize"] = 15, 6
# 平稳性检验(AD检验)
from statsmodels.tsa.stattools import adfuller
# 模型分解
from statsmodels.tsa.seasonal import seasonal_decompose
# ARIMA 模型
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
```


ADFuller 平稳性检验

```python
def stationarity_test(ts):
      # rolling statistics
      rollmean = pd.Series.rolling(ts, window = 12).mean()
      rollstd = pd.Series.rolling(ts, window = 12).std()

      orig = plt.plot(ts, color = "blue", label = "Original")
      mean = plt.plot(rollmean, color = "red", label = "Rolling mean")
      std = plt.plot(rollstd, color = "black", label = "Rolling std")
      plt.legend(loc = "best")
      plt.title("Rolling mean & Standard Deviation")
      plt.show()

      # Dickey Fuller test
      print("Results of Dickey-Fuller Test:")
      dftest = adfuller(ts, autolag = "AIC")
      dfountput = pd.Series(dftest[0:4], 
                           index = ["Test Statistic", 
                                    "p-value", 
                                    "#lag used", 
                                    "Number of observation used"])
      for key, value in dftest[4].items():
         dfountput["Critical Value(%s)" % key] = value
```

ACF 自相关函数, PACF 偏自相关函数

```python
def acf_pacf(data):
      lag_acf = acf(data, nlags = 20)
      lag_pacf = pacf(data, nlags = 20, method = "ols")

      plt.subplot(121)
      plt.plot(lag_acf)
      plt.axhline(y = 0, linestyle = "--", color = "gray")
      plt.axhline(y = - 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
      plt.axhline(y = 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
      plt.title("Autocorrelation Function")

      plt.subplot(122)
      plt.plot(lag_pacf)
      plt.axhline(y = 0, linestyle = "--", color = "gray")
      plt.axhline(y = - 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
      plt.axhline(y = 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
      plt.title("Partial Autocorrelation Function")

      plt.tight_layout()
```

```python
def arima_performance(data, order1):
      model = ARIMA(data, order = order1)
      results_arima = model.fit(disp = -1)
      results_arima_value = results_arima.fittedvalues
      results_future = result_airma.forecast(7)
      return results_arima_value, results_future
```

```python
def arima_plot(data, results_arima_value):
      plt.plot(data)
      plt.plot(results_arima_value, color = "red")
      plt.title("RSS: %.4f" % sum((results_arima_value) ** 2))
```

```python
def add_season(ts_recover_trend, startdate):
      ts2_season = ts2_season
      values = []
      low_conf_values = []
```

# 多元时间序列分析

## 移动平均

## 指数平滑

### 一次指数平滑

### 二次指数平滑

### 三次指数平滑

# 其他

## SARIMA

### SARIMA() 模型

该模型适用于含有趋势(trend)或季节性(seasonal)因素的单变量时间序列

```python
from statsmodels.tsa.statspace.sarima import SARIMAX
from random import random

data = [x + random() for x in range(1, 100)]

model = SARIMAX(data, order = (1, 1, 1), seasonal_order = (1, 1, 1, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

## SARIMAX

### TODO



## SES

### TODO




## HWES 模型

### TODO




## VAR

### TODO




## VARMA

### TODO




## VARMAX

### TODO

# 参考

* [[1]时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484974&idx=1&sn=d841c644fd9289ad5ec8c52a443463a5&chksm=cecef3dbf9b97acd8a9ededc069851afc00db422cb9be4d155cb2c2a9614b2ee2050dc7ab4d7&scene=21#wechat_redirect)
* [时间序列统计分析](https://mp.weixin.qq.com/s/INZgM6hLSEpboaNhS22CaA)
