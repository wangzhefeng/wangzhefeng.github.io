---
title: 非平稳时间序列分析
author: 王哲峰
date: '2023-03-03'
slug: timeseries-nonstationarity-model
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
</style>

<details><summary>目录</summary><p>

- [非平稳时间序列分析介绍](#非平稳时间序列分析介绍)
  - [时间序列分解](#时间序列分解)
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
  - [SARIMAX](#sarimax)
  - [SES](#ses)
  - [HWES 模型](#hwes-模型)
  - [VAR](#var)
    - [TODO](#todo)
  - [VARMA](#varma)
  - [VARMAX](#varmax)
</p></details><p></p>


# 非平稳时间序列分析介绍

在自然界中绝大部分序列都是非平稳的, 因而对非平稳序列的分析更普遍、更重要.
对非平稳时间序列分析方法可以分为**随机时间序列分析**和**确定性时间序列分析**

* 时间序列的分解
* 差分运算
* ARIMA 模型

差分运算具有强大的确定性信息提取能力, 许多非平稳序列差分后会显示出平稳序列的性质, 称这个非平稳序列为差分平稳序列,
对差分平稳序列可以使用 ARIMA(autoregression integrated moving average, 求和自回归移动平均)模型进行拟合

ARIMA 模型的实质就是差分运算和 ARMA 模型的组合, 说明任何非平稳序列如果能通过适当阶数的差分实现差分后平稳, 
就可以对差分后序列进行 ARMA 模型拟合, 而 ARMA 模型的分析方法非常成熟

## 时间序列分解

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



## SES



## HWES 模型



## VAR



### TODO



## VARMA



## VARMAX

