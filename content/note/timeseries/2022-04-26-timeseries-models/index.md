---
title: timeseries-models
author: 王哲峰
date: '2022-04-26'
slug: timeseries-models
categories:
  - timeseries
tags:
  - ml
---




# AR

AR($p$) 模型

AR($p$), $p$ 阶自回归模型, 该模型适用于无趋势 (trend)
和季节性 (seasonal) 因素的单变量时间序列

## AR($p$) 模型结构

\left\{
\begin{array}{**lr**}
x_{t}=\phi_{0} + \phi_{1}x_{t-1} + \phi_{2}x_{t-2} + \cdots + \phi_{p}x_{t-p} + \epsilon_{t} & \\
\phi_{p} \neq 0 & \\
E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t & \\
E(x_{s}\epsilon_{t}) = 0, \forall s < t & 
\end{array}
\right.

其中: 

-  $\epsilon_{t}$ 是白噪声序列
-  $\phi_{0}$ 是常数, 表示时间序列没有进行 0 均值化



## AR($p$) 模型的统计性质

-  均值
-  方差
-  自协方差函数
-  自相关系数
-  偏自相关系数


## AR($p$) 模型应用

```python
from statsmodels.tsa.ar_model import AR
from random import random

data = [x + random() for x in range(1, 100)]

model = AR(data)
model_fit = model.fit()

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```




# ARIMA

## ARIMA($p$, $d$, $q$) 模型

Autoregressive Integrated Moving Average (ARIMA),
差分自回归移动平均模型, 是差分后的时间序列和残差误差的线性函数.

差分运算具有强大的确定性信息提取能力, 许多非平稳序列差分后会显示出平稳序列的性质, 称这个非平稳序列为差分平稳序列, 对差分平稳序列可以使用
ARIMA(autoregression integrated moving average, 求和自回归移动平均)
模型进行拟合.

ARIMA 模型的实质就是差分运算和 ARMA
模型的组合, 说明任何非平稳序列如果能通过适当阶数的差分实现差分后平稳, 就可以对差分后序列进行
ARMA 模型拟合, 而 ARMA 模型的分析方法非常成熟.



## ARIMA($p$, $d$, $q$) 模型结构

.. math::

   \left\{
   \begin{array}{**lr**}
   \Phi(B)\Delta^{d}x_{t} = \Theta(B)\epsilon_{t}& \\
   E(\epsilon_{t}) =0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{s}\epsilon_{t}) = 0, s \neq t& \\
   E(x_{s}\epsilon_{t}) = 0, \forall s < t&
   \end{array}
   \right.

其中: 

-  ${\epsilon_{t}}$ 为零均值白噪声序列
-  $\Delta^{d} = (1-B)^{d}$
-  $\Phi(B) = 1-\sum_{i=1}^{p}\phi_{i}B^{i}$ 为平稳可逆
   ARMA($p$, $q$) 模型的自回归系数多项式
-  $\Theta(B) = 1 + \sum_{i=1}^{q}\theta_{i}B^{i}$ 为平稳可逆
   ARMA($p$, $q$) 模型的移动平滑系数多项式

ARIMA 之所以叫 **求和自回归移动平均** 是因为: $d$
阶差分后的序列可以表示为下面的表示形式, 即差分后序列等于原序列的若干序列值的加权和, 而对它又可以拟合
ARMA 模型: 

$\Delta^{d}x_{t} = \sum_{i=0}^{d}(-1)C_{d}^{i}x_{t-i}, 其中: C_{d}^{i} = \frac{d!}{i!(d-i)!}$

ARIMA 模型的另一种形式: 

$\Delta^{d}x_{t} = \frac{\Theta(B)}{\Phi(B)}\epsilon_{t}$

其中: 

-  当 $d=0$ 时 ARIMA($p$, $0$, $q$) 模型就是
   ARMA($p$, $q$) 模型
-  当 $p=0$ 时, ARIMA($0$, $d$, $q$)
   模型可以简记为 IMA($d$, $q$) 模型
-  当 $q=0$ 时, ARIMA($p$, $d$, $0$)
   模型可以简记为 ARI($p$, $d$) 模型
-  当 $d=1, p=q=0$ 时, ARIMA($0$, $1$, $0$)
   模型为 随机游走 (random walk) 模型:



\left\{
\begin{array}{**lr**}
x_{t} = x_{t-1} + \epsilon_{t}& \\
E(\epsilon_{t}) =0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{s}\epsilon_{t}) = 0, s \neq t& \\
E(x_{s}\epsilon_{t}) = 0, \forall s < t&
\end{array}
\right.



## ARIMA($p$, $d$, $q$) 模型的统计性质

1. 平稳性

2. 方差齐性



## ARIMA($p$, $d$, $q$) 模型建模

1. 获得时间序列观察值
2. 平稳性检验
   -  不平稳: 差分运算 => 平稳性检验
   -  平稳: 下一步
3. 白噪声检验
   -  不通过: 拟合 ARMA 模型 => 白噪声检验
   -  通过: 分析结束

## ARIMA($p$, $d$, $q$) 模型应用

```python
from statsmodels.tsa.arima_model import ARIMA
from random import random

data = [x + random() for x in range(1, 100)]

model = ARIMA(data, order = (1, 1, 1))
model_fit = model.fit(disp = True)

y_hat = model_fit.predict(len(data), len(data), typ = "levels")
print(y_hat)
```




# ARIMA

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


## ADFuller 平稳性检验

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


## ACF 自相关函数, PACF 偏自相关函数

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

## ARIMA

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




# ARMA



## ARMA($p$, $q$) 模型

ARMA($p$,
$q$), 自回归移动平均模型, 是时间序列和残差误差的线性函数, 是
AR($p$) 和 MA($q$) 模型的组合, 该模型适用于无趋势
(trend) 和季节性 (seasonal) 因素的单变量时间序列


## ARMA($p$, $q$) 模型结构

\left\{
\begin{array}{**lr**}
x_{t}=\phi_{0} + \phi_{1}x_{t-1} + \phi_{2}x_{t-2} + \cdots + \phi_{p}x_{t-p} + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + \cdots + \theta_{q}\epsilon_{t-q} & \\
\phi_{p} \neq 0, \theta_{q} \neq 0& \\
E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t & \\
E(x_{s}\epsilon_{t}) = 0, \forall s < t & 
\end{array}
\right.

ARMA 模型的另一种形式: 

$(1-\sum_{i=1}^{p}\phi_{i}B^{i})x_{t} = (1 - \sum_{i=1}^{q}\theta_{i}B^{i})\epsilon_{t}$

$\Phi(B)x_{t} = \Theta(B)\epsilon_{t}$

-  当 $q = 0$ 时, ARMA($p$, $q$) 模型就退化成了
   AR($p$) 模型.

-  当 $p = 0$ 时, ARMA($p$, $q$) 模型就退化成了
   MA($q$) 模型.

-  所以 AR($p$) 和 MA($q$) 实际上是 ARMA($p$,
   $p$) 模型的特例, 它们统称为 ARMA 模型. 而 ARMA($p$,
   $p$) 模型的统计性质也正是 AR($p$) 模型和
   MA($p$) 模型统计性质的有机结合.

## ARMA($p$, $q$) 模型的统计性质

-  均值
-  自协方差函数
-  自相关系数

## ARMA($p$, $q$) 模型的应用

```python
from statsmodels.tsa.arima_model import ARMA
from random import random

data = [random.() for x in range(1, 100)]

model = ARMA(data, order = (2, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```


# HWES 模型




# MA

## MA($q$) 模型

MA($p$), $q$ 阶移动平均模型, 残差误差(residual erros)
的线性函数, 与计算时间序列的移动平均不同, 该模型适用于无趋势 (trend)
和季节性 (seasonal) 因素的单变量时间序列

## MA($q$) 模型结构:

\left\{
\begin{array}{**lr**}
x_{t}=\mu + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + \cdots + \theta_{q}\epsilon_{t-q}& \\
\theta_{q} \neq 0 & \\
E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t &
\end{array}
\right.

其中: 

-  $\epsilon_{t}$ 是白噪声序列
-  $\mu$ 是常数

## MA($q$) 模型的统计性质:

-  常数均值
-  常数方差
-  自协方差函数只与滞后阶数相关, 且 $q$ 阶截尾
-  自相关系数 $q$ 阶截尾


## MA($q$) 模型应用

```python
from statsmodesl.tsa.arima_model import ARMA
from random import random

data = [x + random() for x in range(1, 100)]

model = ARMA(data, order = (0, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```


# SARIMA

## SARIMA() 模型

该模型适用于含有趋势 (trend) 或季节性 (seasonal) 因素的单变量时间序列


```python
from statsmodels.tsa.statspace.sarima import SARIMAX
from random import random

data = [x + random() for x in range(1, 100)]

model = SARIMAX(data, order = (1, 1, 1), seasonal_order = (1, 1, 1, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

# SARIMAX

# SES

# VAR


# VARMA

# VARMAX