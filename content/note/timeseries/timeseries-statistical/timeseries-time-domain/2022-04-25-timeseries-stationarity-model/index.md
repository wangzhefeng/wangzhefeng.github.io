---
title: 平稳时间序列分析
author: 王哲峰
date: '2022-04-25'
slug: timeseries-stationarity
categories:
  - timeseries
tags:
  - machinelearning
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

- [平稳时间序列分析介绍](#平稳时间序列分析介绍)
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
- [参考](#参考)
</p></details><p></p>

# 平稳时间序列分析介绍

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

# AR 模型

## AR(`$p$`) 模型

AR(`$p$`), `$p$` 阶自回归模型, 该模型适用于无趋势(trend)和季节性(seasonal)因素的单变量时间序列

## AR(`$p$`) 模型结构

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

## AR(`$p$`) 模型的统计性质

- 均值
- 方差
- 自协方差函数
- 自相关系数
- 偏自相关系数

## AR(`$p$`) 模型应用

```python
from statsmodels.tsa.ar_model import AR
from random import random

data = [x + random() for x in range(1, 100)]

model = AR(data)
model_fit = model.fit()

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

## 参考资料

* [Autoregressive models](https://otexts.com/fpp2/AR.html)


# MA 模型

## MA(`$q$`) 模型

MA(`$p$`), `$q$` 阶移动平均模型, 残差误差(residual erros)的线性函数, 
与计算时间序列的移动平均不同, 该模型适用于无趋势(trend)和季节性(seasonal)因素的单变量时间序列

## MA(`$q$`) 模型结构

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

## MA(`$q$`) 模型的统计性质

- 常数均值
- 常数方差
- 自协方差函数只与滞后阶数相关, 且 `$q$` 阶截尾
- 自相关系数 `$q$` 阶截尾

## MA(`$q$`) 模型应用

```python
from statsmodesl.tsa.arima_model import ARMA
from random import random

data = [x + random() for x in range(1, 100)]

model = ARMA(data, order = (0, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

# ARMA 模型

## ARMA(`$p$`, `$q$`) 模型

ARMA(`$p$`, `$q$`), 自回归移动平均模型, 是时间序列和残差误差的线性函数, 
是 AR(`$p$`) 和 MA(`$q$`) 模型的组合, 
该模型适用于无趋势(trend)和季节性(seasonal)因素的单变量时间序列

## ARMA(`$p$`, `$q$`) 模型结构

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

## ARMA(`$p$`, `$q$`) 模型的统计性质

- 均值
- 自协方差函数
- 自相关系数

## ARMA(`$p$`, `$q$`) 模型应用

```python
from statsmodels.tsa.arima_model import ARMA
from random import random

data = [random.() for x in range(1, 100)]

model = ARMA(data, order = (2, 1))
model_fit = model.fit(disp = False)

y_hat = model_fit.predict(len(data), len(data))
print(y_hat)
```

# 参考

* [时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484974&idx=1&sn=d841c644fd9289ad5ec8c52a443463a5&chksm=cecef3dbf9b97acd8a9ededc069851afc00db422cb9be4d155cb2c2a9614b2ee2050dc7ab4d7&scene=21#wechat_redirect)
* [时间序列统计分析](https://mp.weixin.qq.com/s/INZgM6hLSEpboaNhS22CaA)

