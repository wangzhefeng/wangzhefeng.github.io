---
title: Lib prophet
author: 王哲峰
date: '2022-04-30'
slug: timeseries-lib-prophet
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

- [Prophet 简介](#prophet-简介)
- [Prophet 数据的输入和输出](#prophet-数据的输入和输出)
  - [时间序列数据格式](#时间序列数据格式)
  - [Prophet 时间序列格式](#prophet-时间序列格式)
- [Prophet 算法原理](#prophet-算法原理)
  - [时间序列分解](#时间序列分解)
  - [趋势项模型 g(t)](#趋势项模型-gt)
    - [逻辑回归函数 Logistic Function](#逻辑回归函数-logistic-function)
    - [分段线性函数 Piecewise Linear Function](#分段线性函数-piecewise-linear-function)
    - [变点的选择 Changepoint Selection](#变点的选择-changepoint-selection)
    - [对未来的预估 Trend Forecast Uncertainty](#对未来的预估-trend-forecast-uncertainty)
  - [季节性趋势 s(t)](#季节性趋势-st)
  - [节假日效应 h(t)](#节假日效应-ht)
  - [模型拟合 Model Fitting](#模型拟合-model-fitting)
- [Prophet 安装](#prophet-安装)
  - [R 中的安装](#r-中的安装)
  - [Python 中的安装](#python-中的安装)
- [Prophet 实际使用](#prophet-实际使用)
  - [Prophet 简单使用](#prophet-简单使用)
  - [Prophet 的参数设置](#prophet-的参数设置)
    - [增长函数](#增长函数)
    - [变点](#变点)
    - [周期性](#周期性)
    - [节假日](#节假日)
  - [Prophet 使用帮助](#prophet-使用帮助)
  - [Quick Start](#quick-start)
  - [饱和预测(Staturating Forecast)](#饱和预测staturating-forecast)
    - [Forecasting Growth](#forecasting-growth)
    - [Staturating Minimum](#staturating-minimum)
  - [Seasonality, Holiday Effects, And Regressors](#seasonality-holiday-effects-and-regressors)
    - [假期和特殊事件建模](#假期和特殊事件建模)
    - [指定内置的国家/地区假期(Build-in Country Holiday)](#指定内置的国家地区假期build-in-country-holiday)
    - [季节性的傅里叶变换(Fourier Order for Seasonalities)](#季节性的傅里叶变换fourier-order-for-seasonalities)
    - [指定自定义季节性](#指定自定义季节性)
- [参考](#参考)
</p></details><p></p>

# Prophet 简介

Facebook 开源了一个时间序列预测的算法, 叫做 **fbprophet**, 
官方网址与基本介绍来自于以下几个网站:

1. Github: https://github.com/facebook/prophet
2. 官方网址: https://facebook.github.io/prophet/
3. 论文地址: https://peerj.com/preprints/3190/

从官网的介绍来看, Facebook 所提供的 prophet 算法不仅可以处理时间序列存在一些异常值的情况, 
也可以处理部分缺失值的情形, 还能够几乎全自动地预测时间序列未来的走势.

从论文上的描述来看, 这个 prophet 算法是基于时间序列分解和机器学习的拟合来做的, 
其中在拟合模型的时候使用了 `pyStan` 这个开源工具, 因此能够在较快的时间内得到需要预测的结果

除此之外, 为了方便统计学家, 机器学习从业者等人群的使用, prophet 同时提供了 R 语言和 Python 语言的接口. 

从整体的介绍来看, 如果是一般的商业分析或者数据分析的需求, 都可以尝试使用这个开源算法来预测未来时间序列的走势. 

# Prophet 数据的输入和输出

首先让我们来看一个常见的时间序列场景, 黑色表示原始的时间序列离散点, 深蓝色的线表示使用
时间序列来拟合所得到的取值, 而浅蓝色的线表示时间序列的一个置信区间, 也就是所谓的合理的
上界和下界. prophet 所做的事情就是: 

1. 输入已知的时间序列的时间戳和相应的值
2. 输入需要预测的时间序列的长度
3. 输出未来的时间序列走势
4. 输出结果可以提供必要的统计指标, 包括拟合曲线, 上界和下界等

## 时间序列数据格式

就一般情况而言, 时间序列的离线存储格式为时间戳和值这种格式, 更多的话可以提供时间序列的 ID, 
标签等内容. 因此, 离线存储的时间序列通常都是以下的形式. 

|date                |category |value |label    |
|--------------------|---------|------|---------|
|2017-10-20 22:00:00 |  id1    |10    | "unknow"|
|2017-10-20 22:01:00 |  id1    |9     |      "1"|
|2017-10-20 22:02:00 |  id1    |0     |      "0"|

其中:

* `date` 指的是具体的时间戳
* `category` 指的是某条特定的时间序列 id
* `value` 指的是在 date 下这个 category 时间序列的取值
* `label` 指的是人工标记的标签
    - "0" 表示异常
    - "1" 表示正常
    - "unknown" 表示没有标记或者人工判断不清

## Prophet 时间序列格式

Prophet 所需要的时间序列也是这种格式的, 根据官网的描述, 只要用 `csv` 文件存储两列即可:

* 第一列的名字是 `ds`
    - 表示时间序列的时间戳    
* 第二列的名称是 `y`    
    - 表示时间序列的取值

通过 `prophet` 的计算, 可以计算出如下的数值: 

* `yhat`: 表示时间序列的预测值
* `yhat_lower`: 表示时间序列预测值的下界
* `yhat_upper`: 表示时间序列预测值的上界

两份表格如下面的两幅图表示:

* 输入数据格式

|    | ds                 | y    |
| ---|--------------------|------|
| 0  |2017-10-20 22:00:00 |  9.5 |
| 1  |2017-10-20 22:01:00 |  8.5 |
| 2  |2017-10-20 22:02:00 |  8.1 |
| 3  |2017-10-20 22:02:00 |  8.0 |
| 4  |2017-10-20 22:02:00 |  7.8 |

* 输出数据格式

|      | ds                 |  yhat | yhat_lower |  yhat_upper |
| -----|--------------------|-------|-----------|--------------|
| 3265 | 2017-10-20 22:00:00| 8.19  | 7.48      |       8.96   |
| 3266 | 2017-10-20 22:01:00| 8.52  | 7.79      |       9.26   |
| 3267 | 2017-10-20 22:02:00| 8.31  | 7.55      |       9.04   |
| 3268 | 2017-10-20 22:02:00| 8.14  | 7.42      |       8.86   |
| 3269 | 2017-10-20 22:02:00| 8.15  | 7.39      |       8.88   |

# Prophet 算法原理

## 时间序列分解

在时间序列分析中, 常见的分析方法叫做时间序列的分解(Decomposition of Time Series), 
时间序列分解的原理是将时间序列 `$y_{t}$` 分为三个部分: 

* 季节性项 `$S_{t}$`
    - 在固定的时间段内重复的模式
* 趋势项 `$T_{t}$`
    - 指标的基本趋势
* 随机噪声项(剩余项) `$R_{t}$`
    - 季节性和趋势项被删除后原始时间序列的残差

根据季节性项 `$S_{t}$` 的变化, 分解方法可分为两种: 

* 加法分解
    - `$y_{t} = S_t + T_t + R_t$`
* 乘法分解
    - `$y_{t} = S_t \times T_t \times R_t$`, 等价于 `$\ln y_{t} = \ln S_t + \ln T_t + \ln R_t$`
    - 由于上面的等价关系, 所以, 有的时候在预测模型的时候, 会先取对数, 然后再进行时间序列的分解, 就能得到乘法的形式. 
      在 Prophet 算法中, 作者们基于这种方法进行了必要的改进和优化. 

一般来说, 在实际生活和生产环节中, 除了季节项、趋势项、剩余项之外, 通常还有节假日的效应. 
所以, 在 Prophet 算法里面, 作者同时考虑了以上四项, Prophet 算法就是通过拟合这几项, 
然后最后把它们累加起来就得到了时间序列的预测值. 也就是: 

`$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$`

其中: 

* `$g(t)$` 表示趋势项, 它表示时间序列在非周期上面的变化趋势
* `$s(t)$` 表示周期项, 或者称为季节项, 一般来说是以周或者年为单位
* `$h(t)$` 表示节假日项, 表示在当天是否存在节假日
* `$\epsilon_t$` 表示误差项或者称为剩余项

## 趋势项模型 g(t)

在 Prophet 算法里面, 趋势项有两个重要的函数：

* 一个是基于逻辑回归函数(logistic function)的
* 另一个是基于分段线性函数(piecewise linear function)

### 逻辑回归函数 Logistic Function

首先, 介绍一下基于逻辑回归的趋势项是怎么做的。
回顾逻辑回归函数，一般都会想起 Sigmoid 函数形式：

`$$\sigma(x) = \frac{1}{1+e^{-x}}$$`

它的导函数是 `$\sigma '(x) = \sigma(x)\cdot (1-\sigma(x))$`，
并且 `$\underset{x \rightarrow +\infty}{lim}\sigma(x) = 1$`，
`$\underset{x \rightarrow -\infty}{lim}\sigma(x) = 0$`

如果增加一些参数的话，逻辑回归就可以改写成：

`$$f(x) = \frac{C}{1+e^{-k(x-m)}}$$`

其中：

* `$C$` 称为曲线的最大渐近值
* `$k$` 表示曲线的增长率
* `$m$` 表示曲线的中点

当 `$C = 1$`，`$k = 1$`，`$m=0$` 时，恰好就是 Sigmoid 函数的形式。
从 Sigmoid 的函数表达式可以看出，它满足以下的微分方程：`$y'=y(1-y)$`。
那么，如果使用分离变量法来求解微分方程 `$y'=y(1-y)$` 就可以得到：

`$$\frac{y'}{y} + \frac{y'}{1-y}=1$$`

`$$ln\frac{y}{1-y}=1$$`

`$$y=\frac{1}{1+ke^{-x}}$$`

在现实情况下，函数 `$f(x)$` 的三个参数 `$C, k, m$` 不可能都是常数，而很有可能是随着时间的迁移而变化的，
因此，在 Prophet 里面，作者考虑把这三个参数全部换成了随着时间变化的函数，即：`$C=C(t)$`，`$k=k(t)$`，`$m=m(t)$`

除此之外，在现实的时间序列中，曲线的走势肯定不会一直保持不变，
在某些特定的时候或者有着某种潜在的周期曲线会发生变化，
这种时候，就有学者会去研究变点检测，也就是所谓变点检测(change point detection)。
例如下面的这幅图的 `$t^{*}_{1}$`，`$t^{*}_{2}$`  就是时间序列的两个变点

![img](images/change_point.jpeg)

> 在 Prophet 里面，是需要设置变点的位置的，而每一段的趋势和走势也是会根据变点的情况而改变的。
在程序里面有两种方法，一种是通过人工指定的方式指定变点的位置；另外一种是通过算法来自动选择。
在默认的函数里面，Prophet 会选择 `n_changepoints = 25` 个变点，然后设置变点的范围是前 `changepoint_range = 80%`，
也就是在时间序列的前 80% 的区间内会设置变点。通过 `forecaster.py` 里面的 `set_changepoints` 函数可以知道，
首先要看一些边界条件是否合理，例如时间序列的点数是否少于 `n_changepoints` 等内容；
其次如果边界条件符合，那变点的位置就是均匀分布的，这一点可以通过 `np.linspace` 这个函数看出来

下面假设已经设置了 `$S$` 个变点了，并且变点的位置是在时间戳 `$s_{j}, 1\leq j \leq S$` 上，
那么在这些时间戳上，就需要给出增长率的变化，也就是在时间戳 `$s_{j}$` 上发生的 change in rate。

可以假设有这样一个向量：`$\delta = (\delta_{1}, \ldots, \delta_{S}) \in \mathbb{R}^{S}$`，
其中 `$\delta_{j}$` 表示在时间戳 `$s_{j}$` 上的增长率的变化量，如果初始增长率使用 `$k$` 表示，
那么在时间戳 `$t$` 上的增长率就是 `$k+\sum_{j:t>s_{j}}\delta_{j}$`，
通过一个指示函数 `$a(t)\in \{0, 1\}^{S}$`：

`$$a_{j}(t)=\begin{cases}
1, if \text{ } t \geq s_{j}, \\
0, otherwise \end{cases}$$`

表示在时间戳 `$t$` 上面的增长率就是：

`$$k+a^{T}\delta$$`

一旦变化率 `$k$` 确定了，另一个参数 `$m$` 也要随之确定。

在这里需要把线段的边界处理好，因此通过数学计算可以得到：

`$$\gamma_{j}=\Bigg(s_{j} - m - \underset{l<j}{\sum}\gamma_{l}\Bigg) \cdot \Bigg(1 - \frac{k+\underset{l<j}{\sum}\delta_{l}}{k+\underset{l \leq j}{\sum}\delta_{l}}\Bigg)$$`

所以，分段的逻辑回归增长模型就是：

`$$\begin{align}
g(t) &= \frac{C(t)}{1+e^{-k(t)(t-m(t))}} \\
&= \frac{C(t)}{1+ e^{-\big(k + a(t)^{T} \delta\big) \cdot \big(t-(m+a(t)^{T}\gamma)\big)}}
\end{align}$$`

其中：

* `$a(t)=(a_{1}(t), \ldots, a_{S}(t))^{T}$`
* `$\delta=(\delta_{1}, \ldots, \delta_{S})^{T}$`
* `$\gamma=(\gamma_{1}, \ldots, \gamma_{S})^{T}$`

在逻辑回归函数里面，有一个参数是需要提前设置的，那就是 Capacity，也就是所谓的  `$C(t)$`，
在使用 Prophet 的 `growth = 'logistic'` 的时候，需要提前设置好 `$C(t)$` 的取值才行

### 分段线性函数 Piecewise Linear Function

再次，我们来介绍一下基于分段线性函数的趋势项是怎么做的。
众所周知，线性函数指的是 `$y=kx+b$`，而分段线性函数指的是在每一个子区间上，
函数都是线性函数，但是在整段区间上，函数并不完全是线性的

因此，基于分段线性函数的模型形如：

`$$g(t) = \big(k + a(t)\delta\big) \cdot t + \big(m + a(t)^{T}\gamma\big)$$`

其中：

* `$k$` 表示增长率(growth rate)
* `$\delta$` 表示增长率的变化量
* `$m$` 表示 offset parameter

逻辑回归函数与分段线性函数最大的区别就是 `$\gamma$` 的设置不一样，在分段线性函数中，

`$$\gamma=(\gamma_{1}, \ldots, \gamma_{S}), \gamma_{j} = -s_{j}\delta_{j}$$`

在 Prophet 的源代码中，`forecast.py` 这个函数里面包含了最关键的步骤，
其中 `piecewise_logistic` 函数表示了前面所说的基于逻辑回归的增长函数，
它的输入包含了 `cap` 这个指标，因此需要用户事先指定 capacity。
而在 `piecewise_linear` 这个函数中，是不需要 capacity 这个指标的，
因此 `m = Prophet()` 这个函数默认的使用 `growth = 'linear'` 这个增长函数，
也可以写作 `m = Prophet(growth = 'linear')`；
如果想用 `growth = 'logistic'`，就要这样写

```python
m = Prophet(growth = "logistic")
df["cap"] = 6
m.fit(df)
future = m.make_future_dataframe(periods = prediction_length, freq = "min")
future["cap"] = 6
```

### 变点的选择 Changepoint Selection

在介绍变点之前，先要介绍一下 Laplace 分布，它的概率密度函数为：

`$$f(x|\mu, b) = \frac{e^{-\frac{|x-\mu|}{b}}}{2b}$$`

其中：

* `$\mu$` 表示位置参数
* `$b>0$` 表示尺度参数

在 Prophet 算法中，是需要给出变点的位置、个数，以及增长的变化率的。
因此，有三个比较重要的指标：

* `changepoint_range`
    - `changepoint_range` 指的是百分比，需要在前 `changepoint_range` 那么长的时间序列中设置变点，
      在默认的函数中是 `changepoint_range = 0.8`
* `n_changepoint`
    - `n_changepoint` 表示变点的个数，在默认的函数中是 `n_changepoint = 25`
* `changepoint_prior_scale`
    - `changepoint_prior_scale` 表示变点增长率的分布情况，
      在论文中 `$\delta_{j} \sim Laplace(0, \tau)$`， 这里的 `$\tau$` 就是 `change_point_scale`

在整个开源框架里面，在默认的场景下，变点的选择是基于时间序列的前 80% 的历史数据，
然后通过等分的方法找到 25 个变点（change points），
而变点的增长率是满足 Laplace 分布 `$\delta_{j} \sim Laplace(0, 0.05)$` 的。
因此，当 `$\tau$` 趋近于零的时候，`$\delta_{j}$` 也是趋向于零的，
此时的增长函数将变成全段的逻辑回归函数或者线性函数。这一点从 `$g(t)$` 的定义可以轻易地看出

### 对未来的预估 Trend Forecast Uncertainty

## 季节性趋势 s(t)

## 节假日效应 h(t)

Holidays and events

## 模型拟合 Model Fitting

时间序列已经可以通过增长项, 季节项, 节假日项来构建了:

`$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$`

下一步我们只需要拟合函数就可以了, 在 Prophet 里面, 
作者使用了 `pyStan` 这个开源工具中的 `L-BFGS` 方法来进行函数的拟合. 
具体可以参考 `forecast.py` 里面的 `stan_init` 函数 



# Prophet 安装

## R 中的安装

```r
# install prophet on CRAN
install.packages("prophet", type = "source")
```

## Python 中的安装

(1) 安装 `pystan`:

- Windows
   - compiler
      - python
      - C++ compiler
      - PyStan
- Linux
   - compilers:
      - gcc(gcc64 on Red Hat)
      - g++(gcc64-c++ on Red Hat)
      - build-essential
   - Python development tools
      - python-dev
      - python3-dev
- Anaconda
   - `conda install gcc`

```bash
# Windows
$ pip install pystan
# or
$ conda install pystan -c conda-forge
```

(2) 安装 `prophet`:

```bash
# install on PyPI
$ pip install fbprophet

# install using conda-forge
$ conda install -c conda-forge fbprophet
```

(3) 安装交互式图形库

```bash
$ pip install plotly
```

# Prophet 实际使用

## Prophet 简单使用

1.因为 Prophet 所需要的两列名称是 ‘ds’ 和 ‘y’, 其中, ’ds’ 表示时间戳, 
’y’ 表示时间序列的值, 因此通常来说都需要修改 pd.dataframe 的列名字. 
如果原来的两列名字是 ‘timestamp’ 和 ‘value’ 的话, 只需要这样写: 

```python
df = df.rename(columns = {
    "timestamp": "ds",
    "value": y
})
```

2.如果 `timestamp` 是使用 unixtime 来记录的, 需要修改成 `YYYY-MM-DD hh:mm:ss` 的形式: 

```python
df["ds"] = pd.to_datetime(df["ds"], unit = "s")
```

3.在一般情况下, 时间序列需要进行归一化的操作, 而 `pd.DataFrame` 的归一化操作也十分简单: 

```python
df["y"] = (df["y"] - df["y"].mean()) / (df["y"].std())
```

4.然后就可以初始化模型, 然后拟合模型, 并且进行时间序列的预测了

```python
# 初始化模型
model = Prophet()

# 拟合模型
model.fit(X_train)
```

```python
# 计算预测值: periods 表示需要预测的点数, freq 表示时间序列的频率. 
future = model.make_future_dataframe(periods = 30, freq = "min")
future.tail()
forecast = model.predict(future)
# 画出预测图
model.plot(forecast)
# 画出时间序列的分量
model.plot_components(forecast)
# 
x1 = forecast['ds']
y1 = forecast['yhat']
y2 = forecast['yhat_lower']
y3 = forecast['yhat_upper']
plt.plot(x1, y1)
plt.plot(x1, y2)
plt.plot(x1, y3)
plt.show()
# rophet 预测的结果都放在了变量 forecast 里面
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

```python
X_test_forecast = model.predict(X_test)
```

## Prophet 的参数设置

在 Prophet 中, 用户一般可以设置以下四种参数: 

* Capacity: 在增量函数是逻辑回归函数的时候, 需要设置的容量值
* Change Points: 可以通过 `n_changepoints` 和 `changepoint_range` 来进行等距的变点设置, 
  也可以通过人工设置的方式来指定时间序列的变点
* 季节性和节假日: 可以根据实际的业务需求来指定相应的节假日
* 光滑参数: 
    - `$\tau$` = `changepoint_prior_scale` 可以用来控制趋势的灵活度
    - `$\sigma$` = `seasonality_prior_scale` 用来控制季节项的灵活度
    - `$v$` = `holidays_prior_scale` 用来控制节假日的灵活度

如果不想设置的话, 使用 Prophet 默认的参数即可


Prophet 的默认参数配置(源码 forecaster.py): 

```python
def __init__(
    self,
    growth='linear',
    changepoints=None,
    n_changepoints=25, 
    changepoint_range=0.8,
    yearly_seasonality='auto',
    weekly_seasonality='auto',
    daily_seasonality='auto',
    holidays=None,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.80,
    uncertainty_samples=1000):
    pass
```

### 增长函数

在 Prophet 里面, 有两个增长函数, 分别是分段线性函数(linear)和逻辑回归函数(logistic). 
而 `m = Prophet()` 默认使用的是分段线性函数(linear), 并且如果要是用逻辑回归函数的时候, 
需要设置 `capacity` 的值, i.e. `df[‘cap’] = 100`, 否则会出错. 

```python
m = Prophet()
m = Prophet(growth = 'linear')
m = Prophet(growth = 'logistic')
```

### 变点

在 Prophet 里面, 变点默认的选择方法是前 80% 的点中等距选择 25 个点作为变点, 
也可以通过以下方法来自行设置变点, 甚至可以人为设置某些点. 

```python
m = Prophet(n_changepoints=25)
m = Prophet(changepoint_range=0.8)
m = Prophet(changepoint_prior_scale=0.05)
m = Prophet(changepoints=['2014-01-01'])
```

变点的作图:

```python
from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

### 周期性

可以在 Prophet 中设置周期性, 无论是按月(month)还是周(week)

- Monthly

```python
# Monthly
model = Prophet(weekly_seasonality = False)
model.add_seasonality(name = "monthly", period = 30.5, fourier_order = 5)
```

- Weekly

```python
# Weekly
model = Prophet(weekly_seasonality = True)
model.add_seasonality(name = "weekly", period = 7, fourier_order = 3, prior_scale = 0.1)
```

### 节假日

可以设置某些天是节假日, 并且设置它的前后影响范围, 也就是 `lower_window` 和 `upper_window`

```python
# build holidays datasets
playoffs = pd.DataFrame({
    "holiday": "playoff",
    "ds": pd.to_datetime([
        '2008-01-13', '2009-01-03', '2010-01-16',
        '2010-01-24', '2010-02-07', '2011-01-08',
        '2013-01-12', '2014-01-12', '2014-01-19',
        '2014-02-02', '2015-01-11', '2016-01-17',
        '2016-01-24', '2016-02-07'
    ]),
    "lower_window": 0,
    "upper_window": 1,
})

superbowls = pd.DataFrame({
    "holiday": "superbowl",
    "ds": pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
    "lower_window": 0,
    "upper_window": 1,
})
holidays = pd.concat([playoffs, superbowl])

# model 
model = Prophet(holidays = holidays, holidays_prior_scale = 10.0)
```

对于商业分析等领域的时间序列, Prophet 可以进行很好的拟合和预测, 但是对于一些周期性或者趋势性不是很强的时间序列, 
用 Prophet 可能就不合适了. 但是, Prophet 提供了一种时序预测的方法, 在用户不是很懂时间序列的前提下都可以使用这
个工具得到一个能接受的结果. 具体是否用 Prophet 则需要根据具体的时间序列来确定. 


## Prophet 使用帮助

```python
from fbprophet import Prophet
help(Prophet)
help(Prophet.fit)
```

## Quick Start

**Example:**

- Question:
   - [Wikipedia page for Peyton Manning](https://en.wikipedia.org/wiki/Peyton_Manning)
- [data](https://github.com/facebook/prophet/blob/master/examples/example_wp_log_peyton_manning.csv)

导入工具库: 

```python
import pandas as pd
from fbprophet import Prophet
```

数据读入: 

```python
df = pd.read_csv("./data/example_wp_log_peyton_manning.csv")
df.head()
```

建立模型: 

```python
m = Prophet()
m.fit(df)
```

模型预测: 

```python
future = m.make_future_dataframe(periods = 365)
future.head()
future.tail()
```

```python
forecast = m.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

forecast 可视化: 

```python
fig1 = m.plot(forecast)
```

forecast 组件可视化: 

- trend
- yearly seasonlity
- weekly seasonlity
- holidays

```python
fig2 = m.plot_components(forecast)
```

```python
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)
py.iplot(fig)
```

## 饱和预测(Staturating Forecast)

### Forecasting Growth

- By default, Prophet uses a **linear model** for its forecast. When
   **forecasting growth**, there is usually some maximum achievable
   point: total market size, total population size, etc. This is called
   the **carrying capacity**, and the forecast should **saturate** at
   this point.

- Prophet allows you to make forecasts using a **logistic growth trend
   model**, with a specified carrying capacity.

```python
import pandas as pd

# ============================================
# data
# ============================================
df = pd.read_csv("./data/example_wp_log_R.csv")
# 设定一个 carrying capacity,根据数据或专家经验说明市场规模
df["cap"] = 8.5

# ============================================
# model 1 - staturating maximum
# ============================================
m = Prophet(growth = "logistic")
m.fit(df)

future_df = m.make_future_dataframe(periods = 1826)
future_df["cap"] = 8.5
forecast = m.predict(future_df)
fig = m.plot(forecast)
```

### Staturating Minimum

- Staturating Capacity
   - staturating maximum
   - staturating minimum(maximum必须设定)

```python
# data
df = pd.read_csv("./data/example_wp_log_R.csv")
df["y"] = 10 - df["y"]
df["cap"] = 6
df["floor"] = 1.5
future["cap"] = 6
future["floor"] = 1.5

# ============================================
# model 1 - staturating maximum and minimum
# ============================================
m = Prophet(growth = "logistic")
m.fit(df)

future_df = m.make_future_dataframe(periods = 1826)
forecast = m.predict(future_df)
fig = m.plot(forecast)
```

## Seasonality, Holiday Effects, And Regressors

### 假期和特殊事件建模

如果一个需要建模的时间序列中存在假期或其他重要的事件, 
则必须为这些特殊事件创建单独的 DataFrame. 
Dataframe 的格式如下: 

* DataFrame
   - holiday
   - ds
   - lower_window
   - upper_window
   - prior_scale

1.创建一个 DataFrame, 其中包括了 Peyton Manning 所有季后赛出场的日期: 

```python
# data
df = pd.read_csv("./data/example_wp_log_peyton_manning.csv")

# 季后赛
playoffs = pd.DataFrame({
   "holiday": "playoff",
   "ds": pd.to_datetime(["2018-01-13", "2019-01-03", "2010-01-16",
                     "2010-01-24", "2010-02-07", "2011-01-08",
                     "2013-01-12", "2014-01-12", "2014-01-19",
                     "2014-02-02", "2015-01-11", "2016-01-17",
                     "2016-01-24", "2016-02-07"]),
   "lower_window": 0,
   "upper_window": 1
})

# 超级碗比赛
superbowls = pd.DataFrame({
   "holiday": "superbowl",
   "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
   "lower_window": 0,
   "upper_window": 1
})
# 所有的特殊比赛
holiday = pd.concat([playoffs, superbowls])
```

```python
   print(holiday)
```

![img](images/holiday_df.png)

2.通过使用 `holidays` 参数传递节假日影响因素

```python
from fbprophet import Prophet

m = Prophet(holidays = holidays)
m.fit(df)
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)
```

```python
forecast[(forecast["playoff"] + forecast["superbowl"]).abs() > 0][["ds", "playoff", "superbowl"]][-10:]
```

时间序列组件可视化: 

```python
prophet_plot_components(m, forecast)
```

```python
plot_forecast_components(m, forecast, "superbowl")
```


### 指定内置的国家/地区假期(Build-in Country Holiday)

- 通过在 `fbprophet.Prophet()` 中的 `holidays` 参数指定任何假期

- 通过使用 `.add_country_holidays()` 方法使用 `fbprophet.Prophet()`
   内置的 `country_name` 参数特定该国家/地方的主要假期

```python
# 模型-指定假期
m = Prophet(holidays = holidays)
m.add_country_holidays(country_name = "US")
m.fit(df)

# 所有指定的假期
m.train_holiday_names

forecast = m.predict(future)
fig = m.plot_components(forecast)
```

### 季节性的傅里叶变换(Fourier Order for Seasonalities)

使用 **偏傅里叶和(Partial Fourier Sum)** 估计季节性, 逼近非定期信号

- The number of terms in the partial sum(the order) is a parameter that determines how quickly the seasonlity can change
- [论文](https://peerj.com/preprints/3190/)
- [Wiki](https://en.wikipedia.org/wiki/Fourier_series#/media/File:Fourier_Series.svg)
- [Wiki图](https://en.wikipedia.org/wiki/Fourier_series#/media/File:Fourier_Series.svg)

```python
from fbprophet.plot import plot_yearly

m = Prophet().fit(df)
a = plot_yearly(m)
```

通常季节性 `yearly_seasonality` 的默认值是比较合适的, 但是当季节性需要适应更高频率的变化时, 可以增加频率. 
但是增加频率后的序列不会太平滑. 即可以在实例化模型时, 可以为每个内置季节性指定傅里叶级别

```python
from fbprophet.plot import plot_yearly

m = Prophet(yearly_seasonality = 20).fit(df)
a = plot_yearly(m)
```

增加傅里叶项的数量可以使季节性适应更快的变化周期, 但也可能导致过拟合: N
傅里叶项对应于用于建模循环的 2N 变量. 

### 指定自定义季节性

如果时间序列长度超过两个周期, Prophet
将默认自适应每周、每年的季节性. 它还适合时间序列的每日季节性, 可以通过
`add_seasonality` 方法添加其他季节性, 比如: 每月、每季度、每小时

```python
m = Prophet(weekly_seasonality = False)
m.add_seasonality(name = "monthly", period = 30.5, fourier_order = 5)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```

# 参考

* [R-tutorial](https://otexts.com/fpp2/ts-objects.html)
* [Blog](https://zr9558.com/2018/11/30/timeseriespredictionfbprophet/)
