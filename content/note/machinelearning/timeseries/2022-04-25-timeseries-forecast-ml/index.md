---
title: 时间序列预测-机器学习
author: 王哲峰
date: '2022-04-25'
slug: timeseries-forecast_ml
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

- [时间序列回归](#时间序列回归)
    - [单步预测](#单步预测)
    - [多步预测](#多步预测)
    - [多变量预测](#多变量预测)
    - [递归预测 Recursive Forecasting](#递归预测-recursive-forecasting)
    - [直接预测 Direct Forecasting](#直接预测-direct-forecasting)
    - [堆叠预测 Stacking Forecasting](#堆叠预测-stacking-forecasting)
    - [修正预测 Rectified Forecasting](#修正预测-rectified-forecasting)
- [时间序列数据特征工程](#时间序列数据特征工程)
  - [创建 Date Time 特征](#创建-date-time-特征)
    - [常见特征](#常见特征)
  - [创建 Lagging 特征](#创建-lagging-特征)
    - [滑动窗口特征](#滑动窗口特征)
    - [滑动窗口统计量特征](#滑动窗口统计量特征)
    - [TODO](#todo)
  - [创建 Window 变量](#创建-window-变量)
- [参考文章](#参考文章)
</p></details><p></p>

# 时间序列回归

时序预测模型与回归预测模型不同，时序预测模型依赖于数值在时间上的先后顺序，是回归模型中的一部分。
简单来说，时间序列的回归分析需要分析历史数据，找到历史数据演化中的模式和特征，
其主要分为线性回归分析和非线性回归分析两种类型

回归分析多采用机器学习方法，我们首先需要明确机器学习(或深度学习)模型构建与验证的主体思路：

1. 分析数据构建数据特征，将数据转化为特征样本集合
2. 明确样本与标签，划分训练集与测试集
3. 比较不同模型在相同的训练集中的效果，或是相同模型的不同参数在同一个训练集中拟合的效果
4. 在验证样本集中验证模型的准确度，通过相关的结果评估公式选择表现最好同时没有过拟合的模型

近年来时间序列预测方法，多采用机器学习方式。机器学习的方法，主要是构建样本数据集，
采用“时间特征”到“样本值”的方式，通过有监督学习，学习特征与标签之前的关联关系，
从而实现时间序列预测。常用的场景有以下几种

### 单步预测

在时间序列预测中的标准做法是使用滞后的观测值 `$x_{t-1}$` 作为输入变量来预测当前的时间的观测值 `$x_{t}$`

### 多步预测

使用过去的观测序列 `$\{\ldots, x_{t-2}, x_{t-1}\}$` 预测未来的观测序列 `$\{x_{t}, x_{t+1}, \ldots\}$`

### 多变量预测

多元时间序列，即每个时间有多个观测值：

`$$\{X_{t} = (x_{t}^{a}, x_{t}^{b}, x_{t}^{c}, \ldots)\}_{t}^{T}$$`

这意味着通过不同的测量手段得到了多种观测值，并且希望预测其中的一个或几个值。
例如，可能有两组时间序列观测值 `$\{x_{t-1}^{a}, x_{t-2}^{a}, \ldots\}$`，
`$\{x_{t-1}^{b}, x_{t-2}^{b}, \ldots\}$`，希望分析这组多元时间序列来预测 `$x_{t}^{a}$` 

基于以上场景，许多监督学习的方法可以应用在时间序列的预测中，
在运用机器学习模型时，可以把时间序列模型当成一个回归问题来解决，
比如 svm/xgboost/逻辑回归/回归树/...

### 递归预测 Recursive Forecasting



### 直接预测 Direct Forecasting




### 堆叠预测 Stacking Forecasting



### 修正预测 Rectified Forecasting


# 时间序列数据特征工程

将时间序列数据转换为监督机器学习问题数据

* 将单变量时间序列数据转换为机器学习问题数据
* 将多变量时间序列数据转换为机器学习问题数据

特征工程的核心要点在于如何从历史的数据中抽取特征, 
这里介绍一些特征构建的经验: 

* 离散类时间特征: 年月日时分数, 周几, 一年中的第几天, 第几周, 一天中的哪个时间段等
* 判断类时间特征: 是否调休, 是否周末, 是否公共假期等
* 滑窗类时间聚合特征: 过去X天平均值, 过去X天方差, 过去X天最大值, 过去X小时四分位数, 过去X天偏态系数等
* 其他时序模型的预测值作为特征: ARIMA、SARIMA、指数平滑等
* 其他相关业务线数据的引入: 比如对于售后业务线, 引入售前业务线/预定业务线等数据, 帮忙进行售后业务线的预测

三种特征:

* Date Time Features
* Lag Features
* Window Features



## 创建 Date Time 特征

### 常见特征

* 一年中的月分
* 一个月中的日期
* 一天过去了几分钟
* 一天的时间
* 营业时间与否
* 周末与否
* 一年中的季节
* 一年中的业务季度
* 夏时制与否
* 公共假期与否
* 是否是闰年

```python
df = pd.DataFrame()
df["Date"] = [series.index[i] for i in range(len(series))]
df["month"] = [series.index[i].month for i in range(len(series))]
df["day"] = [series.index[i].day for i in range(len(series))]
df["temperature"] = [series[i] for i in range(len(series))]
print(df.head())
```

```
        Date  month  day  temperature
0 1981-01-01      1    1         20.7
1 1981-01-02      1    2         17.9
2 1981-01-03      1    3         18.8
3 1981-01-04      1    4         14.6
4 1981-01-05      1    5         15.8
```

## 创建 Lagging 特征

* pushed forward
* pulled back

```python
import pandas as pd 

def series_to_supervised(data, n_lag = 1, n_fut = 1, selLag = None, selFut = None, dropnan = True):
      """
      Converts a time series to a supervised learning data set by adding time-shifted prior and future period
      data as input or output (i.e., target result) columns for each period
      :param data:  a series of periodic attributes as a list or NumPy array
      :param n_lag: number of PRIOR periods to lag as input (X); generates: Xa(t-1), Xa(t-2); min= 0 --> nothing lagged
      :param n_fut: number of FUTURE periods to add as target output (y); generates Yout(t+1); min= 0 --> no future periods
      :param selLag:  only copy these specific PRIOR period attributes; default= None; EX: ['Xa', 'Xb' ]
      :param selFut:  only copy these specific FUTURE period attributes; default= None; EX: ['rslt', 'xx']
      :param dropnan: True= drop rows with NaN values; default= True
      :return: a Pandas DataFrame of time series data organized for supervised learning
      
      NOTES:
      (1) The current period's data is always included in the output.
      (2) A suffix is added to the original column names to indicate a relative time reference: e.g., (t) is the current
         period; (t-2) is from two periods in the past; (t+1) is from the next period
      (3) This is an extension of Jason Brownlee's series_to_supervised() function, customized for MFI use
      """
      n_vars = 1 if type(data) is list else data.shape[1]
      df = pd.DataFrame(data)
      origNames = df.columns
      cols, names = list(), list()
      # include all current period attributes
      cols.append(df.shift(0))
      names += [("%s" % origNames[j]) for j in range(n_vars)]
      # lag any past period attributes (t-n_lag, ..., t-1)
      n_lag = max(0, n_lag) # force valid number of lag periods
      # input sequence (t-n, ..., t-1)
      for i in range(n_lag, 0, -1):
         suffix = "(t-%d)" % i
         if (None == selLag):
            cols.append(df.shift(i))
            names += [("%s%s" % (origNames[j], suffix)) for j in range(n_vars)]
         else:
            for var in (selLag):
                  cols.append(df[var].shift(i))
                  names += [("%s%s" % (var, suffix))]
      # include future period attributes (t+1, ..., t+n_fut)
      n_fut = max(n_fut, 0)
      # forecast sequence (t, t+1, ..., t+n)
      for i in range(0, n_fut + 1):
         suffix = "(t+%d)" % i
         if (None == selFut):
            cols.append(df.shift(-i))
            names += [("%s%s" % (origNames[j], suffix)) for j in range(n_vars)]
         else:
            for var in (selFut):
                  cols.append(df[var].shift(-i))
                  names += [("%s%s" % (var, suffix))]
      # put it all together
      agg = pd.concat(cols, axis = 1)
      agg.columns = names
      # drop rows with NaN values
      if dropnan:
         agg.dropna(inplace = True)

      return agg
```

### 滑动窗口特征

```python
temps = pd.DataFrame(series.values)
df = pd.concat([
    temps.shift(3), 
    temps.shift(2), 
    temps.shift(1), 
    temps
], axis = 1)
df.columns = ["t-3", "t-2", "t-1", "t+1"]
df.dropna(inplace = True)

print(df.head())
```

``` 
    t-3   t-2   t-1   t+1
3  17.9  18.8  14.6  15.8
4  18.8  14.6  15.8  15.8
5  14.6  15.8  15.8  15.8
6  15.8  15.8  15.8  17.4
7  15.8  15.8  17.4  21.8
```

### 滑动窗口统计量特征

```python
temps = pd.DataFrame(series.values)

shifted = temps.shift(1)
window = shifted.rolling(window = 2)
means = window.mean()

df = pd.concat([mean, temps], axis = 1)
df.columns = ["mean(t-2,t-1)", "t+1"]

print(df.head())
```

```
   mean(t-2,t-1)   t+1
0            NaN  17.9
1            NaN  18.8
2          18.35  14.6
3          16.70  15.8
4          15.20  15.8
```

### TODO

```python
temps = pd.DataFrame(series.values)

width = 3
shifted = temps.shift(width - 1)
window = shifted.rolling(windon = width)

df = pd.concat([
    window.min(), 
    window.mean(), 
    window.max(), 
    temps
], axis = 1)
df.columns = ["min", "mean", "max", "t+1"]

print(df.head())
```

```
    min  mean   max   t+1
0   NaN   NaN   NaN  17.9
1   NaN   NaN   NaN  18.8
2   NaN   NaN   NaN  14.6
3   NaN   NaN   NaN  15.8
4  14.6  17.1  18.8  15.8
```

## 创建 Window 变量

```python
temps = pd.DataFrame(series.values)

window = temps.expanding()

df = pd.concat([
    window.min(), 
    window.mean(), 
    window.max(), 
    temps.shift(-1)
], axis = 1)
df.columns = ["min", "mean", "max", "t+1"]

print(df.head())
```

``` 
    min    mean   max   t+1
0  17.9  17.900  17.9  18.8
1  17.9  18.350  18.8  14.6
2  14.6  17.100  18.8  15.8
3  14.6  16.775  18.8  15.8
4  14.6  16.580  18.8  15.8
```

# 参考文章

* [Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
* [How to Convert a Time Series to a Supervised Learning Problem in Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
* [Machine Learning Strategies for Time Series Forecasting](https://link.springer.com/chapter/10.1007%2F978-3-642-36318-4_3)
* [slide](http://di.ulb.ac.be/map/gbonte/ftp/time_ser.pdf)
* [Machine Learning for Sequential Data: A Review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)
* [如何用Python将时间序列转换为监督学习问题](https://cloud.tencent.com/developer/article/1042809)
* [How To Resample and Interpolate Your Time Series Data With Python](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)
* [时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484974&idx=1&sn=d841c644fd9289ad5ec8c52a443463a5&chksm=cecef3dbf9b97acd8a9ededc069851afc00db422cb9be4d155cb2c2a9614b2ee2050dc7ab4d7&scene=21#wechat_redirect)

