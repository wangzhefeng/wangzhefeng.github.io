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
  - [常见线性回归模型](#常见线性回归模型)
  - [常见的非线性回归模型](#常见的非线性回归模型)
- [时间序列数据 vs 监督机器学习数据](#时间序列数据-vs-监督机器学习数据)
  - [数据形式](#数据形式)
  - [模型](#模型)
- [时间序列数据特征工程](#时间序列数据特征工程)
    - [数据](#数据)
  - [创建 Date Time 特征](#创建-date-time-特征)
    - [常见特征](#常见特征)
  - [创建 Lagging 特征](#创建-lagging-特征)
    - [滑动窗口特征](#滑动窗口特征)
    - [滑动窗口统计量特征](#滑动窗口统计量特征)
    - [TODO](#todo)
  - [创建 Window 变量](#创建-window-变量)
- [时间序列特征正规化和标准化](#时间序列特征正规化和标准化)
  - [正规化 Normalize](#正规化-normalize)
  - [标准化 Standardize](#标准化-standardize)
- [参考文章](#参考文章)
</p></details><p></p>


# 时间序列回归

时序预测模型与回归预测模型不同，时序预测模型依赖于数值在时间上的先后顺序，是回归模型中的一部分。
简单来说，时间序列的回归分析需要分析历史数据，找到历史数据演化中的模式和特征，
其主要分为线性回归分析和非线性回归分析两种类型

回归分析多采用机器学习方法，我们首先需要明确机器学习（或深度学习）模型构建与验证的主体思路：

1. 分析数据构建数据特征，将数据转化为特征样本集合
2. 明确样本与标签（Label），划分训练集与测试集
3. 比较不同模型在相同的训练集中的效果，或是相同模型的不同参数在同一个训练集中拟合的效果
4. 在验证样本集中验证模型的准确度，通过相关的结果评估公式选择表现最好同时没有过拟合的模型

## 常见线性回归模型

* 线性回归(Linear Regression)
* 多项式回归(Polynomial Regression)
* 逐步回归(Stepwise Regression)
* 岭回归(Ridge Regression)
* 套索回归(Lasso Regression)
* 弹性回归(ElasticNet Regression)

## 常见的非线性回归模型

时序统计学模型（AR、ARMA、ARIMA）模型，建模的思路源于针对当前观测点的最近P个点和最近Q个点的误差值进行建模，结构如下：

`$$Y_{t} = \sum_{j=1}^{P} \phi_{j}Y_{t-j} - \sum_{k=i}^{Q}\theta_{k}\epsilon_{t-k} + \epsilon_{t}$$`

而在现实背景中，很多数据并不是严格按照线性关系刻画的。为了兼顾模型的可解释性，
很多工作将非线性的数据进行各种变换(幂函数变换、倒数变换、指数变换、对数变换、Box-Cax等)将一个非线性问题转换成一个呈现线性关系的问题，
再利用相应的模型进行解决

* 逻辑回归(Logistic Regression)
* 树回归
* 神经网络模型



# 时间序列数据 vs 监督机器学习数据

将时间序列数据转换为监督机器学习问题数据

* 将单变量时间序列数据转换为机器学习问题数据
* 将多变量时间序列数据转换为机器学习问题数据

## 数据形式

时间序列是按照时间索引排列的一串数字, 可以理解为为有序值构成的一列数据或有序列表: 

| timestamp           | value |
|---------------------|-------|
| 2019-10-01 00:00:00 | 0     |
| 2019-10-01 01:00:00 | 1     |
| 2019-10-01 02:00:00 | 2     |
| 2019-10-01 03:00:00 | 3     |
| 2019-10-01 04:00:00 | 4     |

监督机器学习问题由自变量 `$X$` 和 因变量 `$Y$`
构成, 通常使用自变量 `$X$` 来预测因变量 `$Y$`:


| `$X_1$` | `$X_2$` | `$\cdots$` | `$X_p$` | `$Y$` |
|-------|-------|----------|-------|-----|
|a1     |b1     |`$\cdots$`  | c1    | y1  |
|a2     |b2     |`$\cdots$`  | c2    | y2  |
|a3     |b3     |`$\cdots$`  | c3    | y3  |
|a4     |b4     |`$\cdots$`  | c4    | y4  |
|a5     |b5     |`$\cdots$`  | c5    | y5  |


## 模型

`$$\sum_{i=0}^{19}a_{i,j}A_{t-i}+\sum_{i=0}^{19}b_{i,j}B_{t-i}+ \cdots+\sum_{i=0}^{19}z_{i,j}Z_{t-i} = A_{t+j}+\epsilon, j = 1, 2, 3, 4, 5, 6$$`

| `$A_1$`         | `$A_2$`         | `$\cdots$`  | `$A_{20}$`       | `$B_1$`         | `$B_1$`         | `$\cdots$`  | `$B_{20}$`       | `$\cdots$`  | `$Z_{1}$`       | `$Z_{2}$`       | `$\cdots$`  | `$Z_{20}$`       | `$A_{21}$`       |
|---------------|---------------|----------|----------------|---------------|---------------|----------|----------------|----------|---------------|---------------|----------|----------------|----------------|
| `$a_{(0, 1)}$`  | `$a_{(1, 1)}$`  | `$\cdots$`  | `$a_{(19, 1)}$`  | `$b_{(0, 1)}$`  | `$b_{(1, 1)}$`  | `$\cdots$`  | `$b_{(19, 1)}$`  | `$\cdots$`  | `$z_{(0, 1)}$`  | `$z_{(1, 1)}$`  | `$\cdots$`  | `$z_{(19, 1)}$`  | `$a_{(20, 1)}$`  |
| `$a_{(0, 2)}$`  | `$a_{(1, 2)}$`  | `$\cdots$`  | `$a_{(19, 2)}$`  | `$b_{(0, 2)}$`  | `$b_{(1, 2)}$`  | `$\cdots$`  | `$b_{(19, 2)}$`  | `$\cdots$`  | `$z_{(0, 2)}$`  | `$z_{(1, 2)}$`  | `$\cdots$`  | `$z_{(19, 2)}$`  | `$a_{(20, 2)}$`  |
| `$a_{(0, 3)}$`  | `$a_{(1, 3)}$`  | `$\cdots$`  | `$a_{(19, 3)}$`  | `$b_{(0, 3)}$`  | `$b_{(1, 3)}$`  | `$\cdots$`  | `$b_{(19, 3)}$`  | `$\cdots$`  | `$z_{(0, 3)}$`  | `$z_{(1, 3)}$`  | `$\cdots$`  | `$z_{(19, 3)}$`  | `$a_{(20, 3)}$`  |
| `$a_{(0, 4)}$`  | `$a_{(1, 4)}$`  | `$\cdots$`  | `$a_{(19, 4)}$`  | `$b_{(0, 4)}$`  | `$b_{(1, 4)}$`  | `$\cdots$`  | `$b_{(19, 4)}$`  | `$\cdots$`  | `$z_{(0, 4)}$`  | `$z_{(1, 4)}$`  | `$\cdots$`  | `$z_{(19, 4)}$`  | `$a_{(20, 4)}$`  |
| `$a_{(0, 5)}$`  | `$a_{(1, 5)}$`  | `$\cdots$`  | `$a_{(19, 5)}$`  | `$b_{(0, 5)}$`  | `$b_{(1, 5)}$`  | `$\cdots$`  | `$b_{(19, 5)}$`  | `$\cdots$`  | `$z_{(0, 5)}$`  | `$z_{(1, 5)}$`  | `$\cdots$`  | `$z_{(19, 5)}$`  | `$a_{(20, 5)}$`  |
| `$a_{(0, 6)}$`  | `$a_{(1, 6)}$`  | `$\cdots$`  | `$a_{(19, 6)}$`  | `$b_{(0, 6)}$`  | `$b_{(1, 6)}$`  | `$\cdots$`  | `$b_{(19, 6)}$`  | `$\cdots$`  | `$z_{(0, 6)}$`  | `$z_{(1, 6)}$`  | `$\cdots$`  | `$z_{(19, 6)}$`  | `$a_{(20, 6)}$`  |


# 时间序列数据特征工程

三种特征:

* Date Time Features
* Lag Features
* Window Features

### 数据

[澳大利亚墨尔本市10年(1981-1990年)内的最低每日温度](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv)

```python
import pandas as pd 
from pandas import Grouper
import matplotlib.pyplot as plt 

series = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    header = 0,
    index_col = 0, 
    parse_dates = True,
    date_parser = lambda dates: pd.to_datetime(dates, format = '%Y-%m-%d'),
    squeeze = True
)
print(series.head())
```

```
Date
1981-01-01    20.7
1981-01-02    17.9
1981-01-03    18.8
1981-01-04    14.6
1981-01-05    15.8
Name: Temp, dtype: float64
```

```python
series.plot()
plt.show()
```

![img](images/line.png)

```python
series.hist()
plt.show()
```

![img](images/hist.png)


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

# 时间序列特征正规化和标准化

## 正规化 Normalize

```python
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

# Data
series = pd.read_csv(
    "daily-minimum-temperautures-in-me.csv", 
    header = 0, 
    index_col = 0
)
print(series.head())
values = series.values
values = values.reshape((len(values), 1))

# Normalization
scaler = MinMaxScaler(feature_range = (0, 1))
scaler = scaler.fit(values)
print("Min: %f, Max: %f" % (scaler.data_min_, scaler.data_max_))

# 正规化
normalized = scaler.transform(values)
for i in range(5):
    print(normalized[i])

# 逆变换
inversed = scaler.inverse_transform(normalized)
for i in range(5):
    print(inversed[i])
```

``` 
            Temp
Date            
1981-01-02  17.9
1981-01-03  18.8
1981-01-04  14.6
1981-01-05  15.8
1981-01-06  15.8
Min: 0.000000, Max: 26.300000

[0.68060837]
[0.7148289]
[0.55513308]
[0.60076046]
[0.60076046]
[17.9]
[18.8]
[14.6]
[15.8]
[15.8]
```

```python
normalized.plot()
```

![img](images/line_normalized.png)


```python
inversed.plot()
```

![img](images/line_inversed_normalized.png)

## 标准化 Standardize

```python
from pandas import read_csv
from sklearn.preprocessing import StandarScaler
from math import sqrt

# Data
series = pd.read_csv(
    "daily-minimum-temperatures-in-me.csv", 
    header = 0,
    index_col = 0
)
print(series.head())
values = series.values
values = values.reshape((len(values), 1))

# Standardization
scaler = StandardScaler()
scaler = scaler.fit(values)
print("Mean: %f, StandardDeviation: %f" % (scaler.mean_, sqrt(scaler.var_)))

# 标准化
normalized = scaler.transform(values)
for i in range(5):
    print(normalized[i])

# 逆变换
inversed = scaler.inverse_transform(normalized)
for i in range(5):
    print(inversed[i])
```

```python
normalized.plot()
```

![img](images/line_standard.png)


```python
inversed.plot()
```

![img](images/line_inversed_standard.png)

# 参考文章

- [blog1](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
- [blog2](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
- [Machine Learning Strategies for Time Series Forecasting](https://link.springer.com/chapter/10.1007%2F978-3-642-36318-4_3)
- [slide](http://di.ulb.ac.be/map/gbonte/ftp/time_ser.pdf)
- [Machine Learning for Sequential Data: A Review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)
- [blog3](https://cloud.tencent.com/developer/article/1042809)
- [resample](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)

