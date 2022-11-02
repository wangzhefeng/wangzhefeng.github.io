---
title: 时间序列预处理
author: 王哲峰
date: '2022-04-22'
slug: timeseries-process
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

- [时间序列聚合](#时间序列聚合)
  - [Pandas 重采样](#pandas-重采样)
    - [API](#api)
    - [升采样](#升采样)
    - [降采样](#降采样)
    - [Function application](#function-application)
    - [Indexing 和 iteration](#indexing-和-iteration)
    - [稀疏采用](#稀疏采用)
  - [SciPy 重采样](#scipy-重采样)
    - [API](#api-1)
    - [升采样](#升采样-1)
    - [降采样](#降采样-1)
    - [等间隔采样](#等间隔采样)
    - [不等间隔采样](#不等间隔采样)
- [时间序列缺失处理](#时间序列缺失处理)
  - [时间序列插值](#时间序列插值)
  - [Pandas 插值算法](#pandas-插值算法)
    - [Pandas 中缺失值的处理](#pandas-中缺失值的处理)
    - [缺失值插值算法 API](#缺失值插值算法-api)
      - [pandas.DataFrame.interpolate](#pandasdataframeinterpolate)
  - [Scipy 插值算法](#scipy-插值算法)
    - [1-D interpolation](#1-d-interpolation)
    - [Multivariate data interpolation](#multivariate-data-interpolation)
    - [Spline interpolation](#spline-interpolation)
    - [Using radial basis functions for smoothing/interpolate](#using-radial-basis-functions-for-smoothinginterpolate)
- [时间序列平滑](#时间序列平滑)
  - [时间序列平滑简介](#时间序列平滑简介)
  - [差分运算、变化率](#差分运算变化率)
    - [差分运算](#差分运算)
      - [p 阶差分](#p-阶差分)
      - [k 步差分](#k-步差分)
      - [差分运算 API](#差分运算-api)
    - [百分比变化率](#百分比变化率)
    - [移动索引](#移动索引)
  - [移动平均、指数平滑](#移动平均指数平滑)
    - [时间序列预测](#时间序列预测)
    - [描述趋势特征](#描述趋势特征)
    - [简单移动平均](#简单移动平均)
    - [加权移动平均](#加权移动平均)
    - [指数加权移动/指数平滑](#指数加权移动指数平滑)
      - [一次指数平滑](#一次指数平滑)
      - [二次指数平滑](#二次指数平滑)
      - [三次指数平滑](#三次指数平滑)
      - [平滑系数 `$\alpha$` 的取值](#平滑系数-alpha-的取值)
      - [平滑初值 `$S^{(1)}_{1}$` 的取值](#平滑初值-s1_1-的取值)
  - [滤波算法](#滤波算法)
    - [限幅滤波(程序判断滤波法)](#限幅滤波程序判断滤波法)
    - [中位数滤波](#中位数滤波)
    - [算法平均滤波](#算法平均滤波)
    - [递推平均滤波(滑动平均滤波)](#递推平均滤波滑动平均滤波)
    - [中位数平均滤波(防脉冲干扰平均滤波)](#中位数平均滤波防脉冲干扰平均滤波)
    - [限幅平均滤波](#限幅平均滤波)
    - [一阶滞后滤波](#一阶滞后滤波)
    - [加权递推平均滤波](#加权递推平均滤波)
    - [消抖滤波](#消抖滤波)
    - [限幅滤波](#限幅滤波)
    - [低通滤波](#低通滤波)
    - [高通滤波](#高通滤波)
    - [带通滤波](#带通滤波)
    - [带阻滤波](#带阻滤波)
    - [卡尔曼滤波](#卡尔曼滤波)
    - [参考](#参考)
    - [滤波算法 Python 实现](#滤波算法-python-实现)
- [时间序列降噪](#时间序列降噪)
  - [移动平均](#移动平均)
  - [傅里叶变换](#傅里叶变换)
  - [小波分析](#小波分析)
</p></details><p></p>


# 时间序列聚合


通常来自不同数据源的时间轴常常无法一一对应, 此时就要用到改变时间频率的方法进行数据处理. 
由于无法改变实际测量数据的频率, 我们能做的是改变数据收集的频率
    
- 上采样(up sampling): 在某种程度上是凭空获得更高频率数据的方式, 不增加额外的信息
- 下采样(down sampling): 减少数据收集的频率, 也就是从原始数据中抽取子集的方式


重采样(resampling)指的是将时间序列从一个频率转换到另一个频率的处理过程。
是对原样本重新处理的一个方法，是一个对常规时间序列数据重采样和频率转换的便捷的方法

将高频率的数据聚合到低频率称为降采样(downsampling)，
而将低频率数据转换到高频率则称为升采样(upsampling)

## Pandas 重采样

### API

```python
pd.DataFrame.resample(
    rule, 
    how = None, 
    axis = 0, 
    fill_method = None, 
    closed = None, 
    label = None, 
    convention = 'start', 
    kind = None, 
    loffset = None, 
    limit = None, 
    base = 0, 
    on = None, 
    level = None
)
```

主要参数说明:

* `rule`: DateOffset, Timedelta or str
    - 表示重采样频率，例如 ‘M’、‘5min’，Second(15)
* `how`: str
    - 用于产生聚合值的函数名或数组函数，例如'mean'、'ohlc'、'np.max'等，默认是'mean'，
      其他常用的值有：'first'、'last'、'median'、'max'、'min'
* `axis`: {0 or 'index', 1 or 'columns'}, default 0
    - 默认是纵轴，横轴设置 axis=1
* `fill_method`: str, default None
    - 升采样时如何插值，比如 ffill、bfill 等
* `closed`: {'right', 'left'}, default None
    - 在降采样时，各时间段的哪一段是闭合的，'right' 或 'left'，默认 'right'
* `label`: {'right', 'left'}, default None
    - 在降采样时，如何设置聚合值的标签，例如，9:30-9:35 会被标记成 9:30 还是 9:35, 默认9:35
* `convention`: {'start', 'end', 's', 'e'}, default 'start'
    - 当重采样时期时，将低频率转换到高频率所采用的约定（'start'或'end'）。默认'end'
* `kind`: {'timestamp', 'period'}, optional, default None
    - 聚合到时期（'period'）或时间戳（'timestamp'），默认聚合到时间序列的索引类型
* `loffset`: timedelta, default None
    - 面元标签的时间校正值，比如'-1s'或Second(-1)用于将聚合标签调早1秒
* `limit`: int, default None
    - 在向前或向后填充时，允许填充的最大时期数


* `pd.DataFrame.resample()` 和 `pd.Series.resample()`
    - 上采样
        - `.resample().ffill()`
        - `.resample().bfill()`
        - `.resample().pad()`
        - `.resample().nearest()`
        - `.resample().fillna()`
        - `.resample().asfreq()`
        - `.resample().interpolate()`
    - 下采样(计算聚合、统计函数)
        - `.resample().<func>()`
        - `.resample.count()`
        - `.resample.nunique()`
        - `.resample.first()`
        - `.resample.last()`
        - `.resample.ohlc()`
        - `.resample.prod()`
        - `.resample.size()`
        - `.resample.sem()`
        - `.resample.std()`
        - `.resample.var()`
        - `.resample.quantile()`
        - `.resample.mean()`
        - `.resample.median()`
        - `.resample.min()`
        - `.resample.max()`
        - `.resample.sum()`
    - Function application
        - `.resample().apply(custom_resampler)`: 自定义函数
        - `.resample().aggregate()`
        - `.resample().transfrom()`
        - `.resample().pipe()`
    - Indexing, iteration
        - `.__iter__`
        - `.groups`
        - `.indices`
        - `get_group()`
    - 稀疏采样

```python
import pandas as pd

rng = pd.data_range("2000-01-01", periods = 100, freq = "D")
ts = pd.Series(np.random.randn(len(rng)), index = rng)
ts
```

```
2000-01-01   -0.184415
2000-01-02   -0.078049
2000-01-03    1.550158
2000-01-04    0.206498
2000-01-05    0.184059
                ...   
2000-04-05   -0.574207
2000-04-06   -1.719587
2000-04-07    0.140673
2000-04-08   -1.234146
2000-04-09   -0.835341
Freq: D, Length: 100, dtype: float64
```

### 升采样





### 降采样

在用 `resample` 对数据进行降采样时，需要考虑两个参数:

* 各区间哪边是闭合的
* 如何标记各个聚合面元，用区间的开头还是末尾


```python
df = pd.DataFrame({
    'A': [1, 1, 2, 1, 2],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 1, 1, 2]
}, columns = ['A', 'B', 'C'])

df.groupby("A").mean()
df.groupby(["A", "B"]).mean()
df.groupby("A")["B"].mean()
```

### Function application


### Indexing 和 iteration


### 稀疏采用








## SciPy 重采样

### API

```python
import scipy

scipy.signal.resample(
    x, 
    num, 
    t = None, 
    axis = 0, 
    window = None, 
    domain = 'time'
)
```

参数解释:

* `x`: 待重采样的数据
* `num`: 重采样的点数，int 型
* `t`: 如果给定 t，则假定它是与 x 中的信号数据相关联的等距采样位置
* `axis`: 对哪个轴重采样，默认是 0

### 升采样

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# 原始数据
t = np.linspace(0, 3, 20, endpoint = False)
y  = np.cos(t ** 2)

# 升采样, 20 个点采 40 个点
re_y = singal.resample(y, 40)
t_new = np.linspace(0, 3, len(re_y), endpoint = False)

# plot
plt.plot(t, y, "b*--", label = "raw")
plt.plot(t_new, re_y, "r.--", lable = "resample data")
plt.legend()
plt.show()
```

![img](images/scipy_upsample.png)

### 降采样

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 原始数据
t = np.linspace(0, 3, 40, endpoint = False)
y = np.cos(t ** 2)

# 降采样, 40 个点采 20 个点
re_y = signal_resample(y, 20)
t_new = np.linspace(0, 3, len(re_y), endpoint = False)

# plot
plt.plot(t, y, "b*--", label = "raw data")
plt.plot(t_new, re_y, "r.--", label = "resample data")
plt.legend()
plt.show()
```

![img](images/scipy_downsample.png)

### 等间隔采样

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 原始数据
t = np.linspace(0, 3, 40, endpoint = False)
y = np.cos(t ** 2)

# 等间隔降采样，40 个点采 20 个点
re_y, re_t = signal.resample(y, 20, t = t)

# plot
plt.plot(t, y, "b*--", label = "raw data")
plt.plot(re_t, re_y, "g.--", label = "resample data")
plt.legend()
plt.show()
```

![img](images/scipy_downsample_t.png)

### 不等间隔采样

原始数据不等间隔，采样成等间隔的点(点数不变)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 原始数据
t = np.array([
    0, 2, 3, 3.5, 4.5, 
    5.2, 6, 6.3, 8, 9, 
    10, 11.2, 12.3, 12.9, 14.5,
    16, 17, 18, 19, 20]) / 10
x = np.sin(3 * t)

# 重采样为等间隔
t1 = np.linspace(0, 2, 20, endpoint = True)
re_x, re_t = signal.resample(x, 20, t = t1)

# plot
plt.plot(t, x, 'b*--', label = 'raw data')
plt.plot(re_t, re_x, 'g.--', label = 'resample data')
plt.legend()
plt.show()
```

![img](images/scipy_resample1.png)


```python
import numpy as np
import matplotlib.pyplot
from scipy import signal

# 原始数据
t = np.array([
    0, 2, 3, 3.5, 4.5, 
    5.2, 6, 6.3, 8, 9, 
    10, 11.2, 12.3, 12.9, 14.5, 
    16, 17, 18, 19, 20])/10
x = np.sin(3 * t)

# 重采样
t1 = np.linspace(0, 2, 20, endpoint = True)
re_x = signal.resample(x, 20)

plt.plot(t, x, 'b*--', label = 'raw data')
plt.plot(t1, re_x, 'r.--', label = 'resample data')
plt.legend()
plt.show()
```

![img](images/scipy_resample2.png)


# 时间序列缺失处理

最常用的处理缺失值的方法包括填补(imputation) 和删除(deletion)两种

常见的数据填补(imputation)方法:

* forward fill 和 backward fill
    - 根据缺失值前/后的最近时间点数据填补当前缺失值
* moving average
* interpolation
    - 插值方法要求数据和邻近点之间满足某种拟合关系, 
      因此插值法是一种先验方法且需要代入一些业务经验

## 时间序列插值

## Pandas 插值算法

### Pandas 中缺失值的处理

1. Python 缺失值类型
   - `None`
   - `numpy.nan`
   - `NaN`
   - `NaT`
2. 缺失值配置选项
   - 将 `inf` 和 `-inf` 当作 `NA`
      - `pandas.options.mod.use_inf_as_na = True`
3. 缺失值检测
   - `pd.isna()`
   - `.isna()`
   - `pd.notna()`
   - `.notna()`
4. 缺失值填充
   - `.fillna()`
      - `.where(pd.notna(df), dict(), axis = "columns")`
      - `.ffill()`
         - `.fillna(method = "ffill", limit = None)`
      - `.bfill()`
         - `.fillna(method = "bfill", limit = None)`
   - `.interpolate(method)`
   - `.replace(, method)`
5. 缺失值删除
   - `.dropna(axis)`

### 缺失值插值算法 API

#### pandas.DataFrame.interpolate

```python

pandas.DataFrame.interpolate(
   method = 'linear', 
   axis = 0, 
   limit = None, 
   inplace = False, 
   limit_direction = 'forward', 
   limit_area = None, 
   downcast = None, 
   **kwargs
)
```

- `method`
   - `linear`: 等间距插值, 支持多多索引
   - `time`: 对索引为不同分辨率时间序列数据插值
   - `index`, `value`: 使用索引的实际数值插值
   - `pad`: 使用其他不为 `NaN` 的值插值
   - `nearest`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `zero`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `slinear`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `quadratic`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `cubic`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `spline`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `barycentric`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `polynomial`: 参考 `scipy.iterpolate.interp1d` 中的插值算法
   - `krogh`:
      - `scipy.interpolate.KroghInterpolator`
   - `piecewise_polynomial`
   - `spline`
      - `scipy.interpolate.CubicSpline`
   - `pchip`
      - `scipy.interpolate.PchipInterpolator`
   - `akima`
      - `scipy.interpolate.Akima1DInterpolator`
   - `from_derivatives`
      - `scipy.interpolate.BPoly.from_derivatives`
- `axis`:
   - 1
   - 0
- `limit`: 要填充的最大连续 `NaN` 数量,  :math:`>0`
- `inplace`: 是否在原处更新数据
- `limit_direction`: 缺失值填充的方向
   - forward
   - backward
   - both
- `limit_area`: 缺失值填充的限制区域
   - None
   - inside
   - outside
- `downcast`: 强制向下转换数据类型
   - infer
   - None
- `**kwargs`
   - 传递给插值函数的参数


```python
s = pd.Series([])
df = pd.DataFrame({})

s.interpolate(args)
df.interpolate(args)
df[""].interpolate(args)
```

## Scipy 插值算法

- scipy.interpolate.Akima1DInterpolator
   - 三次多项式插值
- scipy.interpolate.BPoly.from_derivatives
   - 多项式插值
- scipy.interpolate.interp1d
   - 1-D 函数插值
- scipy.interpolate.KroghInterpolator
   - 多项式插值
- scipy.interpolate.PchipInterpolator
   - PCHIP 1-d 单调三次插值
- scipy.interpolate.CubicSpline
   - 三次样条插值

### 1-D interpolation

```python
class scipy.interpolate.interp1d(x, y, 
                           kind = "linear", 
                           axis = -1, 
                           copy = True, 
                           bounds_error = None, 
                           fill_value = nan, 
                           assume_sorted = False)
```

- `kind`
   - linear
   - nearest
   - 样条插值(spline interpolator):
      - zero
         - zeroth order spline
      - slinear
         - first order spline
      - quadratic
         - second order spline
      - cubic
         - third order spline
   - previous
      - previous value
   - next
      - next value

```python
import numpy
from scipy.interpolate import interp1d

# 原数据
x = np.linspace(0, 10, num = 11, endpoint = True)
y = np.cos(-x ** 2 / 9.0)

# interpolation
f1 = interp1d(x, y, kind = "linear")
f2 = interp1d(x, y, kind = "cubic")
f3 = interp1d(x, y, kind = "nearst")
f4 = interp1d(x, y, kind = "previous")
f5 = interp1d(x, y, kind = "next")

xnew = np.linspace(0, 10, num = 1004, endpoint = True)
ynew1 = f1(xnew)
ynew2 = f2(xnew)
ynew3 = f3(xnew)
ynew4 = f4(xnew)
ynew5 = f5(xnew)
```

### Multivariate data interpolation

### Spline interpolation

### Using radial basis functions for smoothing/interpolate


# 时间序列平滑

## 时间序列平滑简介

数据平滑通常是为了消除一些极端值或测试误差. 
即使有些极端值本身是真实的, 但是并没有反映出潜在的数据模式, 仍需处理

* 数据平滑方法
    - 移动平均(weighted averaging, moving average), 既可以给定时间窗口内样本点相同权重, 或邻近点指定更高权重
    - 指数平滑(exponential smoothing), 所有的指数平滑法都要更新上一时间步长的计算结果, 
      并使用当前时间步长的数据中包含的新信息. 通过"混合"新信息和旧信息来实现, 
      而相关的新旧信息的权重由一个可调整的参数来控制
        - 一次指数平滑(exponential smoothing)   
            - 从最邻近到最早的数据点的权重呈现指数型下降的规律, 针对没有趋势和季节性的序列
        - 二次指数平滑(Holt exponential smoothing)   
            - 通过引入一个额外的系数来解决指数平滑无法应用于具有趋势性数据的问题, 
              针对有趋势但没有季节性的序列
        - 三次指数平滑(Holt-Winters exponential smoothing)
            - 通过再次引入一个新系数的方式同时解决了二次平滑无法解决具有季节性变化数据的不足
    - 差分运算
    - 时间序列分解
    - 日历调增
    - 数学变换
* 数据平滑工具
    - `.diff()`
    - `.shift()`
    - `.rolling()`
    - `.expanding()`
    - `.ewm()`
    - `.pct_change()`

## 差分运算、变化率

### 差分运算

#### p 阶差分

相距一期的两个序列值至之间的减法运算称为 `$1$` 阶差分运算; 对
`$1$` 阶差分后序列在进行一次 `$1$` 阶差分运算称为 `$2$`
阶差分; 以此类推, 对 `$p-1$` 阶差分后序列在进行一次 `$1$`
阶差分运算称为 `$p$` 阶差分.

`$$\Delta x_{t} = x_{t-1} - x_{t-1}$$`

`$$\Delta^{2} x_{t} = \Delta x_{t} - \Delta x_{t-1}$$`

`$$\Delta^{p} x_{t} = \Delta^{p-1} x_{t} - \Delta^{p-1} x_{t-1}$$`

#### k 步差分

相距 `$k$` 期的两个序列值之间的减法运算称为 `$k$` 步差分运算.

`$$\Delta_{k}x_{t} = x_{t} - x_{t-k}$$`

#### 差分运算 API

* pandas.Series.diff
* pandas.DataFrame.diff
* pandas.DataFrame.percent
* pandas.DataFrame.shift

```python
# 1 阶差分、1步差分
pandas.DataFrame.diff(periods = 1, axis = 0)

# 2 步差分
pandas.DataFrame.diff(periods = 2, axis = 0)

# k 步差分
pandas.DataFrame.diff(periods = k, axis = 0)

# -1 步差分
pandas.DataFrame.diff(periods = -1, axis = 0)
```

### 百分比变化率

当前值与前一个值之间的百分比变化

```python
DataFrame/Series.pct_change(periods = 1, 
               fill_method = 'pad', 
               limit = None, 
               freq = None, 
               **kwargs)
```

- periods
- fill_method
- limit
- freq



### 移动索引

```python
pandas.DataFrame.shift(periods, freq, axis, fill_value)
```





## 移动平均、指数平滑

移动平均作为时间序列中最基本的预测方法，计算虽然简单但却很实用。
不仅可以用于预测，还有一些其他的重要作用，比如平滑序列波动，
揭示时间序列的趋势特征

* 移动平均(moving average, SMA)
* 加权移动平均(weighted moving average, WMA)
* 指数加权移动平均(exponential weighted moving average, EMA, EWMA)

### 时间序列预测


### 描述趋势特征


### 简单移动平均

`$$m_t = \frac{1}{k}\sum_{i=1}^{k}y_{t-i}$$`

### 加权移动平均

`$$m_t = \sum_{i=1}^{k}\omega_{i}y_{t-i}$$`

### 指数加权移动/指数平滑

- 产生背景: 
    - 指数平滑由布朗提出、他认为时间序列的态势具有稳定性或规则性, 所以时间序列可被合理地顺势推延; 
      他认为最近的过去态势, 在某种程度上会持续的未来, 所以将较大的权数放在最近的资料. 
    - 指数平滑是在 20 世纪 50 年代后期提出的, 并激发了一些十分成功的预测方法. 
      使用指数平滑方法生成的预测是过去观测值的加权平均值, 并且随着过去观测值离预测值的距离的增大,
      权重呈指数型衰减. 换句话说, 观测值越近, 相应的权重越高. 该框架能够快速生成可靠的预测结果, 
      并且适用于广泛的时间序列, 这是一个巨大的优势并且对于工业应用来说非常重要.
- 基本原理: 
    - 移动平均(weighted averaging, moving average)
         - 既可以给定时间窗口内样本点相同权重, 或邻近点指定更高权重
    - 指数平滑(exponential smoothing), 所有的指数平滑法都要更新上一时间步长的计算结果, 
      并使用当前时间步长的数据中包含的新信息. 通过"混合"新信息和旧信息来实现, 
      而相关的新旧信息的权重由一个可调整的参数来控制
        - 一次指数平滑(exponential smoothing)
            - 从最邻近到最早的数据点的权重呈现指数型下降的规律, 针对没有趋势和季节性的序列
        - 二次指数平滑(Holt exponential smoothing) 
            - 通过引入一个额外的系数来解决指数平滑无法应用于具有趋势性数据的问题, 
              针对有趋势但没有季节性的序列
        - 三次指数平滑(Holt-Winters exponential smoothing)
            - 通过再次引入一个新系数的方式同时解决了二次平滑无法解决具有季节性变化数据的不足
- 方法应用: 
    - 指数平滑法是生产预测中常用的一种方法. 也用于中短期经济发展趋势预测, 
      所有预测方法中, 指数平滑是用得最多的一种. 

#### 一次指数平滑

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
* `$\alpha \in [0, 1]$`: 平滑系数/衰减因子, 可以控制对历史数据的遗忘程度,
  当 `$\alpha$` 接近 1 时表示只保留当前时间点的值

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

`$$\hat{x}_{i+h} = s_{i}$$`

其中:

* `$s_{i}$` 表示最后一个时间点对应的值
* `$h=1$` 表示下个预测值

#### 二次指数平滑

二次指数平滑是对一次指数平滑的再平滑. 它适用于具线性趋势的时间数列. 
虽然一次指数平均在产生新的数列的时候考虑了所有的历史数据, 
但是仅仅考虑其静态值, 即没有考虑时间序列当前的变化趋势. 
所以二次指数平滑一方面考虑了所有的历史数据, 另一方面也兼顾了时间序列的变化趋势. 

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

#### 三次指数平滑

若时间序列的变动呈现出二次曲线趋势, 则需要采用三次指数平滑法进行预测. 
实际上是在二次指数平滑的基础上再进行一次指数平滑. 它与二次指数平滑的区别
就是三次平滑还考虑到了季节性效应. 

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

#### 平滑系数 `$\alpha$` 的取值

1. 当时间序列呈现较稳定的水平趋势时, 应选较小的 `$\alpha$`, 一般可在 0.05~0.20 之间取值
2. 当时间序列有波动, 但长期趋势变化不大时, 可选稍大的 `$\alpha$` 值, 常在 0.1~0.4 之间取值
3. 当时间序列波动很大, 长期趋势变化幅度较大, 呈现明显且迅速的上升或下降趋势时, 宜选择较大的 `$\alpha$` 值, 
   如可在 0.6~0.8 间选值. 以使预测模型灵敏度高些, 能迅速跟上数据的变化. 
4. 当时间序列数据是上升(或下降)的发展趋势类型, `$\alpha$` 应取较大的值, 在 0.6~1 之间

#### 平滑初值 `$S^{(1)}_{1}$` 的取值

不管什么指数平滑都会有个初值, 平滑初值 `$S^{(1)}_{1}$` 的取值遵循如下规则即可: 

* 如果时间序列长度小于 20, 一般取前三项的均值

`$$S^{(1)}_{1} = S^{(2)}_{1} = S^{(3)}_{1} = \frac{1}{n}(x_{1} + x_{2} + x_{3})$$`

* 如果时间序列长度大于 20, 看情况调整第一项的值, 一般取第一项就行, 
  因为数据多, `$y^{*}_{0}$` 的取值对最后的预测结果影响不大


## 滤波算法

### 限幅滤波(程序判断滤波法)

- 方法: 
   - 根据经验判断, 确定两次采样允许的最大偏差值, 假设为 $\delta$ , 每次检测到新的值是判断: 如果本次值与上次值之差小于等于
      $\delta$ , 则本次值有效；如果本次值与上次值之差大于 $\delta$ , 则本次值无效, 放弃本次值, 用上一次值代替本次值；
- 优点: 
   - 能有效克服因偶然因素引起的脉冲干扰；
- 缺点: 
   - 无法抑制周期性的干扰, 平滑度差


### 中位数滤波

- 方法: 
   - 连续采样 $N$ 次($N$ 取奇数), 把 $N$ 次采样值按照大小排列, 取中间值为本次有效值；
- 优点: 
   - 能有效克服因偶然因素引起的波动干扰, 对温度、液位的变化缓慢的被测参数有良好的的滤波效果；
- 缺点: 
   - 对流量、速度等快速变化的参数不适用；

### 算法平均滤波

- 方法: 连续取 $N$ 个采样值进行算术平均运算, $N$ 值较大时: 信号平滑度较高, 但灵活性较低, $N$
   值较小时: 信号平滑度较低, 但灵敏度较高. $N$ 值的选取: 一般流量: $N=12$, 压力: $N = 4$；
- 优点: 适用于对一般具有随机干扰的信号进行滤波, 这样的信号的特点是有一个平均值, 信号在某一数值范围附近上下波动；
- 缺点: 对于测量速度较慢或要求数据计算速度较快的实时控制不适用, 比较浪费RAM；

### 递推平均滤波(滑动平均滤波)

- 方法: 
   - 把连续取 $N$ 个采样值看成一个队列队列的长度固定为 $N$
      每次采样到一个新数据放入队尾, 并扔掉原来队首的一次数据.(先进先出原则)
      把队列中的 $N$ 个数据进行算术平均运算, 就可获得新的滤波结果
      $N$ 值的选取: 流量, $N=1$ 压力: $N=4$ ；液面, $N=4~12$ ；温度, $N=1~4$；
- 优点: 
   - 对周期性干扰有良好的抑制作用, 平滑度高适用于高频振荡的系统；
- 缺点: 
   - 灵敏度低对偶然出现的脉冲性干扰的抑制作用较差不易消除由于脉冲干扰所引起的采样值偏差不适用于脉冲干扰比较严重的场合比较浪费 RAM；

### 中位数平均滤波(防脉冲干扰平均滤波)

- 方法: 
   - 相当于`中位值滤波法 + 算术平均滤波法` 连续采样 $N$ 个数据, 
      去掉一个最大值和一个最小值然后计算 $N-2$ 个数据的算术平均值N值的选取: $3~14$；
- 优点: 
   - 融合了两种滤波法的优点对于偶然出现的脉冲性干扰, 可消除由于脉冲干扰所引起的采样值偏差；
- 缺点: 
   - 测量速度较慢, 和算术平均滤波法一样比较浪费 RAM；

### 限幅平均滤波

- 方法: 
   - 相当于 `限幅滤波法 + 递推平均滤波法` 每次采样到的新数据先进行限幅处理, 再送入队列进行递推平均滤波处理；
- 优点: 
   - 融合了两种滤波法的优点对于偶然出现的脉冲性干扰, 可消除由于脉冲干扰所引起的采样值偏差；
- 缺点: 
   - 比较浪费 RAM；

### 一阶滞后滤波

- 方法: 
   - 取 $a=0~1$ 本次滤波结果 $=(1-a)$ 本次采样值 + a $\times$  上次滤波结果；
- 优点: 
   - 对周期性干扰具有良好的抑制作用 适用于波动频率较高的场合；
- 缺点: 
   - 相位滞后, 灵敏度低滞后程度取决于a值大小不能消除滤波频率高于采样频率的1/2的干扰信号；

### 加权递推平均滤波

- 方法: 
   - 是对递推平均滤波法的改进, 即不同时刻的数据加以不同的权通常是, 越接近现时刻的数据, 权取得越大. 给予新采样值的权系数越大, 则灵敏度越高, 但信号平滑度越低；
- 优点:
   - 适用于有较大纯滞后时间常数的对象 和采样周期较短的系统；
- 缺点: 
   - 对于纯滞后时间常数较小, 采样周期较长, 变化缓慢的信号不能迅速反应系统当前所受干扰的严重程度, 滤波效果差；

### 消抖滤波

- 方法: 
   - 设置一个滤波计数器将每次采样值与当前有效值比较: 如果采样值 ＝= 当前有效值, 则计数器清零如果采样值 <> 当前有效值, 
      则计数器 + 1, 并判断计数器是否 >= 上限 N(溢出) 如果计数器溢出,则将本次值替换当前有效值,并清计数器；
- 优点: 
   - 对于变化缓慢的被测参数有较好的滤波效果, 可避免在临界值附近控制器的反复开/关跳动或显示器上数值抖动
- 缺点: 
   - 对于快速变化的参数不宜如果在计数器溢出的那一次采样到的值恰好是干扰值,则会将干扰值当作有效值导入系统

### 限幅滤波

- 方法: 
   - 相当于 `限幅滤波法 + 消抖滤波法` 先限幅, 后消抖；
- 优点: 
   - 继承了 `限幅` 和 `消抖` 的优点改进了 `消抖滤波法` 中的某些缺陷, 避免将干扰值导入系统；
- 缺点: 
   - 对于快速变化的参数不宜；

### 低通滤波

- 低通滤波指的是去除高于某一阈值频率的信号
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 400hz 以上频率成分, 
 即截至频率为 400hz, 则归一化截止频率 $W_{n}=\frac{2 \times 400}{1000}=0.8$

### 高通滤波

- 高通滤波去除低于某一频率的信号
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 100hz 以下频率成分, 
即截至频率为 100hz, 则归一化截止频率 $W_{n}=\frac{2 \times 100}{100W_{}=0.2$

### 带通滤波

- 带通滤波指的是类似低通高通的结合保留中间频率信号
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 100hz 以下, 400hz 以上频率成分, 
   即截至频率为 100hz, 400hz, 则归一化截止频率 $W_{n1}=\frac{2 \times 100}{1000}=0.2`,
   $W_{n2}=\frac{2 \times 400}{1000}=0.8$, 所以:  $W_{n}=[0.02,0.8]$

### 带阻滤波

- 带阻滤波也是低通高通的结合只是过滤掉的是中间部分
- 假设采样频率为 1000hz, 信号本身最大的频率为 500hz, 要滤除 100hz 以上, 400hz 以下频率成分, 
 即截至频率为 100hz, 400hz, 则 $W_{n1}=\frac{2 \times 100}{1000}=0.2$,
 $W_{n2}=\frac{2 \times 400}{1000}=0.8$, 所以:  $W_{n}=[0.2,0.8]$
 和带通相似, 但是带通是保留中间, 而带阻是去除


### 卡尔曼滤波

- 什么是卡尔曼滤波？
   - 你可以在任何含有不确定信息的动态系统中使用卡尔曼滤波, 对系统下一步的走向做出有根据的预测, 即使伴随着各种干扰, 卡尔曼滤波总是能指出真实发生的情况；
   - 在连续变化的系统中使用卡尔曼滤波是非常理想的, 它具有占内存小的优点(除了前一个状态量外, 不需要保留其它历史数据), 而且速度很快, 很适合应用于实时问题和嵌入式系统；
- 算法的核心思想:
   - 根据当前的仪器 `测量值` 和上一刻的 `预测值` 和 `误差值`, 计算得到当前的最优量, 再预测下一刻的量. 
      - 核心思想比较突出的观点是把误差纳入计算, 而且分为 `预测误差` 和 `测量误差` 两种, 统称为 `噪声`. 
      - 核心思想还有一个非常大的特点是: 误差独立存在, 始终不受测量数据的影响. 

### 参考

- 1 https://blog.csdn.net/u010720661/article/details/63253509
- 2 http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
- 3 https://www.geek-workshop.com/thread-7694-1-1.html>
- 4 https://blog.csdn.net/u014033218/article/details/97004609?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-0&spm=1001.2101.3001.4242
- 5 https://blog.csdn.net/kengmila9393/article/details/81455165
- 6 https://blog.csdn.net/phker/article/details/48468591

### 滤波算法 Python 实现

```python
from scipy import signal
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def limiting_filter(inputs, per):
   """
   限幅滤波(程序判断滤波法)
   Args:
      inputs:
      per:
   """
   pass


def median_filter(inputs, per):
   """
   中位值滤波
   Args:
      inputs:
      per:
   """
   pass


def arithmetic_average_filter(inputs, per):
   '''
   算术平均滤波法
   Args:
      inputs:
      per:
   '''
   if np.shape(inputs)[0] % per != 0:
      lengh = np.shape(inputs)[0] / per
      for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
   inputs = inputs.reshape((-1,per))
   mean = []
   for tmp in inputs:
      mean.append(tmp.mean())
   return mean


def sliding_average_filter(inputs, per):
   '''
   递推平均滤波法(滑动平均滤波)
   Args:
      inputs:
      per: 
   filter = np.ones(200)*(1/200)
   sample_filter_03 = np.convolve(data,filter,'valid')
   sample_filter_012 = np.convolve(data,filter,'valid')
   '''
   if np.shape(inputs)[0] % per != 0:
      lengh = np.shape(inputs)[0] / per
      for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
   inputs = inputs.reshape((-1,per))
   tmpmean = inputs[0].mean()
   mean = []
   for tmp in inputs:
      mean.append((tmpmean+tmp.mean())/2)
      tmpmean = tmp.mean()
   return mean


def median_average_filter(inputs, per):
   '''
   中位值平均滤波法(防脉冲干扰平均滤波)
   Args:
      inputs:
      per:
   '''
   if np.shape(inputs)[0] % per != 0:
      lengh = np.shape(inputs)[0] / per
      for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
   inputs = inputs.reshape((-1,per))
   mean = []
   for tmp in inputs:
      tmp = np.delete(tmp,np.where(tmp==tmp.max())[0],axis = 0)
      tmp = np.delete(tmp,np.where(tmp==tmp.min())[0],axis = 0)
      mean.append(tmp.mean())
   return mean


def amplitude_limiting_average_filter(inputs, per, amplitude):
   '''
   限幅平均滤波法
   Args:
      inputs:
      per:
      amplitude: 限制最大振幅
   '''
   if np.shape(inputs)[0] % per != 0:
      lengh = np.shape(inputs)[0] / per
      for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
   inputs = inputs.reshape((-1,per))
   mean = []
   tmpmean = inputs[0].mean()
   tmpnum = inputs[0][0] #上一次限幅后结果
   for tmp in inputs:
      for index,newtmp in enumerate(tmp):
            if np.abs(tmpnum-newtmp) > amplitude:
               tmp[index] = tmpnum
            tmpnum = newtmp
      mean.append((tmpmean+tmp.mean())/2)
      tmpmean = tmp.mean()
   return mean


def first_order_lag_filter(inputs, a):
   '''
   一阶滞后滤波法
   Args:
      inputs:
      a: 滞后程度决定因子, 0~1
   '''
   tmpnum = inputs[0] #上一次滤波结果
   for index,tmp in enumerate(inputs):
      inputs[index] = (1-a)*tmp + a*tmpnum
      tmpnum = tmp
   return inputs


def weight_backstep_average_filter(inputs, per):
   '''
   加权递推平均滤波法
   Args:
      inputs: 
      per:
   '''
   weight = np.array(range(1,np.shape(inputs)[0]+1)) # 权值列表
   weight = weight/weight.sum()

   for index,tmp in enumerate(inputs):
      inputs[index] = inputs[index]*weight[index]
   return inputs


def shake_off_filter(inputs, N):
   '''
   消抖滤波法
   Args:
      inputs:
      N: 消抖上限
   '''
   usenum = inputs[0] #有效值
   i = 0 # 标记计数器
   for index,tmp in enumerate(inputs):
      if tmp != usenum:
            i = i + 1
            if i >= N:
               i = 0
               inputs[index] = usenum
   return inputs


def amplitude_limiting_shake_off_filter(inputs, amplitude, N):
   '''
   限幅消抖滤波法
   Args:
      inputs:
      amplitude: 限制最大振幅
      N:         消抖上限
   '''
   tmpnum = inputs[0]
   for index,newtmp in enumerate(inputs):
      if np.abs(tmpnum-newtmp) > amplitude:
            inputs[index] = tmpnum
      tmpnum = newtmp
   usenum = inputs[0]
   i = 0
   for index2,tmp2 in enumerate(inputs):
      if tmp2 != usenum:
            i = i + 1
            if i >= N:
               i = 0
               inputs[index2] = usenum
   return inputs


def low_pass_filer(data, N, Wn):
   """
   低通滤波:  低通滤波指的是去除高于某一阈值频率的信号

   Args:
      data ([type]): 要过滤的信号
      N ([type]): 滤波器的阶数
      Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
      b: 滤波器的分子
      a: 滤波器的分母
   Returns:
      [type]: [description]
   """
   b, a = signal.butter(N = N, Wn = Wn, btype = "lowpass")
   filted_data = signal.filtfilt(b, a, data)
   return filted_data


def high_pass_filter(data, N, Wn):
   """
   高通滤波: 高通滤波去除低于某一频率的信号

   Args:
      data ([type]): 要过滤的信号
      N ([type]): 滤波器的阶数
      Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
      b: 滤波器的分子
      a: 滤波器的分母

   Returns:
      [type]: [description]
   """
   b, a = signal.butter(N = N, Wn = Wn, btype = "highpass")
   filted_data = signal.filtfilt(b, a, data)
   return filted_data


def band_pass_filter(data, N, Wn):
   """
   带通滤波: 带通滤波指的是类似低通高通的结合保留中间频率信号

   Args:
      data ([type]): 要过滤的信号
      N ([type]): 滤波器的阶数
      Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
      b: 滤波器的分子
      a: 滤波器的分母

   Returns:
      [type]: [description]
   """
   b, a = signal.butter(N = N, Wn = Wn, btype = "bandpass")
   filted_data = signal.filtfilt(b, a, data)
   return filted_data


def band_stop_filter(data, N, Wn):
   """
   带阻滤波: 带阻滤波也是低通高通的结合只是过滤掉的是中间部分

   Args:
      data ([type]): 要过滤的信号
      N ([type]): 滤波器的阶数
      Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
      b: 滤波器的分子
      a: 滤波器的分母

   Returns:
      [type]: [description]
   """
   b, a = signal.butter(N = N, Wn = Wn, btype = "bandstop")
   filted_data = signal.filtfilt(b, a, data)
   return filted_data


if __name__ == "__main__":
   num = signal.chirp(np.arange(0, 0.5, 1 / 4410.0), f0 = 10, t1 = 0.5, f1 = 1000.0)
   result = arithmetic_average_filter(num.copy(), 30)
   plt.subplot(2, 1, 1)
   plt.plot(num)
   plt.subplot(2, 1, 2)
   plt.plot(result)
   plt.show()
```

# 时间序列降噪

## 移动平均

滚动平均值是先前观察窗口的平均值，其中窗口是来自时间序列数据的一系列值。
为每个有序窗口计算平均值。这可以极大地帮助最小化时间序列数据中的噪声

## 傅里叶变换

傅里叶变换可以通过将时间序列数据转换到频域去除噪声，可以过滤掉噪声频率，
然后应用傅里叶变换得到滤波后的时间序列

## 小波分析
