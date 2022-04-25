---
title: 时间序列平滑
author: 王哲峰
date: '2022-04-25'
slug: timeseries-smoothing
categories:
  - timeseries
tags:
  - ml
---




# 时间序列平滑简介

- 数据平滑通常是为了消除一些极端值或测试误差. 即使有些极端值本身是真实的, 
   但是并没有反映出潜在的数据模式, 仍需处理
- 数据平滑方法
   - 移动平均(weighted averaging, moving average), 既可以给定时间窗口内样本点相同权重, 
      或邻近点指定更高权重
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
- 数据平滑工具
   - .diff()
   - .shift()
   - .rolling()
   - .expanding()
   - .ewm()
   - .pct_change()

# 差分运算、变化率

## 差分运算

$p$ 阶差分:

   相距一期的两个序列值至之间的减法运算称为 $1$ 阶差分运算; 对
   $1$ 阶差分后序列在进行一次 $1$ 阶差分运算称为 $2$
   阶差分; 以此类推, 对 $p-1$ 阶差分后序列在进行一次 $1$
   阶差分运算称为 $p$ 阶差分.

   $\Delta x_{t} = x_{t-1} - x_{t-1}$

   $\Delta^{2} x_{t} = \Delta x_{t} - \Delta x_{t-1}$

   $\Delta^{p} x_{t} = \Delta^{p-1} x_{t} - \Delta^{p-1} x_{t-1}$

$k$ 步差分: 

   相距 $k$ 期的两个序列值之间的减法运算称为 $k$ 步差分运算.

   $\Delta_{k}x_{t} = x_{t} - x_{t-k}$

差分运算 API:

   - pandas.Series.diff
   - pandas.DataFrame.diff
   - pandas.DataFrame.percent
   - pandas.DataFrame.shift

   ```python

      # 1 阶差分、1步差分
      pandas.DataFrame.diff(periods = 1, axis = 0)

      # 2 步差分
      pandas.DataFrame.diff(periods = 2, axis = 0)

      # k 步差分
      pandas.DataFrame.diff(periods = k, axis = 0)

      # -1 步差分
      pandas.DataFrame.diff(periods = -1, axis = 0)

## 百分比变化率

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

# 移动索引

```python
pandas.DataFrame.shift(periods, freq, axis, fill_value)
```


# 移动平均、指数平滑
  
- 移动平均(moving average, SMA)
- 加权移动平均(weighted moving average, WMA)
- 指数加权移动平均(exponential weighted moving average, EMA, EWMA)

## API


- Standard moving window functions

  - ts.rolling(window, min_periods, center).count()
  - ts.rolling(window, min\ *periods, center, win*\ type).sum()

     - win_type

        - boxcar
        - triang
        - blackman
        - hamming
        - bartlett
        - parzen
        - bohman
        - blackmanharris
        - nuttall
        - barthann
        - kaiser (needs beta)
        - gaussian (needs std)
        - general_gaussian (needs power, width)
        - slepian (needs width)
        - exponential (needs tau)

  - ts.rolling(window, min\ *periods, center, win*\ type).mean()

     - win_type

        - boxcar
        - triang
        - blackman
        - hamming
        - bartlett
        - parzen
        - bohman
        - blackmanharris
        - nuttall
        - barthann
        - kaiser (needs beta)
        - gaussian (needs std)
        - general_gaussian (needs power, width)
        - slepian (needs width)
        - exponential (needs tau)

  - ts.rolling(window, min_periods, center).median()
  - ts.rolling(window, min_periods, center).var()
  - ts.rolling(window, min_periods, center).std()
  - ts.rolling(window, min_periods, center).min()
  - ts.rolling(window, min_periods, center).max()
  - ts.rolling(window, min_periods, center).corr()
  - ts.rolling(window, min_periods, center).cov()
  - ts.rolling(window, min_periods, center).skew()
  - ts.rolling(window, min_periods, center).kurt()
  - ts.rolling(window, min_periods, center).apply(func)
  - ts.rolling(window, min_periods, center).aggregate()
  - ts.rolling(window, min_periods, center).quantile()
  - ts.window().mean()
  - ts.window().sum()

- Standard expanding window functions

  - ts.expanding(window, min_periods, center).count()
  - ts.expanding(window, min\ *periods, center, win*\ type).sum()

     - win_type

        - boxcar
        - triang
        - blackman
        - hamming
        - bartlett
        - parzen
        - bohman
        - blackmanharris
        - nuttall
        - barthann
        - kaiser (needs beta)
        - gaussian (needs std)
        - general_gaussian (needs power, width)
        - slepian (needs width)
        - exponential (needs tau)

  - ts.expanding(window, min\ *periods, center, win*\ type).mean()

     - win_type

        - boxcar
        - triang
        - blackman
        - hamming
        - bartlett
        - parzen
        - bohman
        - blackmanharris
        - nuttall
        - barthann
        - kaiser (needs beta)
        - gaussian (needs std)
        - general_gaussian (needs power, width)
        - slepian (needs width)
        - exponential (needs tau)

  - ts.expanding(window, min_periods, center).median()
  - ts.expanding(window, min_periods, center).var()
  - ts.expanding(window, min_periods, center).std()
  - ts.expanding(window, min_periods, center).min()
  - ts.expanding(window, min_periods, center).max()
  - ts.expanding(window, min_periods, center).corr()
  - ts.expanding(window, min_periods, center).cov()
  - ts.expanding(window, min_periods, center).skew()
  - ts.expanding(window, min_periods, center).kurt()
  - ts.expanding(window, min_periods, center).apply(func)
  - ts.expanding(window, min_periods, center).aggregate()
  - ts.expanding(window, min_periods, center).quantile()

- Exponentially-weighted moving window functions

  - ts.ewm(window, min\ *periods, center, win*\ type).mean()

     - win_type

        - boxcar
        - triang
        - blackman
        - hamming
        - bartlett
        - parzen
        - bohman
        - blackmanharris
        - nuttall
        - barthann
        - kaiser (needs beta)
        - gaussian (needs std)
        - general_gaussian (needs power, width)
        - slepian (needs width)
        - exponential (needs tau)

  - ts.ewm(window, min_periods, center).std()
  - ts.ewm(window, min_periods, center).var()
  - ts.ewm(window, min_periods, center).corr()
  - ts.ewm(window, min_periods, center).cov()

- Rolling

```python
s = pd.Series(
   np.random.randn(1000),
   index = pd.date_range("1/1/2000", periods = 1000)
)
s = s.cumsum()
r = s.rolling(window = 60)
```

- Expanding

## 简单移动平均

$$m_t = \frac{1}{k}\sum_{i=1}^{k}y_{t-i}$$

## 加权移动平均


$$m_t = \sum_{i=1}^{k}\omega_{i}y_{t-i}$$


## 指数加权移动/指数平滑

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

### 一次指数平滑

当时间序列数据没有明显的变化趋势, 就可以使用一次指数平滑. 实际就是对历史数据的加权平均, 
它可以用于任何一种没有明显函数规律但确实存在某种前后关联的时间序列的短期预测. 
但是一次指数平滑也具有一定的局限性: 

1. 预测值不能反映趋势变动、季节波动等有规律的变动
2. 这种方法多适用于短期预测, 而不适合作中长期的预测
3. 由于预测值是历史数据的均值, 因此与实际序列的变化相比有滞后现象

一次指数平滑的预测公式/递推公式如下:

$$S_{i} = \alpha x_{i} + (1 - \alpha) S_{i-1}$$

其中: 

- $S_{i}$: $i$ 时刻的平滑值/预测值
- $x_{i}$: $i$ 时刻的实际观测值
- $S_{i-1}$: $i-1$ 时刻的平滑值/预测值
- $\alpha \in [0, 1]$: 平滑系数/衰减因子, 可以控制对历史数据的遗忘程度,
   当 $\alpha$ 接近 1 时表示只保留当前时间点的值

递推公式展开可以得到以下形式:


$$
\begin{aligned} 
S_{i} &= \alpha x_{i}+(1-\alpha) S_{i-1} \\ 
      &= \alpha x_{i}+(1-\alpha)\left[\alpha x_{i-1}+(1-\alpha) S_{i-2}\right] \\ 
      &= \alpha x_{i}+(1-\alpha)\left[\alpha x_{i-1}+(1-\alpha)\left[\alpha x_{i-2}+(1-\alpha) S_{i-3}\right]\right] \\ 
      &= \alpha\left[x_{i}+(1-\alpha) x_{i-1}+(1-\alpha)^{2} x_{i-2}+(1-\alpha)^{3} S_{i-3}\right] \\ 
      &= \ldots \\ 
      &= \alpha \sum_{j=0}^{i}(1-\alpha)^{j} x_{i-j} 
\end{aligned}
$$

从上式即可看出指数平滑考虑所有历史观测值对当前值的影响, 但影响程度随时间增长而减小, 
对应的时间序列预测公式:


$$\hat{x}_{i+h} = s_{i}$$


其中:

   - $s_{i}$ 表示最后一个时间点对应的值
   - $h=1$ 表示下个预测值


### 二次指数平滑

二次指数平滑是对一次指数平滑的再平滑. 它适用于具线性趋势的时间数列. 
虽然一次指数平均在产生新的数列的时候考虑了所有的历史数据, 
但是仅仅考虑其静态值, 即没有考虑时间序列当前的变化趋势. 
所以二次指数平滑一方面考虑了所有的历史数据, 另一方面也兼顾了时间序列的变化趋势. 

- 预测公式

$$
\left \{
\begin{array}{lcl}
S^{(1)}_{t} = \alpha x_{t-1} + (1 - \alpha) S^{(1)}_{t-1} \\ 
S^{(2)}_{t} = \alpha S^{(1)}_{t} + (1 - \alpha)S^{(2)}_{t-1} \\ 
\end{array} \right.
$$

$$
\left \{
\begin{array}{lcl}
a_{t} = 2S^{(1)}_{t} - S^{(2)}_{t} \\
b_{t} = \frac{\alpha}{1 - \alpha}(S^{(1)}_{t} - S^{(2)}_{t}) \\
\end{array} \right.
$$

$$x^{*}_{t+T} = a_{t} + b_{t} T$$

其中: 

- $x^{*}_{t+T}$: 未来 T 期的预测值
- $\alpha$: 平滑系数


### 三次指数平滑

若时间序列的变动呈现出二次曲线趋势, 则需要采用三次指数平滑法进行预测. 
实际上是在二次指数平滑的基础上再进行一次指数平滑. 它与二次指数平滑的区别
就是三次平滑还考虑到了季节性效应. 

- 预测公式

$$
\left \{
\begin{array}{lcl}
S^{(1)}_{t} = \alpha x_{t-1} + (1 - \alpha) S^{(1)}_{t-1} \\
S^{(2)}_{t} = \alpha S^{(1)}_{t} + (1 - \alpha)S^{(2)}_{t-1} \\
S^{(3)}_{t} = \alpha S^{(2)}_{t} + (1 - \alpha)S^{(3)}_{t-1} \\
\end{array} \right.
$$

$$
\left \{
\begin{array}{lcl}
a_{t} = 3S^{(1)}_{t} - 3S^{(2)}_{t} + S^{(3)}_{t} \\
b_{t} = \frac{\alpha}{2(1 - \alpha)^{2}}[(6 - 5 \alpha)S^{(1)}_{t} - 2(5 - 4\alpha)S^{(2)}_{t} + (4 - 3\alpha)S^{(3)}_{t}] \\
c_{t} = \frac{\alpha^{2}}{2(1 - \alpha)^{2}}(S^{(1)}_{t} - 2S^{(2)}_{t} + S^{(3)}_{t}) \\
\end{array} \right.
$$

$$x^{*}_{t+T} = a_{t} + b_{t} T + c_{t} T^{2}$$

其中: 

   - $x^{*}_{t+T}$: 未来 T 期的预测值
   - $\alpha$: 平滑系数

### 平滑系数 $\alpha$ 的取值

1. 当时间序列呈现较稳定的水平趋势时, 应选较小的 $\alpha$, 一般可在 0.05~0.20 之间取值
2. 当时间序列有波动, 但长期趋势变化不大时, 可选稍大的 $\alpha$ 值, 常在 0.1~0.4 之间取值
3. 当时间序列波动很大, 长期趋势变化幅度较大, 呈现明显且迅速的上升或下降趋势时, 宜选择较大的 $\alpha$ 值, 
   如可在 0.6~0.8 间选值. 以使预测模型灵敏度高些, 能迅速跟上数据的变化. 
4. 当时间序列数据是上升(或下降)的发展趋势类型,  $\alpha$ 应取较大的值, 在 0.6~1 之间

### 平滑初值 $S^{(1)}_{1}$ 的取值

不管什么指数平滑都会有个初值, 平滑初值 $S^{(1)}_{1}$ 的取值遵循如下规则即可: 

- 如果时间序列长度小于 20, 一般取前三项的均值
   
$$S^{(1)}_{1} = S^{(2)}_{1} = S^{(3)}_{1} = \frac{1}{n}(x_{1} + x_{2} + x_{3})$$

- 如果时间序列长度大于 20, 看情况调整第一项的值, 一般取第一项就行, 因为数据多, $y^{*}_{0}$ 的取值对最后的预测结果影响不大
