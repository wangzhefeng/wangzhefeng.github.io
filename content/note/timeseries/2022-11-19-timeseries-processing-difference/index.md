---
title: 时间序列差分
author: 王哲峰
date: '2022-11-19'
slug: timeseries-processing-difference
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
</style>

<details><summary>目录</summary><p>

- [差分运算](#差分运算)
  - [p 阶差分](#p-阶差分)
  - [k 步差分](#k-步差分)
  - [滞后算子](#滞后算子)
  - [差分运算 API](#差分运算-api)
- [百分比变化率](#百分比变化率)
</p></details><p></p>

# 差分运算

## p 阶差分

相距一期的两个序列值至之间的减法运算称为 `$1$` 阶差分运算; 对
`$1$` 阶差分后序列在进行一次 `$1$` 阶差分运算称为 `$2$`
阶差分; 以此类推, 对 `$p-1$` 阶差分后序列在进行一次 `$1$`
阶差分运算称为 `$p$` 阶差分.

`$$\Delta x_{t} = x_{t-1} - x_{t-1}$$`

`$$\Delta^{2} x_{t} = \Delta x_{t} - \Delta x_{t-1}$$`

`$$\Delta^{p} x_{t} = \Delta^{p-1} x_{t} - \Delta^{p-1} x_{t-1}$$`

## k 步差分

相距 `$k$` 期的两个序列值之间的减法运算称为 `$k$` 步差分运算.

`$$\Delta_{k}x_{t} = x_{t} - x_{t-k}$$`

## 滞后算子

滞后算子类似于一个时间指针, 当前序列值乘以一个滞后算子, 
就相当于把当前序列值的时间向过去拨了一个时刻

假设 `$B$` 为滞后算子:

`$$x_{t-1} = Bx_{t}$$`
`$$x_{t-2} = B^{2}x_{t}$$`
`$$\vdots$$`
`$$x_{t-p} = B^{p}x_{t}$$`

也可以用滞后算子表示差分运算:

`$p$` 阶差分:

`$$\Delta^{p}x_{t} = (1-B)^{p}x_{t} = \sum_{i=0}^{p}(-1)C_{p}^{i}x_{t-i}$$`

`$k$` 步差分:

`$$\Delta_{k}x_{t} = x_{t} - x_{t-k} = (1-B^{k})x_{t}$$`


## 差分运算 API

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

# 百分比变化率

当前值与前一个值之间的百分比变化

```python
DataFrame/Series.pct_change(
    periods = 1, 
    fill_method = 'pad', 
    limit = None, 
    freq = None, 
    **kwargs
)
```

