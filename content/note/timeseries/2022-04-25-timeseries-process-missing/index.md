---
title: 时间序列预处理-缺失处理
author: 王哲峰
date: '2022-04-25'
slug: timeseries-process-missing
categories:
  - timeseries
tags:
  - ml
---

# 时间序列插值

# Pandas 插值算法

## Pandas 中缺失值的处理

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

## 缺失值插值算法 API

### pandas.DataFrame.interpolate

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

# Scipy 插值算法

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

## 1-D interpolation

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

## Multivariate data interpolation

## Spline interpolation

## Using radial basis functions for smoothing/interpolate
