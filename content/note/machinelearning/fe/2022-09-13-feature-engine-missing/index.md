---
title: 缺失值处理
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-missing
categories:
  - feature engine
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

- [缺失值处理理论](#缺失值处理理论)
- [缺失值处理实战](#缺失值处理实战)
  - [Pandas 中缺失值的处理](#pandas-中缺失值的处理)
  - [缺失值插值算法 API](#缺失值插值算法-api)
    - [pandas.DataFrame.interpolate](#pandasdataframeinterpolate)
    - [scipy.interpolate.XXX](#scipyinterpolatexxx)
</p></details><p></p>


# 缺失值处理理论

- 当缺失数据比例很小时, 可直接对缺失记录进行舍弃或进行手工处理
- 实际数据中, 缺失数据往往占有相当的比重, 这时如果手工处理, 非常低效；
  如果舍弃缺失记录, 则会丢失大量信息, 使不完全观测数据与观测数据间产生系统差异, 
  对这样的数据进行分析, 可能会得出错误的结论

缺失的类型:

- 在对缺失数据进行处理前, 了解数据缺失的机制和形式是十分必要的. 
  将数据集中不含缺失值的变量称为 **完全变量**, 数据集中含有缺失值的变量称为不完全变量. 
- 从缺失的分布可以将缺失分为: 
   - 完全随机缺失(missing completely at random, MCAR)
      - 数据的缺失完全随机的, 不依赖任何不完全变量或完全变量, 不影响样本的无偏性. 如: 家庭地址缺失
   - 随机缺失(missing at random, MAR)
      - 数据的缺失不是完全随机的, 即该类数据的缺失依赖于其他完全变量. 如: 财务数据缺失情况与企业的大小有关
   - 完全非随机缺失(missing not at random, MNAR)
      - 数据的缺失于不完全变量自身的取值有关. 如: 高收入人群不愿意提供家庭收入

对于完全随机缺失和完全非随机缺失, 删除记录是不合适的, 
随机缺失可以通过已知变量对缺失值进行估计；非随机缺失没有很好的处理方法；

# 缺失值处理实战

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
   - `index`,\ `value`: 使用索引的实际数值插值
   - `pad`: 使用其他不为\ `NaN`\ 的值插值
   - `nearest`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `zero`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `slinear`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `quadratic`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `cubic`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `spline`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `barycentric`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
   - `polynomial`: 参考\ `scipy.iterpolate.interp1d`\ 中的插值算法
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
- `limit`: 要填充的最大连续 `NaN`\ 数量, \ :math:`>0`
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

### scipy.interpolate.XXX

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

