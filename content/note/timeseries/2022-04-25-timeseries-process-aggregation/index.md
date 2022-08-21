---
title: 时间序列预处理-聚合
author: 王哲峰
date: '2022-04-25'
slug: timeseries-process-aggregation
categories:
  - timeseries
tags:
  - ml
---



# 时间序列降采样

# Pandas 降采样

## API

```python
pd.DataFrame.resample(rule, 
                        how=None, 
                        axis=0, 
                        fill_method=None, 
                        closed=None, 
                        label=None, 
                        convention='start', 
                        kind=None, 
                        loffset=None, 
                        limit=None, 
                        base=0, 
                        on=None, 
                        level=None)
```

- `pd.DataFrame.resample()` 和 `pd.Series.resample()`
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

## 降采样

```python
df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
                     'B': [np.nan, 2, 3, 4, 5],
                     'C': [1, 2, 1, 1, 2]}, 
                     columns=['A', 'B', 'C'])
df.groupby("A").mean()
df.groupby(["A", "B"]).mean()
df.groupby("A")["B"].mean()
```

