---
title: 时间序列处理技巧 
author: 王哲峰
date: '2023-03-06'
slug: process-utils
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

- [时区](#时区)
  - [API](#api)
  - [原理](#原理)
  - [示例](#示例)
- [重采样](#重采样)
- [缺失值填充](#缺失值填充)
- [参考](#参考)
</p></details><p></p>

# 时区

## API

* pandas.to_datetime(df.index)

## 原理

本地化是什么意思？

* 本地化意味着将给定的时区更改为目标或所需的时区。这样做不会改变数据集中的任何内容，
  只是日期和时间将显示在所选择的时区中

为什么需要它？

* 如果你拿到的时间序列数据集是 UTC 格式的，而你的客户要求你根据例如美洲时区来处理气候数据。
  你就需要在将其提供给模型之前对其进行更改，因为如果您不这样做模型将生成的结果将全部基于 UTC

如何修改？

* 只需要更改数据集的索引部分

## 示例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data
df = pd.DataFrame({
    "ts": pd.datetime_range(),
    "value": range(10),
})

# 制作日期时间类型的索引
df.index = pd.to_datetime(df.index)

# 数据集的索引部分发生变化。日期和时间和以前一样，但现在它在最后显示 +00:00
# 这意味着 pandas 现在将索引识别为 UTC 时区的时间实例
df.index = df.index.tz_localize("UTC")

# 现在可以专注于将 UTC 时区转换为我们想要的时区
df.index = df.index.tz_convert("Asia/Qatar")
```

# 重采样

```python
resampled_df = df["value"].resample("1D")  # object
resampled_df.mean()  # agg
resampled_df = resampled_df.mean().to_frame()  # 转换成 DateFrame
```

# 缺失值填充


# 参考

* [用于时间序列数据整理的Pandas函数](https://mp.weixin.qq.com/s/uy8jduqnA0tQM7qC476XSQ)

