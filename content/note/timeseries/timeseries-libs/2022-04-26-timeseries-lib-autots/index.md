---
title: AutoTS
author: 王哲峰
date: '2022-04-26'
slug: timeseries-lib-autots
categories:
  - timeseries
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

- [AutoTS 安装](#autots-安装)
- [简单示例](#简单示例)
- [数据格式](#数据格式)
- [仅支持宽数据的低阶 API](#仅支持宽数据的低阶-api)
- [快速和大数据](#快速和大数据)
  - [使用适当的模型列表，尤其是预定义的列表](#使用适当的模型列表尤其是预定义的列表)
- [参考](#参考)
</p></details><p></p>

AutoTS is a time series package for Python designed for rapidly deploying high-accuracy forecasts at scale.

* 朴素模型
* 统计模型
* 机器学习模型
* 深度学习模型

所有模型都支持预测多变量（多个时间序列）输出，还支持概率（上限/下限）预测。
大多数模型可以轻松扩展到数万甚至数十万个输入序列。许多模型还支持传入用户定义的外生回归变量

# AutoTS 安装

```bash
$ pip install autots
```

# 简单示例

```python
from autots.datasets import load_monthly
from autots import AutoTS

df_long = load_monthly(long = True)

model = AutoTS(
    forecast_length = 3,
    frequency = "inter",
    ensemble = "simple",
    max_generations = 5,
    num_validations = 2,
)

model = model.fit(
    df_long, 
    date_col = "datetime",
    value_col = "value",
    id_col = "series_id",
)

# best model info
print(model)
```



# 数据格式

* 宽格式
    - `pandas.DataFrame` with a `pandas.DatetimeIndex`
    - 每一列都是一个不同的序列
* 长格式
    - Date: `date_col`
    - Series ID: `id_col`
    - Value: `value_col`

# 仅支持宽数据的低阶 API

```python
from autots import AutoTS
from autots import load_hourly, load_daily, load_weekly, load_yearly, load_live_daily

# data
long = False
df = load_daily(long = long)


# model
model = AutoTS(
    forecast_length = 21,
    frequency = "infer",
    prediction_interval = 0.9,
    ensemble = None,
    model_list = "fast",  # "superfast", "default", "fast_parallel"
    transformer_list = "fast",  # "superfast"
    drop_most_recent = 1,
    max_generations = 4,
    num_validations = 2,
    validation_method = "backwards",
)
model = model.fit(
    df,
    date_col = "datetime" if long else None,
    value_col = "value" if long else None,
    id_col = "series_id" if long else None,
)

# prediction
prediction = model.predict()

# plot a sample
prediction.plot(
    model.df_wide_metric,
    series = model.df_wide_numeric.columns[0],
    start_date = "2019-01-01",
)

# best model
print(model)

# point forecast datetime
forecast_df = prediction.forecast
# upper and lower forecast
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# accuracy of all tried model results
model_results = model.results()

# aggregated from cross validation
validation_results = model.results("validation")
```




# 快速和大数据

## 使用适当的模型列表，尤其是预定义的列表

查看预定义列表:

```python
from autots.models.model_list import model_lists
```

* superfast: 简单的朴素模型
* fast: 更复杂但更快的模型
* fast_parallel 或 parallel: 有许多 CPU 内核可用
    - "n_jobs"
    - "auto"



# 参考

* [GitHub](https://github.com/winedarksea/AutoTS)
* [Document](https://winedarksea.github.io/AutoTS/build/html/index.html)

