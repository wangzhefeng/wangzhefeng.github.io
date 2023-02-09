---
title: Lib tsfresh 时间序列特征工程
author: 王哲峰
date: '2022-05-03'
slug: timeseries-lib-tsfresh
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

- [时间序列特征工程](#时间序列特征工程)
- [tsfresh 安装](#tsfresh-安装)
- [tsfresh 数据格式](#tsfresh-数据格式)
  - [输入数据格式](#输入数据格式)
    - [Flat DataFrame](#flat-dataframe)
    - [Stacked DataFrame](#stacked-dataframe)
    - [Dictionary of flat DataFrame](#dictionary-of-flat-dataframe)
  - [输出数据格式](#输出数据格式)
- [scikit-learn Transformers](#scikit-learn-transformers)
  - [Feature extraction](#feature-extraction)
  - [Feature selection](#feature-selection)
    - [Feature extraction and selection](#feature-extraction-and-selection)
- [大数据](#大数据)
  - [Dask](#dask)
  - [PySpark](#pyspark)
- [Rolling 和 时间序列预测](#rolling-和-时间序列预测)
</p></details><p></p>

# 时间序列特征工程

* [时间序列特征工程](../../feature_engine/2022-09-13-feature-engine-type-timeseries/index.md)

# tsfresh 安装

```bash
$ pip install tsfresh
```

# tsfresh 数据格式

## 输入数据格式 

* Flat DataFrame
* Stacked DataFrame
* dictionary of flat DataFrame

| column_id | column_value | column_sort | column_kind |
|-----------|--------------|-------------|-------------|
| id        | value        | sort        | kind        |


适合的 API:

* `tsfresh.extract_features()`
* `tsfresh.`

### Flat DataFrame

| id       | time     | x        | y        |          |          |          |          |          |    | A  | t1  | x(A, t1) | y(A, t1) |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----|----|----|----|----|
| A        | t2       | x(A, t2) | y(A, t2) | A        | t3       | x(A, t3) | y(A, t3) | B        | t1 |    |     |          |          |
| x(B, t1) | y(B, t1) | B        | t2       | x(B, t2) | y(B, t2) | B        | t3       | x(B, t3) |    |    |     |          |          |
| y(B, t3) |          |          |          |          |          |          |          |          |    |    |     |          |          |

### Stacked DataFrame

| id | time | kind | value    |   |    |   |          |   |   | A | t1       | x | x(A, t1) |
|----|------|------|----------|---|----|---|----------|---|---|---|----------|---|----------|
| A  | t2   | x    | x(A, t2) | A | t3 | x | x(A, t3) | A |t1 |y  | y(A, t1) |   |          |
| A  | t2   | y    | y(A, t2) | A | t3 | y | y(A, t3) | B |t1 |x  | x(B, t1) |   |          |
| B  | t2   | x    | x(B, t2) | B | t3 | x | x(B, t3) | B |t1 |y  | y(B, t1) |   |          |
| B  | t2   | y    | y(B, t2) | B | t3 | y | y(B, t3) |   |   |   |          |   |          |

### Dictionary of flat DataFrame

```
{ 
    "x”:
        | id | time | value    |
        |----|------|----------|
        | A  | t1   | x(A, t1) |
        | A  | t2   | x(A, t2) |
        | A  | t3   | x(A, t3) |
        | B  | t1   | x(B, t1) |
        | B  | t2   | x(B, t2) |
        | B  | t3   | x(B, t3) |
  , "y”:
        | id | time | value    |
        |----|------|----------|
        | A  | t1   | y(A, t1) |
        | A  | t2   | y(A, t2) |
        | A  | t3   | y(A, t3) |
        | B  | t1   | y(B, t1) |
        | B  | t2   | y(B, t2) |
        | B  | t3   | y(B, t3) |
}
```

## 输出数据格式

| id | x feature 1 | … | x feature N | y feature 1 | `$\ldots$` | y feature N |
|----|-----------------|---|-----------------|-----------------|---|-----------------|
| A  | …               | … | …               | …               | … | …               |
| B  | …               | … | …               | …               | … | …               |


# scikit-learn Transformers

## Feature extraction

* `tsfresh.FeatureAugmenter`

## Feature selection

* `tsfresh.FeatureSelector`

### Feature extraction and selection

* `tsfresh.RelevantFeatureAugmenter`

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd

# download data
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
download_robot_execution_failures()

pipeline = Pipeline([
    ("augmenter", RelevantFeatureAugmenter(column_id = "id", column_sort = "time")),
    ("classifier", RandomForestClassifier()),
])

df_ts, y = load_robot_execution_failures()
X = pd.DataFrame(index = y.index)

pipeline.set_params(augmenter__timeseries_container = df_ts)
pipeline.fit(X, y)
```


# 大数据

## Dask


## PySpark

# Rolling 和 时间序列预测


![img](images/rolling_mechanism_1.png)
