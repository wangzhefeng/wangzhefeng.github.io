---
title: Lib tsfresh
subtitle: 特征工程
author: 王哲峰
date: '2022-05-03'
slug: lib-tsfresh
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

- [tsfresh 安装](#tsfresh-安装)
- [tsfresh 使用步骤](#tsfresh-使用步骤)
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
- [参考](#参考)
</p></details><p></p>


tsfresh 是一个自动化提取时序特征的库

# tsfresh 安装

```bash
$ pip install tsfresh
```

# tsfresh 使用步骤

使用tsfresh的使用步骤如下：

前期训练阶段：

1. 数据准备：准备符合 tsfresh 输入格式的数据集
2. 样本抽样：以步长 s 为间隔滑窗抽样
3. 特征生成：对采样样本生成特征，并收集它们
4. 特征选择：收集多个特征下的衍生特征，进行特征选择


后期部署阶段：

1. 数据准备：准备符合 tsfresh 输入格式的数据集
2. 特征选择：对滑窗样本生成特征，并收集它们

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



# 参考

* tsfresh Github：https://github.com/blue-yonder/tsfresh‍
* tsfresh 文档：https://tsfresh.readthedocs.io
* [tsfresh 使用小结](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247487150&idx=1&sn=d070e11282e3bd1f9656e6be656d16e0&chksm=fa0410c5cd7399d3b1acec6d379c659d1344cf52232206e86c6d5867ba54c77062ca3581e06a&scene=178&cur_album_id=1577157748566310916#rd)

