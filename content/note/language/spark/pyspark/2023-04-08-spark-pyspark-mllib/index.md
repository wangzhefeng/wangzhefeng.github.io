---
title: MLlib
subtitle: Machine Learning Library
author: 王哲峰
date: '2023-04-08'
slug: spark-pyspark-mllib
categories:
  - machinelearning
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [MLlib 介绍](#mllib-介绍)
  - [MLlib 简介](#mllib-简介)
  - [MLlib API](#mllib-api)
- [基础统计](#基础统计)
  - [相关性计算](#相关性计算)
  - [假设检验](#假设检验)
  - [汇总](#汇总)
- [数据源](#数据源)
  - [Image](#image)
  - [LIBSVM](#libsvm)
- [ML Pipelines](#ml-pipelines)
  - [DataFrame](#dataframe)
  - [Pipeline](#pipeline)
    - [Transformers](#transformers)
    - [Estimator](#estimator)
    - [Properties](#properties)
  - [Parameters](#parameters)
  - [持久化](#持久化)
- [特征工程](#特征工程)
  - [特征抽取](#特征抽取)
  - [特征转换](#特征转换)
  - [特征选择](#特征选择)
- [模型与算法](#模型与算法)
  - [分类](#分类)
  - [回归](#回归)
  - [聚类](#聚类)
  - [协同过滤](#协同过滤)
- [模型选择和调参](#模型选择和调参)
</p></details><p></p>

# MLlib 介绍

## MLlib 简介

MLlib 是 Spark 的机器学习库，目标是使实用的机器学习具有可扩展性和简单性。MLlib 提供了以下工具：

* ML 算法
    - 回归
    - 分类
    - 聚类
    - 协同过滤
* 特征化(Featurization)
    - 特征提取(feature extraction)
    - 转换(transformation)
    - 降维(dimensionality reduction)
    - 特征选择(selection)
* 管道(Pipelines)
    - 构建、评估、调参
* 持久化(Persistence)
    - 保存模型、算法、管道
    - 加载模型、算法、管道
* 工具(Utilities)
    - 线性代数
    - 统计学
    - 数据处理

## MLlib API

基于 DataFrame 的 API 是 MLlib 的主要 API

```
spark.ml
```

# 基础统计

## 相关性计算

* Pearson 相关系数
* Spearman 相关系数

```python
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

data = [
    (Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
    (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
    (Vecotrs.dense([6.0, 7.0, 0.0, 8.0]),),
    (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)
]
df = spark.createDataFrame(data, ["features"])

# 
r1 = Correlation.corr(df, "features").head()
print(f"Pearson correlation matrix:\n {str(r1[0])}")

# Spearman
r2 = Correlation.corr(df, "features", "spearman").head()
print(f"Spearman correlation matrix:\n {str(r2[0])}")
```

## 假设检验

* 卡方检验，Pearson Chi-squared(`$\chi^{2}$`)

```python
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

data = [
    (0.0, Vectors.dense(0.5, 10.0)),
    (0.0, Vectors.dense(1.5, 20.0)),
    (1.0, Vectors.dense(1.5, 30.0)),
    (0.0, Vectors.dense(3.5, 30.0)),
    (0.0, Vectors.dense(3.5, 40.0)),
    (1.0, Vectors.dense(3.5, 40.0))
]
df = spark.createDataFrame(data, ["label", "features"])

# chi square test
r = ChiSquareTest.test(df, "features", "labels").head()

print(f"pValue: {str(r.pValues)}")
print(f"degreesOfFreedom: {str(r.degreesOfFreedom)}")
print(f"statistics: {str(r.statistics)}")
```

## 汇总

```python
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Summarizer

sc = SparkContext(master = "local[2]")

df = sc.parallelize([
    Row(weight = 1.0, features = Vectors.dense(1.0, 1.0, 1.0)),
    Row(weight = 0.0, features = Vectors.dense(1.0, 2.0, 3.0))
]).toDF()

# summarizer
summarizer = Summarizer.metrics("mean", "count")

df.select(summarizer.summary(df.features, df.weight)).show(truncate = False)
df.select(summarizer.summary(df.features)).show(truncate = False)
df.select(Summarizer.mean(df.features, df.weight)).show(truncate = False)
df.select(Summarizer.mean(df.features)).show(truncate = False)
```

# 数据源

* Parquet
* CSV
* JSON
* JDBC
* Image
* LIBSVM

## Image

图像数据格式：

* jpeg
* png
* ...

图像数据 DataFrame 有一个 `StructType` 列 `image`，
包含存储为图像模式(images schema)的图像数据，`image` 列 schema 如下：

* origin：`StringType`
    - 图像的文件路径
* height：`IntegerType`
    - 高度
* width：`IntegerType`
    - 宽度
* nChannels：`IntegerType`
    - 通道
* mode：`IntegerType`
    - OpenCV 兼容类型(type)
* data：`BinaryType`
    - OpenCV 兼容的图像字节(image bytes)顺序，在大多数情况下为 BGR

```python
from pyspark.sql import SparkSession

spark = SparkSession(master = "local[1]")

df = spark \
    .read \
    .format("image") \
    .option("dropInvalid", True) \
    .load("data/mllib/images/origin/kittens")
df \
    .select("image.origin", "image.width", "image.height") \
    .show(truncate = False)
```

## LIBSVM

加载的 LIBSVM DataFrame 有两列：

* label：`DoubleType`
    - 代表实例标签
* features：`VectorUDT`
    - 代表特征向量

```python
from pyspark.sql import SparkSession

spark = SparkSession(master = "local[1]")

df = spark \
    .read \
    .format("libsvm") \
    .option("numFeatures", "780") \
    .load("data/mllib/sample_libsvm_data.txt")
df.show(10)
```

# ML Pipelines

ML Pipelines 提供了一组统一的高级 API，这些 API 构建在 DataFrame 之上，
可帮助用户创建和调整实用的机器学习管道

* DataFrame
* Transformer
* Estimator
* Pipeline
* Parameter

## DataFrame

数据类型：

* vector: `pyspark.ml.linalg.Vector`
* text
* image
* structured data

```python
# vector
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import (
    # int
    ByteType, ShortType, IntegerType, LongType,
    # float
    FloatType, DoubleType, DecimalType,
    # string
    StringType,
    BinaryType,
    BooleanType,
    # datetime
    TimestampType, DateType, DayTimeIntervalType,
    # array, map
    ArrayType, MapType,
    # struct data
    StructType, StructField,
)
```

## Pipeline

### Transformers

* transform()

### Estimator

* fit()

### Properties

* Transformer.transform()
* Estimator.fit()

## Parameters

## 持久化

# 特征工程

## 特征抽取


## 特征转换

## 特征选择


# 模型与算法

## 分类



## 回归

## 聚类

## 协同过滤

# 模型选择和调参
