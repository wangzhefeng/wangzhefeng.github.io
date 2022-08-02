---
title: PySpark Pandas API on Spark
author: 王哲峰
date: '2022-07-27'
slug: spark-pyspark-pandas-api
categories:
  - spark
tags:
  - tool
---

<style>
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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

- [Options 和 setings](#options-和-setings)
  - [getting and setting options](#getting-and-setting-options)
    - [属性](#属性)
    - [方法](#方法)
- [数据对象](#数据对象)
  - [pandas-on-Spark Series](#pandas-on-spark-series)
  - [pandas-on-Spark DataFrame](#pandas-on-spark-dataframe)
  - [pandas DataFrame to pandas-on-Spark DataFrame](#pandas-dataframe-to-pandas-on-spark-dataframe)
  - [pandas DataFrame to Spark DataFrame](#pandas-dataframe-to-spark-dataframe)
  - [Spark DataFrame to pandas-on-Spark DataFrame](#spark-dataframe-to-pandas-on-spark-dataframe)
  - [dtypes](#dtypes)
  - [head](#head)
  - [index](#index)
  - [columns](#columns)
  - [to_numpy](#to_numpy)
  - [describe](#describe)
  - [T](#t)
  - [sort_index](#sort_index)
  - [sort_values](#sort_values)
- [缺失数据](#缺失数据)
  - [drop rows with missing data](#drop-rows-with-missing-data)
  - [fill missing data](#fill-missing-data)
- [数据操作](#数据操作)
  - [Stats](#stats)
  - [Spark Configurations](#spark-configurations)
  - [transform and apply a function](#transform-and-apply-a-function)
- [Grouping](#grouping)
- [Plotting](#plotting)
  - [pandas-on-Spark Series](#pandas-on-spark-series-1)
  - [pandas-on-Spark DataFrame](#pandas-on-spark-dataframe-1)
- [Getting data in/out](#getting-data-inout)
  - [CSV](#csv)
  - [Parquet](#parquet)
  - [Spark IO](#spark-io)
- [Type support and hints](#type-support-and-hints)
- [最佳实践](#最佳实践)
- [支持的 pandas API](#支持的-pandas-api)
</p></details><p></p>

```python
import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
```

# Options 和 setings

属性:

* `pyspark.pandas.options.display.max_rows`
* `pyspark.pandas.options.display.max_columns`

方法:

* `get_option("option_name")`
* `set_option("option_name", new_value)`
* `reset_option()`
* `option_context("option_name", option_value, "option_name", option_value, ...)`

API 均在 `pandas_on_spark` 命名空间有效

## getting and setting options

### 属性

```python
>>> import pyspark.pandas as ps
>>> ps.options.display.max_rows
1000
>>> ps.options.display.max_rows = 10
>>> ps.options.display.max_rows
10
```

### 方法

```python
>>> import pyspark.pandas as pd
>>> ps.get_option("display.max_rows")
1000
>>> ps.set_option("display.max_rows", 10)
>>> ps.get_option("display.max_rows")
10
```




# 数据对象

## pandas-on-Spark Series

```pyhton
psser = ps.Series([1, 3, 5, np.nan, 6, 8])
psser
```

## pandas-on-Spark DataFrame

```python
psdf = ps.DataFrame({
    "a": [1, 2, 3, 4, 5, 6],
    "b": [100, 200, 300, 400, 500, 600],
    "c": ["one", "two", "three", "four", "five", "six"]
}, index = [10, 20, 30, 40, 50, 60])
psdf
```

## pandas DataFrame to pandas-on-Spark DataFrame

```python
dates = pd.date_range("20130101", periods = 6)
pdf = pd.DataFrame(
    np.random.randn(6, 4), 
    index = dates,
    columns = list("ABCD"),
)
pdf
```

```python
psdf = ps.from_pandas(pdf)
type(psdf)
psdf
```

## pandas DataFrame to Spark DataFrame

```python
spark = SparkSession.builder.getOrCreate()
sdf = spark.createDataFrame(pdf)
sdf.show()
```

## Spark DataFrame to pandas-on-Spark DataFrame

```python
psdf = sdf.pandas_api()
psdf
```

## dtypes

```python
psdf.dtypes
```

## head

```python
psdf.head()
```

## index

```python
psdf.index
```

## columns

```python
psdf.columns
```

## to_numpy

```python
psdf.to_numpy()
```

## describe

```python
psdf.describe()
```

## T

```python
psdf.T
```

## sort_index


```python
psdf.sort_index(ascending = False)
```

## sort_values

```python
psdf.sort_values(by = "B")
```

# 缺失数据

```python
pdf1 = pdf.reindex(index = dates[0:4], columns = list(pdf.columns) + ["E"])
pdf1.loc[dates[0]:dates[1], "E"] = 1
psdf1 = ps.from_pandas(pdf1)
psdf1
```

## drop rows with missing data

```python
psdf1.dropna(how = "any")
```

## fill missing data

```python
psdf1.fillna(value = 5)
```




# 数据操作

## Stats

```python
psdf.mean()
```

## Spark Configurations

* Arrow optimization

```python
from pyspark.sql import SparkSession
import pyspark.pandas as ps

spark = SparkSession.builder.getOrCreate()

prev = spark.conf.get("spark.sql.execution.arrow.pyspark.enable")
ps.set_option("compute.default_index_type", "distributed")

import warnings
warnings.filterwarnings("ignore")
```

```python
spark.conf.set("spark.sql.execution.arrow.pyspark.enable", True)
%timeit ps.range(300000).to_pandas()
```

```python
spark.conf.set("spark.sql.execution.arrow.pyspark.enable", True)
%timeit ps.range(300000).to_pandas()
```


```python
ps.reset_option("compute.default_index_type")
spark.conf.set("spark.sql.execution.arrow.pyspark.enable", prev)
```

## transform and apply a function






# Grouping

* split-apply-combine

```python
from pyspark.pandas as ps

psdf = ps.DataFrame({
    "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
    "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
    "C": np.random.randn(8),
    "D": np.random.randn(8),
})
psdf.groupby("A").sum()
psdf.groupby(["A", "B"]).sum()
```

# Plotting

## pandas-on-Spark Series

```python
pser = pd.Series(
    np.random.randn(1000),
    index = pd.date_range("1/1/2000", periods = 10000)
)
psser = ps.Series(pser)
psser = psser.cummax()
psser.plot()
```

## pandas-on-Spark DataFrame

```python
pdf = pd.DataFrame(
    np.random.randn(1000, 4),
    index = pser.index,
    columns = ["A", "B", "C", "D"]
)
psdf = ps.from_pandas(pdf)
psdf = psdf.cummax()
psdf.plot()
```

# Getting data in/out

## CSV

```python
psdf.to_csv("foo.csv")
ps.read_csv("foo.csv").head(10)
```

## Parquet

```python
psdf.to_parquet("bar.parquet")
ps.read_parquet("bar.parquet").head(10)
```


## Spark IO

```python
psdf.to_spark_io("zoo.orc", format = "orc")
ps.read_spark_io("zoo.orc", format = "orc").head(10)
```


# Type support and hints





# 最佳实践


# 支持的 pandas API


