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
  - [getting 和 setting options](#getting-和-setting-options)
    - [属性](#属性)
    - [方法](#方法)
  - [不同 DataFrame 上的数据操作](#不同-dataframe-上的数据操作)
  - [默认 Index 类型](#默认-index-类型)
    - [sequence](#sequence)
    - [distributed-sequence(default)](#distributed-sequencedefault)
    - [distributed](#distributed)
  - [可用 options](#可用-options)
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
  - [Type Support](#type-support)
    - [PySpark & Pandas API on Spark](#pyspark--pandas-api-on-spark)
    - [pandas & pandas API on Spark](#pandas--pandas-api-on-spark)
    - [Type mapping](#type-mapping)
  - [Type Hints](#type-hints)
- [From/to others DBMSes](#fromto-others-dbmses)
  - [读写 DataFrame](#读写-dataframe)
    - [SQLLite](#sqllite)
  - [PostgreSQL](#postgresql)
- [最佳实践](#最佳实践)
  - [利用 PySpark API](#利用-pyspark-api)
    - [示例 1](#示例-1)
    - [示例 2](#示例-2)
    - [其他 Spark 功能](#其他-spark-功能)
  - [检查执行计划](#检查执行计划)
  - [使用 checkpoint](#使用-checkpoint)
  - [避免 shuffling](#避免-shuffling)
  - [避免在单个分区上计算](#避免在单个分区上计算)
  - [避免使用保留列名](#避免使用保留列名)
  - [不要使用重复列名](#不要使用重复列名)
  - [在从 Saprk DataFrame 转换为 pandas-on-Spark DataFrame 时明确指定索引列名](#在从-saprk-dataframe-转换为-pandas-on-spark-dataframe-时明确指定索引列名)
  - [使用 distributed 或者 distributed-sequence 默认索引](#使用-distributed-或者-distributed-sequence-默认索引)
  - [减少在不同 DataFrame/Seris 上的数据操作](#减少在不同-dataframeseris-上的数据操作)
  - [尽可能直接使用 Pandas API on Spark](#尽可能直接使用-pandas-api-on-spark)
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

## getting 和 setting options

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

* get_option() & set_option()

```python
>>> import pyspark.pandas as pd
>>> ps.get_option("display.max_rows")
1000
>>> ps.set_option("display.max_rows", 10)
>>> ps.get_option("display.max_rows")
10
```

* reset_option()

```python
>>> import pyspark.pandas as ps
>>> ps.get_option("compute.max_rows")
1000
>>> ps.set_option("compute.max_rows", 2000)
>>> ps.get_option("compute.max_rows")
2000

>>> ps.reset_option("compute.max_rows")
>>> ps.get_option("compute.max_rows")
1000
```

* option_context()

```python
>>> with ps.option_context("display.max_rows", 10, "compute.max_rows", 5):
...    print(ps.get_option("display.max_rows"))
...    print(ps.get_option("compute.max_rows"))
10
5

>>> print(ps.get_option("display.max_rows"))
>>> print(ps.get_option("compute.max_rows"))
1000
1000
```

## 不同 DataFrame 上的数据操作

为了防止高资源消耗，Pandas API on Spark 默认不允许在不同 DataFrame 或 Series 上的操作。
可以通过设置 `compute.ops_on_diff_frames` 为 `True` 启用操作

```python
>>> import pyspark.pandas as ps

>>> ps.set_option("compute.ops_on_diff_frames", True)
>>> psdf1 = ps.range(5)
>>> psdf2 = ps.DataFrame({
...     "id": [5, 4, 3],
... })
>>> (psdf1 - psdf2).sort_index()
    id
0 -5.0
1 -3.0
2 -1.0
3  NaN
4  NaN
>>> ps.reset_option("compute.ops_on_diff_frames")
```

```python
import pyspark.pandas as ps

>>> ps.set_option("compute.ops_on_diff_frames", True)
>>> psdf = ps.range(5)
>>> psser_a = ps.Series([1, 2, 3, 4])
>>> psdf["new_col"] = psser_a
>>> psdf
   id  new_col
0   0      1.0
1   1      2.0
3   3      4.0
2   2      3.0
4   4      NaN
>>> ps.reset_option("compute.ops_on_diff_frames")
```

## 默认 Index 类型

可以通过 `compute.default_index_type` 配置索引类型

### sequence

`sequence` 通过 PySpark 的 Window 函数实现了一个递增序列，而无需指定分区。
因此，它可以在单个节点中以整个分区完成。当数据很大时，应该避免使用这种索引类型

```python
>>> import pyspark.pandas as ps
>>> 
>>> ps.set_option("compute.default_index_index", "sequence")
>>> psdf = ps.range(3)
>>> ps.reset_option("compute.default_index_type")
>>> psdf.index
Int64Index([0, 1, 2], dtype='int64')
```


### distributed-sequence(default)


### distributed


## 可用 options

| 配置型 | 默认值 | 说明 |
|----|----|----|
| display.max_rows | 1000 | |
| compute.max_rows | 1000 | |
| compute.shorcut_limit | 1000 | |
| compute.ops_on_diff_frames | False | |
| compute.default_index_type | 'distributed-sequence' | |
| compute.ordered_head | False | |
| compute.eager_check | True | |
| compute.is_in_limit | 80 | |
| plotting.max_rows | 1000 | |
| plotting.sample_ration | None | |
| plotting.backend | 'plotly' |  |

# 数据对象

## pandas-on-Spark Series

```python
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

## Type Support


### PySpark & Pandas API on Spark


### pandas & pandas API on Spark



### Type mapping




* `pyspark.pandas.typedef.as_spark_type`

```python
>>> import typing
>>> import numpy as np
>>> from pyspark.pandas.typedef import as_spark_type

>>> as_spark_type(int)
LongType

>>> as_spark_type(np.int32)
IntegerType

>>> as_spark_type(typing.List[float])
ArrayType(DoubleType, true)
```

## Type Hints


# From/to others DBMSes

Pandas API on Spark 中与其他 DBMS 交互的 API 与 pandas 中的 API 略有不同，
因为 Pandas API on Spark 利用 PySpark 中的 JDBC API 来读取和写入其他 DBMS

读写外部 DBMS 的 API 如下:

| API | 说明 |
|----|----|
| read_sql_table(table_name, con[, schema, ...]) | 将 SQL 数据库表读入 DataFrame |
| read_sql_query(sql, con[, index_col]) | 将 SQL 查询读入 DataFrame |
| read_sql(sql, con[, index_col, columns]) | 将 SQL 查询或数据库表读入 DataFrame |

Pandas on Spark 需要一个规范的 JBDC URL `con`，
并且能够为 PySpark JDBC API 中的配置项采用额外的关键字参数:

```python
ps.read_sql(..., dbtable = "...", driver = "", keytab = "", ...)
```

## 读写 DataFrame

### SQLLite

1. 创建数据库

```python
import sqlite3

# 数据库连接
con = sqlite.connect("example.db")

# 游标
cur = con.cursor()
```

2. 创建表

```python
# create table
cur.execute(
    '''
    CREATE TABLE stocks
    (date text, trans text, symbol text, qty real, price reql)
    '''
)

# insert a row of data
cur.execute(
    '''
    INSERT INTO stocks 
    VALUES ('2006-01-05', 'BUY', 'RHAT', 100, 35.14)
    '''
)

# save(commit) changes
con.commit()
con.close()
```

3. 安装 JDBC 驱动程序

Pandas API on Spark 需要读取 JDBC 驱动程序
因此它需要特定数据库的驱动程序位于 Spark 的类路径中。
SQLite JDBC 驱动可以通过下面的方式下载

```bash
$ curl -O https://repo1.maven.org/maven2/org/xerial/sqlite-jdbc/3.34.0/sqlite-jdbc-3.34.0.jar
```

4. 添加 JDBC 驱动程序到 `SparkSession` 中
  
添加 JDBC 驱动程序到 `SparkSession` 后，Pandas API on Spark 将自动检测 `SparkSession` 并使用

```python
import os
from pyspark.sql import SparkSession

(SparkSession.builder
    .master("local")
    .appName("SQLite JDBC")
    .config(
        "spark.jar",
        "{}/sqlite-jdbc-3.34.0.jar".format(os.getcwd())
    ),
    .config(
        "spark.driver.extraClassPath",
        "{/sqlite-jdbc-3.34.0.jar}".format(os.getcwd())
    ),
    getOrCreate()
)
```

5. 读取表

```python
import pyspark.pandas as ps

df = pd.read_sql(
    "stocks", 
    con = f"jdbc:sqlite:{os.getcwd()}/example.db"
)
df
```

```
         date trans symbol    qty  price
0  2006-01-05   BUY   RHAT  100.0  35.14
```

6. 写入表

```python
import pyspark.pandas as ps
df.price += 1
df.spark.to_spark_io(
    format = "jdbc",
    mode = "append",
    dbtable = "stocks",
    url = f"jdbc:sqlite:{os.getcwd()}/example.db"
)
```

查看写入的数据:

```python
ps.read_sql(
    "stocks", 
    con = f"jdbc:sqlite:{os.getcwd()}/example.db"
)
```

```
         date trans symbol    qty  price
0  2006-01-05   BUY   RHAT  100.0  35.14
1  2006-01-05   BUY   RHAT  100.0  36.14
```

## PostgreSQL

```bash
$ ./bin/spark-shell --driver-class-path postgresql-8.4.1207.jar --jars postgresql-9.4.1207.jar
```


# 最佳实践

## 利用 PySpark API

Pandas API on Spark 在后台使用 Spark；
因此，Spark 的许多功能和性能优化 都可以在 Pandas API on Spark 中使用。
要充分将 Pandas API on Spark 和这些高阶功能结合起来

* 现有的 Spark context 和 session 在 Pandas API on Spark 中开箱即用。
  如果已经有自己配置的 Spark context 或正在运行的 session，
  Pandas API on Spark 会自动使用它们

* 如果环境(例如普通的 Python 解释器)中没有运行 Spark context 或 session，
  则可以手动配置 `SparkContext` 和/或 `SparkSession`. 
  创建 Spark context 和/或 session 后，
  Pandas API on Spark 可以自动使用 context 和/或 session。

### 示例 1 

* 在 Spark 中配置执行器内存

```python
from pyspark import SparkConf, SparkContext

conf = SparkConf()
conf.set("spark.executor.memory", "2g")
SparkContext(conf = conf)

import pyspark.pandas as ps
...
```

### 示例 2

* 配置 PySpark 中的 Arrow 优化

```python
from pyspark.sql import SparkSession

# Pandas API on Spark automatically uses this Spark session with the configurations set.
builder = SparkSession.builder
    .appName("pandas-on-spark")
    .config("spark.sql.execution.arrow.pyspark.enable", "true")    
    .getOrCreate()

import pyspark.pandas as ps
...
```

### 其他 Spark 功能

* [tune spark](https://spark.apache.org/docs/latest/tuning.html)

## 检查执行计划



## 使用 checkpoint



## 避免 shuffling


## 避免在单个分区上计算


## 避免使用保留列名


## 不要使用重复列名


## 在从 Saprk DataFrame 转换为 pandas-on-Spark DataFrame 时明确指定索引列名




## 使用 distributed 或者 distributed-sequence 默认索引

## 减少在不同 DataFrame/Seris 上的数据操作

## 尽可能直接使用 Pandas API on Spark

# 支持的 pandas API


