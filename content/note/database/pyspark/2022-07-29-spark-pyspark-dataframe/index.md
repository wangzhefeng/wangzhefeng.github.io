---
title: PySpark DataFrame
author: 王哲峰
date: '2022-07-29'
slug: spark-pyspark-dataframe
categories:
  - spark
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

- [必备知识](#必备知识)
- [初始化 SparkSession](#初始化-sparksession)
  - [PySpark App](#pyspark-app)
  - [PySpark Shell](#pyspark-shell)
- [DataFrame 创建](#dataframe-创建)
  - [List of List](#list-of-list)
  - [List of Tuple](#list-of-tuple)
  - [pandas DataFrame](#pandas-dataframe)
  - [Row](#row)
  - [RDD](#rdd)
  - [查看数据](#查看数据)
- [DataFrame 查看](#dataframe-查看)
  - [横向查看](#横向查看)
  - [纵向查看](#纵向查看)
  - [即时产看](#即时产看)
  - [设置查看的最大行数](#设置查看的最大行数)
  - [Schema 和列名](#schema-和列名)
  - [Summary](#summary)
  - [收集分布式数据集到本地内存](#收集分布式数据集到本地内存)
  - [收集部分分布式数据集到本地内存](#收集部分分布式数据集到本地内存)
  - [转换为 pandas.DataFrame](#转换为-pandasdataframe)
- [DataFrame 筛选](#dataframe-筛选)
  - [选择列](#选择列)
    - [select](#select)
    - [withColumn](#withcolumn)
  - [选择行](#选择行)
- [DataFrame Apply](#dataframe-apply)
  - [pandas\_udf](#pandas_udf)
  - [mapInPandas](#mapinpandas)
- [DataFrame Grouping](#dataframe-grouping)
  - [groupby](#groupby)
  - [applyInPandas](#applyinpandas)
  - [cogroup](#cogroup)
- [数据读写](#数据读写)
  - [CSV](#csv)
  - [Parquet](#parquet)
  - [ORC](#orc)
- [DataFrame 和 Spark SQL](#dataframe-和-spark-sql)
  - [DataFrame 注册为一个表格并运行 SQL](#dataframe-注册为一个表格并运行-sql)
  - [UDF 可以开箱即用地在 SQL 中注册和调用](#udf-可以开箱即用地在-sql-中注册和调用)
  - [SQL 表达式混入作用 PySpark 列](#sql-表达式混入作用-pyspark-列)
- [参考](#参考)
</p></details><p></p>


# 必备知识

* Spark
    - RDD
    - transforms
    - actions
* Python
    - pyspark
        - DataFrame
    - pandas
        - DataFrame
* SQL

# 初始化 SparkSession

## PySpark App

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

## PySpark Shell

```python
$ pyspark
```

![images](images/pyspark_shell.jpg)

其中包含的信息：

* Python Version：3.7.10
* Spark context Web UI：[http://192.168.1.8:4040](http://192.168.1.8:4040)
* Spark context：`sc`(`master = local[*], app id = local-1659113580876`)
* SparkSession：`spark`

# DataFrame 创建

```python
from datetime import datetime, date
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

相关 API：

* `spark.createDataFrame(, schema)`
* `Row()`
* `spark.sparkContext.parallelize()`

## List of List

```python
df = spark.createDataFrame(
    [[1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)],
     [2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)],
     [3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0)]], 
    schema = "a long, b double, c string, d date, e timestamp"
)
df
```

## List of Tuple

```python
df = spark.createDataFrame(
    [(1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
     (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
     (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))], 
     schema = "a long, b double, c string, d date, e timestamp"
)
df
```

## pandas DataFrame

```python
# pandas DataFrame
import pandas as pd
pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})

# pyspark DataFrame
df = spark.createDataFrame(pandas_df)
df
```

## Row

> List of pyspark.sql.Row

```python
from pyspark.sql import Row

df = spark.createDataFrame([
    Row(a = 1, b = 2., c = "string1", d = date(2000, 1, 1), e = datetime(2000, 1, 1, 12, 0)),
    Row(a = 2, b = 3., c = "String2", d = date(2000, 2, 1), e = datetime(2000, 1, 2, 12, 0)),
    Row(a = 4, b = 5., c = "String3", d = date(2000, 3, 1), e = datetime(2000, 1, 3, 12, 0)),
])
df
```

## RDD

> RDD of list of tuples

```python
rdd = spark.sparkContext.parallelize([
    (1, 2., "string1", date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., "string2", date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., "string3", date(2000, 3, 1), datetime(2000, 1, 3, 12, 0)),
])

df = spark.createDataFrame(
    rdd,
    schema = ["a", "b", "c", "d", "e"]
)
```

## 查看数据

```python
df.show()
```

```
+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
|  2|3.0|string2|2000-02-01|2000-01-02 12:00:00|
|  3|4.0|string3|2000-03-01|2000-01-03 12:00:00|
+---+---+-------+----------+-------------------+
```

```python
df.printSchema()
```

```
root
 |-- a: long (nullable = true)
 |-- b: double (nullable = true)
 |-- c: string (nullable = true)
 |-- d: date (nullable = true)
 |-- e: timestamp (nullable = true)
```

# DataFrame 查看

相关 API：

* `DataFrame.show()`
* `spark.conf.set()`
* `DataFrame.printSchema()`
* `DataFrame.columns`
* `DataFrame.select().describe().show()`
* `DataFrame.collect()`
* `DataFrame.take()`
* `DataFrame.tail()`
* `DataFrame.toPandas()`

## 横向查看

```python
df.show(1)
```

## 纵向查看

```python
df.show(1, vertical = True)
```

## 即时产看

```python
spark.conf.set("spark.sql.repl.eagerEval.enable", True)
df
```

## 设置查看的最大行数

```python
spark.config.set("spark.sql.repl.eagerEval.maxNumRows", 10)
df
```

## Schema 和列名

```python
df.printSchema()
df.columns
```

## Summary

```python
df.select("a", "b", "c").describe().show()
```

## 收集分布式数据集到本地内存

* 注意内存溢出

```python
df.collect()
```

## 收集部分分布式数据集到本地内存

* 避免内存溢出

```python
df.take(1)
df.tail()
```

## 转换为 pandas.DataFrame

* 收集分布式数据集到本地内存, 注意内存溢出

```python
df.toPandas()
```

# DataFrame 筛选

相关 API：

* `DataFrame.column_name`
* `pyspark.sql.Column`
* `pyspark.sql.functions.uper`
* `DataFrame.column_name.isNull()`
* `DataFrame.select(DataFrame.column_name).show()`
* `DataFrame.withColumn("", DataFrame.column).show()`
* `DataFrame.filter(condition).show()`

## 选择列

PySpark DataFrame 是惰性计算的，所以选择一个列不会马上计算，而只会返回一个 `Column` 实例：

```python
df.a
```

```
Column<b'a'>
```

大部分列操作都会返回 `Column`：

```python
from pyspark.sql import Column
from pyspark.sql.functions import upper

type(df.c) == type(upper(df.c)) == type(df.c.isNull())
```

```
True
```

### select

```python
df.select(df.c).show()
```

```
+-------+
|      c|
+-------+
|string1|
|string2|
|string3|
+-------+
```

### withColumn

* 分配一个新 `Column` 实例

```python
df.withColumn("upper_c", upper(df.c)).show()
```

```
+---+---+-------+----------+-------------------+-------+
|  a|  b|      c|         d|                  e|upper_c|
+---+---+-------+----------+-------------------+-------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|STRING1|
|  2|3.0|string2|2000-02-01|2000-01-02 12:00:00|STRING2|
|  3|4.0|string3|2000-03-01|2000-01-03 12:00:00|STRING3|
+---+---+-------+----------+-------------------+-------+
```

## 选择行

```python
df.filter(df.a == 1).show()
```

```
+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+
```

# DataFrame Apply

相关 API：

* `pyspark.sql.functions.pandas_udf()`
* `DataFrame.mapInPandas()`

## pandas_udf

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf


@pandas_udf("long")
def pandas_plus_one(series: pd.Series) -> pd.Series:
    """
    Simply plus one by using pandas Series.
    """
    return series + 1

df.select(pandas_plus_one(df.a)).show()
```

```
+------------------+
|pandas_plus_one(a)|
+------------------+
|                 2|
|                 3|
|                 4|
+------------------+
```

## mapInPandas

```python
def pandas_filter_func(iterator):
    for pandas_df in iterator:
        yield pandas_df[pandas_df.a == 1]

df.mapInPandas(pandas_filter_func, schema = df.schema).show()
```

# DataFrame Grouping

> `split-apply-combine`

相关 API：

* `DataFrame.groupby().func().show()`
* `DataFrame.groupby().applyInPandas().show()`
* `DataFrame.groupby().cogroup(DataFrame.groupby()).applyInPandas().show()`

## groupby

```python
df = spark.createDataFrame([
    ['red', 'banana', 1, 10], 
    ['blue', 'banana', 2, 20], 
    ['red', 'carrot', 3, 30],
    ['blue', 'grape', 4, 40], 
    ['red', 'carrot', 5, 50], 
    ['black', 'carrot', 6, 60],
    ['red', 'banana', 7, 70], 
    ['red', 'grape', 8, 80],
], schema = ["color", "fruit", "v1", "v2"])
df.show()
```

```python
df \
    .groupby("color") \
    .avg() \
    .show()
```

## applyInPandas

```python
def plus_mean(pandas_df):
    return pandas_df.assign(v1 = pandas_df.v1 - pandas_df.v1.mean())

df \
    .groupby("color") \
    .applyInPandas(plus_mean, schema = df.schema) \
    .show()
```

## cogroup

```python
df1 = spark.createDataFrame([
    (20000101, 1, 1.0), 
    (20000101, 2, 2.0), 
    (20000102, 1, 3.0), 
    (20000102, 2, 4.0),
], schema = ("time", "id", "v1"))

df2 = spark.createDataFrame([
    (20000101, 1, "x"),
    (20000101, 2, "y"),
], schema = ("time", "id", "v2"))


def asof_join(l, r):
    return pd.merge_asof(l, r, on = "time", by = "id")


df1 \
    .groupby("id") \
    .cogroup(df2.groupby("id")) \
    .applyInPandas(
        asof_join, 
        schema = "time int, id int, v1 double, v2 string") \
    .show()
```

# 数据读写

相关 API：

* csv
    - `DataFrame.write.csv()`
    - `spark.read.csv()`
* parquet
    - `DataFrame.write.parquet()`
    - `spark.read.parquet()`
* ORC
    - `DataFrame.write.orc()`
    - `spark.read.orc()` 
* JDBC
* text
* binaryFile
* Avro

## CSV

```python
df.write.csv("foo.csv", header = True)
spark.read.csv("foo.csv", header = True).show()
```

## Parquet

```python
df.write.parquet("bar.parquet")
spark.read.parquet("bar.parquet").show()
```

## ORC

```python
df.write.orc("zoo.orc")
spark.read.orc("zoo.orc").show()
```

# DataFrame 和 Spark SQL

DataFrame 和 Spark SQL 共享相同的执行引擎，因此它们可以无缝互换使用

相关 API：

* `DataFrame.createOrReplaceTempView()`
* `spark.sql()`

## DataFrame 注册为一个表格并运行 SQL

```python
df.createOrReplaceTempView("tableA")
spark.sql("SELECT count(*) from tableA").show()
```

```
+--------+
|count(1)|
+--------+
|       8|
+--------+
```

## UDF 可以开箱即用地在 SQL 中注册和调用

```python
@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

spark.udf.register("add_one", add_one)
spark.sql("SELECT add_one(v1) FROM tableA").show()
```

```
+-----------+
|add_one(v1)|
+-----------+
|          2|
|          3|
|          4|
|          5|
|          6|
|          7|
|          8|
|          9|
+-----------+
```

## SQL 表达式混入作用 PySpark 列

```python
from pyspark.sql.functions import expr

df.selectExpr("add_one(v1)").show()
df.select(expr("count(*)") > 0).show()
```

```
+-----------+
|add_one(v1)|
+-----------+
|          2|
|          3|
|          4|
|          5|
|          6|
|          7|
|          8|
|          9|
+-----------+

+--------------+
|(count(1) > 0)|
+--------------+
|          true|
+--------------+
```

# 参考

* [Quickstart: DataFrame](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)

