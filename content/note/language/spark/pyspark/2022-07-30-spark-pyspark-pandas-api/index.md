---
title: PySpark Pandas API
subtitle: Pandas API on Spark
author: 王哲峰
date: '2022-07-30'
slug: spark-pyspark-pandas-api
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [配置项](#配置项)
  - [可用配置项](#可用配置项)
  - [getting 和 setting](#getting-和-setting)
    - [属性](#属性)
    - [方法](#方法)
  - [不同 DataFrame 数据操作](#不同-dataframe-数据操作)
  - [索引类型](#索引类型)
    - [sequence](#sequence)
    - [distributed-sequence](#distributed-sequence)
    - [distributed](#distributed)
- [数据对象](#数据对象)
  - [概念区分](#概念区分)
  - [PoS DataFrame 和 Series](#pos-dataframe-和-series)
  - [Pandas df 转换为 PoS df](#pandas-df-转换为-pos-df)
  - [PoS df 转换为 Pandas df](#pos-df-转换为-pandas-df)
  - [Pandas df 转换为 PySpark df](#pandas-df-转换为-pyspark-df)
  - [PySpark df 转换为 Pandas df](#pyspark-df-转换为-pandas-df)
  - [PySpark df 转换为 PoS df](#pyspark-df-转换为-pos-df)
  - [PoS df 转换为 PySpark df](#pos-df-转换为-pyspark-df)
  - [其他属性与方法](#其他属性与方法)
- [缺失数据](#缺失数据)
  - [缺失值删除](#缺失值删除)
  - [缺失值填充](#缺失值填充)
- [数据操作](#数据操作)
  - [统计](#统计)
  - [Spark 配置](#spark-配置)
  - [transform 和 apply](#transform-和-apply)
    - [transform 和 apply](#transform-和-apply-1)
    - [transform\_batch](#transform_batch)
    - [apply\_batch](#apply_batch)
- [Grouping](#grouping)
- [Plotting](#plotting)
  - [Pandas-on-Spark Series](#pandas-on-spark-series)
  - [Pandas-on-Spark DataFrame](#pandas-on-spark-dataframe)
- [数据读写](#数据读写)
  - [CSV](#csv)
  - [Parquet](#parquet)
  - [Spark IO](#spark-io)
- [类型支持和提示](#类型支持和提示)
  - [类型支持](#类型支持)
    - [PySpark df 转换为 PoS df](#pyspark-df-转换为-pos-df-1)
    - [PoS df 转换为 PySpark df](#pos-df-转换为-pyspark-df-1)
    - [PoS df 转换为 Pandas df](#pos-df-转换为-pandas-df-1)
    - [Pandas DataFrame 仅有的类型](#pandas-dataframe-仅有的类型)
    - [Type mapping](#type-mapping)
  - [类型提示](#类型提示)
    - [Pandas-on-Spark DataFrame 和 Pandas DataFrame](#pandas-on-spark-dataframe-和-pandas-dataframe)
    - [Type Hinting with Names](#type-hinting-with-names)
    - [Type Hinting with Index](#type-hinting-with-index)
      - [Index](#index)
      - [MultiIndex](#multiindex)
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
  - [在从 Saprk DataFrame 转换为 Pandas-on-Spark DataFrame 时明确指定索引列名](#在从-saprk-dataframe-转换为-pandas-on-spark-dataframe-时明确指定索引列名)
  - [使用 distributed 或者 distributed-sequence 默认索引](#使用-distributed-或者-distributed-sequence-默认索引)
  - [减少在不同 DataFrame/Seris 上的数据操作](#减少在不同-dataframeseris-上的数据操作)
  - [尽可能直接使用 Pandas API on Spark](#尽可能直接使用-pandas-api-on-spark)
- [支持的 pandas API](#支持的-pandas-api)
- [在 Structured Streaming 上使用 Pandas API on Spark](#在-structured-streaming-上使用-pandas-api-on-spark)
- [参考](#参考)
</p></details><p></p>


# 配置项

> 此处的 API 均在 `pandas_on_spark` 命名空间有效

属性:

* `pyspark.pandas.options.`
    - `display.max_rows`
    - `display.max_columns`

方法:

* `pyspark.pandas.`
    - `get_option("option_name")`
    - `set_option("option_name", new_value)`
    - `reset_option()`
    - `option_context("option_name", option_value, "option_name", option_value, ...)`

## 可用配置项

| 配置型                      | 默认值    | 说明 |
|----------------------------|----------|----|
| `display.max_rows`           | `1000`     | 显示最大行数量 |
| `compute.max_rows`           | `1000`     | 计算适用的最大行数 |
| `compute.shorcut_limit`      | `1000`     | |
| `compute.ops_on_diff_frames` | `False`    | 是否允许在不同的 DataFrame 或 Series 上的数据操作 |
| `compute.default_index_type` | `'distributed-sequence'` | 配置索引类型 |
| `compute.ordered_head`       | `False`    | |
| `compute.eager_check`        | `True`     | |
| `compute.is_in_limit`        | `80`       | |
| `plotting.max_rows`          | `1000`     | |
| `plotting.sample_ration`     | `None`     | |
| `plotting.backend`           | `'plotly'` |  |


## getting 和 setting

### 属性

```python
>>> import pyspark.pandas as ps
>>>
>>> ps.options.display.max_rows
1000
>>> ps.options.display.max_rows = 10
>>> ps.options.display.max_rows
10
```

### 方法

* `get_option()` 和 `set_option()`

```python
>>> import pyspark.pandas as ps
>>> 
>>> ps.get_option("display.max_rows")
1000
>>> ps.set_option("display.max_rows", 10)
>>> ps.get_option("display.max_rows")
10
```

* `reset_option()`

```python
>>> import pyspark.pandas as ps
>>> 
>>> ps.get_option("compute.max_rows")
1000
>>> ps.set_option("compute.max_rows", 2000)
>>> ps.get_option("compute.max_rows")
2000

>>> ps.reset_option("compute.max_rows")
>>> ps.get_option("compute.max_rows")
1000
```

* `option_context()`

```python
>>> import pyspark.pandas as ps
>>> 
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

## 不同 DataFrame 数据操作

为了防止高资源消耗，Pandas API on Spark 默认不允许在不同 DataFrame 或 Series 上的操作。
可以通过设置 `compute.ops_on_diff_frames` 为 `True` 启用操作

DataFrame 和 DataFrame：

```python
>>> import pyspark.pandas as ps
>>> 
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

DataFrame 和 Series：

```python
>>> import pyspark.pandas as ps
>>> 
>>> ps.set_option("compute.ops_on_diff_frames", True)
>>> psdf = ps.range(5)
>>> psser = ps.Series([1, 2, 3, 4])
>>> psdf["new_col"] = psser
>>> psdf
   id  new_col
0   0      1.0
1   1      2.0
3   3      4.0
2   2      3.0
4   4      NaN
>>> ps.reset_option("compute.ops_on_diff_frames")
```

## 索引类型

可以通过 `compute.default_index_type` 配置索引类型，常用的索引类型有：

* sequence
* distributed-sequence：默认配置
* distributed

### sequence

`sequence` 通过 PySpark 的 Window 函数实现了一个递增序列，而无需指定分区。
因此，它可以在单个节点中以整个分区完成。当数据很大时，应该避免使用这种索引类型

```python
>>> import pyspark.pandas as ps
>>> 
>>> ps.set_option("compute.default_index_index", "sequence")
>>> 
>>> psdf = ps.range(3)
>>> 
>>> ps.reset_option("compute.default_index_type")
>>> psdf.index
Int64Index([0, 1, 2], dtype='int64')
```

等价于：

```python
from pyspark.sql import functions as F, Window
import pyspark.pandas as ps

# DataFrame
spark_df = ps.range(3).to_spark()

# Index
sequential_index = F.row_number().over(
    Window.orderBy(F.monotonically_increasing().asc())
) - 1

spark_df \
    .select(sequential_index) \
    .rdd \
    .map(lambda r: r[0])  \
    .collect() \
```

### distributed-sequence

以分布式方式通过分组(group-by)和分组映射(group-map)方法实现一个递增的序列。
它仍然在全局范围内生成顺序索引。如果默认索引必须是大型数据集中的序列，则必须使用该索引

```python
import pyspark.pandas as ps

ps.set_option("compute.default_index_type", "distributed-sequence")
psdf = ps.range(3)
ps.reset_option("compute.default_index_type")

psdf.index
```

```
Int64Index([0, 1, 2], dtype='int64')
```

等价于:

```python
import pyspark.pandas as ps

spark_df = ps.range(3).to_spark()
spark_df.rdd.zipWithIndex().map(lambda p: p[1]).collect()
```

```
[1, 2, 3]
```

### distributed

通过使用 PySpark 的 `monotonically_increasing_id` 函数以完全分布式的方式简单地实现了一个单调递增的序列。
这些值是不确定的。如果索引不必是一个递增的序列，则应使用该索引。在性能方面，与其他索引类型相比，该索引几乎没有任何损失

```python
import pyspark.pandas as ps

ps.set_option("compute.default_index_type", "distribute")
psdf = ps.range(3)
ps.reset_option("compute.default_index_type")

psdf.index
```

```
Int64Index([25769803776, 60129542144, 94489280512], dtype='int64')
```

等价于：

```python
from pyspark.sql import functions as F
import pyspark.pandas as ps

spark_df = ps.range(3).to_spark()
spark_df \
    .select(F.monotonically_increasing_id()) \
    .rdd.map(lambda r: r[0]) \
    .collect() 
```

```
[25769803776, 60129542144, 94489280512]
```

# 数据对象

## 概念区分

* Spark API
    - [Spark DataFrame](https://spark.apache.org/sql/)
* PySpark API
    - [PySpark DataFrame](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)
* Pandas API
    - [Pandas Series](https://pandas.pydata.org/docs/reference/series.html)
    - [Pandas DataFrame](https://pandas.pydata.org/docs/reference/frame.html)
* Pandas API on Spark
    - [Pandas-on-Spark Series](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)
    - [Pandas-on-Spark DataFrame](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)

## PoS DataFrame 和 Series

DataFrame：

```python
>>> import pyspark.pandas as ps
>>> 
>>> psdf = ps.DataFrame({
...     "a": [1, 2, 3, 4, 5, 6],
...     "b": [100, 200, 300, 400, 500, 600],
...     "c": ["one", "two", "three", "four", "five", "six"]
... }, index = [10, 20, 30, 40, 50, 60])
>>> psdf
```

Series：

```python
>>> import pyspark.pandas as ps
>>> 
>>> psser = ps.Series([1, 3, 5, np.nan, 6, 8])
>>> psser
```

## Pandas df 转换为 PoS df

```python
>>> import pandas as pd
>>> import pyspark.pandas as ps
>>> 
>>> dates = pd.date_range("20130101", periods = 6)
>>> pdf = pd.DataFrame(
...     np.random.randn(6, 4), 
...     index = dates,
...     columns = list("ABCD"),
... )
>>> 
>>> psdf = ps.from_pandas(pdf)
>>> type(psdf)
>>> psdf
```

## PoS df 转换为 Pandas df

```python
>>> import pyspark.pandas as ps
>>>
>>> psdf = ps.DataFrame({
...     'id': range(10)
... }, index = range(10))
>>> pdf = psdf.to_pandas()
>>> pdf.values
array([[0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9]])
```

## Pandas df 转换为 PySpark df

```python
>>> import pandas as pd
>>> from pyspark.sql import SparkSession
>>> 
>>> dates = pd.date_range("20130101", periods = 6)
>>> pdf = pd.DataFrame(
...     np.random.randn(6, 4), 
...     index = dates,
...     columns = list("ABCD"),
... )
>>> 
>>> spark = SparkSession.builder.getOrCreate()
>>> sdf = spark.createDataFrame(pdf)
>>> sdf.show()
```

## PySpark df 转换为 Pandas df

```python

```

## PySpark df 转换为 PoS df

```python
>>> from pandas as pd
>>> from pyspark.sql import SparkSession
>>> 
>>> spark = SparkSession.builder.getOrCreate()
>>>
>>> pdf = pd.DataFrame({  # Pandas DataFrame
...     'id': range(10)
... }, index = range(10))
>>> 
>>> sdf = spark.createDataFrame(pdf)  # Spark DataFrame
>>>
>>> psdf = sdf.pandas_api(index_col = "index")  # Pandas-on-Spark DataFrame
>>> psdf
```

## PoS df 转换为 PySpark df

```python
>>> import pyspark.pandas as ps
>>>
>>> psdf = ps.DataFrame({
...     'id': range(10)
... }, index = range(10))
>>> sdf = psdf.to_spark(index_col = "index").filter("id > 5")
>>> sdf.show()
+---+
| id|
+---+
|  6|
|  7|
|  8|
|  9|
+---+
```

## 其他属性与方法

dtypes：

```python
psdf.dtypes
```

head：

```python
psdf.head()
```

index：
```python
psdf.index
```

columns：

```python
psdf.columns
```

to_numpy：

```python
psdf.to_numpy()
```

describe：

```python
psdf.describe()
```

T：

```python
psdf.T
```

sort_index：

```python
psdf.sort_index(ascending = False)
```

sort_values：

```python
psdf.sort_values(by = "B")
```

# 缺失数据

```python
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.pandas as ps


# pandas df
dates = pd.date_range("20130101", periods = 6)
pdf = pd.DataFrame(
    np.random.randn(6, 4), 
    index = dates,
    columns = list("ABCD"),
)
pdf1 = pdf.reindex(
    index = dates[0:4], 
    columns = list(pdf.columns) + ["E"],
)
pdf1.loc[dates[0]:dates[1], "E"] = 1

# pandas on spark df
psdf = ps.from_pandas(pdf1)
psdf
```

## 缺失值删除

```python
psdf.dropna(how = "any")
```

## 缺失值填充

```python
psdf.fillna(value = 5)
```

# 数据操作

## 统计

```python
psdf.mean()
```

## Spark 配置

> Arrow optimization

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
ps.reset_option("compute.default_index_type")
spark.conf.set("spark.sql.execution.arrow.pyspark.enable", prev)
```

## transform 和 apply

APIs：

* `DataFrame.transform()`
* `DataFrame.apply()`
* `DataFrame.pandas_on_spark.transform_batch()`
* `DataFrame.pandas_on_spark.apply_batch()`
* `Series.pandas_on_spark.transform_batch()`

### transform 和 apply

* transform: 要求输入有相同的长度

```python
psdf = ps.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

def pandas_plus(pser):
    return pser + 1

psdf.transform(pandas_plus)
```

* apply: 不要求输入有相同的长度

```python
psdf = ps.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

def pandas_plus(pser):
    return pser[pser % 2 == 1]

psdf.apply(pandas_plus)
```

每个函数传入一个 Pandas Series，并且 Pandas API on Spark 以分布式的方式运行该函数:

![](https://spark.apache.org/docs/latest/api/python/_images/pyspark-pandas_on_spark-transform_apply1.png)

```python
psdf = ps.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

def pandas_plus(pser):
    return sum(pser)

psdf.apply(pandas_plus, axis = "columns")
```
![](https://spark.apache.org/docs/latest/api/python/_images/pyspark-pandas_on_spark-transform_apply2.png)

### transform_batch

> pandas_on_spark.transform_batch

batch 后缀代表在 Pandas-on-Spark DataFrame 或者 Series 上的每个数据块。
这些 API 索引/切片 Pandas-on-Spark DataFrame 或 Series，
然后将 Pandas DataFrame 或 Series 作为给定的函数的输入、输出

* DataFrame.pandas_on_spark.transform_batch() 要求输入、输出长度必须相同
* DataFrame.pandas_on_spark.transform_batch() 的输出与输入是同一个 DataFrame

```python
# DataFrame.pandas_on_spark.transform_batch
psdf = ps.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

def pandas_plus(pdf):
    return pdf + 1  # should always return the same length as input

psdf.pandas_on_spark.transform_batch(pandas_plus)  # Pandas DataFrame
```

![](https://spark.apache.org/docs/latest/api/python/_images/pyspark-pandas_on_spark-transform_apply3.png)

* Series.pandas_on_spark.transform_batch() 需要 Pandas Series 作为 Pandas-on-Spark Series 的数据块(chunk)

```python
# Series.pandas_on_spark.transform_batch
psdf = ps.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

def pandas_plus(pser):
    return pser + 1  # should always return the same length as input

psdf.a.pandas_on_spark.transform_batch(pandas_plus)
```

在后台，每个 Pandas-on-Spark Series 被分割为多个 Pandas Series，每个函数被应用在每个 Pandas Series

![](https://spark.apache.org/docs/latest/api/python/_images/pyspark-pandas_on_spark-transform_apply4.png)

### apply_batch

> pandas_on_spark.apply_batch

batch 后缀代表在 Pandas-on-Spark DataFrame 或者 Series 上的每个数据块。
这些 API 索引/切片 Pandas-on-Spark DataFrame 或 Series，
然后将 Pandas DataFrame 或 Series 作为给定的函数的输入、输出

* DataFrame.pandas_on_spark.apply_batch() 不要求输入、输出长度必须相同
* DataFrame.pandas_on_spark.apply_batch() 返回一个 Series 时，它的输出是一个新的 DataFrame。
  并且可以通过在不同的 DataFrame 上的操作避免 shuffle

```python
psdf = ps.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

def pandas_plus(pdf):
    return pdf[pdf.a > 1]

psdf.pands_on_spark.apply_batch(pandas_plus)  # Pandas DataFrame
```

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

## Pandas-on-Spark Series

```python
pser = pd.Series(
    np.random.randn(1000),
    index = pd.date_range("1/1/2000", periods = 10000)
)
psser = ps.Series(pser)
psser = psser.cummax()
psser.plot()
```

## Pandas-on-Spark DataFrame

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

# 数据读写

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

# 类型支持和提示

## 类型支持

### PySpark df 转换为 PoS df

```python
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.getOrCreeate()
>>> 
>>> sdf = spark.createDataFrame([
...     (1, Decimal(1.0), 1., 1., 1, 1, 1, datetime(2020, 10, 27), "1", True, datetime(2020, 10, 27)),
... ], 'tinyint tinyint, decimal decimal, float float, double double, integer integer, long long, short short, timestamp timestamp, string string, boolean boolean, date date')

>>> sdf
DataFrame[tinyint: tinyint, decimal: decimal(10,0), float: float, double: double, integer: int, long: bigint, short: smallint, timestamp: timestamp, string: string, boolean: boolean, date: date]

>>> psdf = sdf.pandas_api()
>>> psdf.dtypes
tinyint                int8
decimal              object
float               float32
double              float64
integer               int32
long                  int64
short                 int16
timestamp    datetime64[ns]
string               object
boolean                bool
date                 object
dtype: object
```

### PoS df 转换为 PySpark df

```python
>>> from pyspark.pandas as ps
>>>
>>> psdf = ps.DataFrame({
...     "int8": [1], 
...     "bool": [True], 
...     "float32": [1.0], 
...     "float64": [1.0], 
...     "int32": [1], 
...     "int64": [1], 
...     "int16": [1], 
...     "datetime": [datetime.datetime(2020, 10, 27)], 
...     "object_string": ["1"], 
...     "object_decimal": [decimal.Decimal("1.1")], 
...     "object_date": [datetime.date(2020, 10, 27)],
... })
>>>
>>> psdf["int8"] = psdf["int8"].astype("int8")
>>> psdf["int16"] = psdf["int16"].astype("int16")
>>> psdf["int32"] = psdf["int32"].astype("int32")
>>> psdf["float32"] = psdf["float32"].astype("float32")
>>>
>>> psdf.dtypes
int8                        int8
bool                        bool
float32                  float32
float64                  float64
int32                      int32
int64                      int64
int16                      int16
datetime          datetime64[ns]
object_string             object
object_decimal            object
object_date               object
dtype: object
>>>
>>> sdf = psdf.to_spark()
>>> sdf
DataFrame[int8: tinyint, bool: boolean, float32: float, float64: double, int32: int, int64: bigint, int16: smallint, datetime: timestamp, object_string: string, object_decimal: decimal(2,1), object_date: date]
```

### PoS df 转换为 Pandas df

```python
>>> from pyspark.pandas as ps
>>>
>>> psdf = ps.DataFrame({
...     "int8": [1], 
...     "bool": [True], 
...     "float32": [1.0], 
...     "float64": [1.0], 
...     "int32": [1], 
...     "int64": [1], 
...     "int16": [1], 
...     "datetime": [datetime.datetime(2020, 10, 27)], 
...     "object_string": ["1"], 
...     "object_decimal": [decimal.Decimal("1.1")], 
...     "object_date": [datetime.date(2020, 10, 27)],
... })
>>>
>>> psdf["int8"] = psdf["int8"].astype("int8")
>>> psdf["int16"] = psdf["int16"].astype("int16")
>>> psdf["int32"] = psdf["int32"].astype("int32")
>>> psdf["float32"] = psdf["float32"].astype("float32")
>>> 
>>> pdf = psdf.to_pandas()
>>> pdf.dtypes
int8                        int8
bool                        bool
float32                  float32
float64                  float64
int32                      int32
int64                      int64
int16                      int16
datetime          datetime64[ns]
object_string             object
object_decimal            object
object_date               object
dtype: object
```

### Pandas DataFrame 仅有的类型

目前 Pandas API on Spark 不支持的 Pandas 类型:

* `pd.Timedelta`
* `pd.Categorical`
* `pd.CategoricalDtype`

以后 Pandas API on Spark 不支持的 Pandas 类型:

* `pd.SparseDtype`
* `pd.DatetimeTZDtype`
* `pd.UInt*Dtype`
* `pd.BooleanDtype`
* `pd.StringDtype`

示例:

```python
>>> from pyspark.pandas as ps
>>> 
>>> ps.Series([pd.Categorical[1, 2, 3]])
Traceback (most recent call last):
...
pyarrow.lib.ArrowInvalid: Could not convert [1, 2, 3]
Categories (3, int64): [1, 2, 3] with type Categorical: did not recognize Python value type when inferring an Arrow data type
```

### Type mapping

在 Pandas-on-Spark API 中，Numpy 数据类型与 PySpark 的内部数据类型的映射:

| NumPy | PySpark |
|-------|---------|
| np.character | BinaryType |
| np.bytes_ | BinaryType |
| np.string_ | BinaryType |
| np.int8 | ByteType |
| np.byte | ByteType |
| np.int16 | ShortType |
| np.int32 | IntegerType |
| np.int64 | LongType |
| np.int | LongType |
| np.float32 | FloatType |
| np.float | DoubleType |
| np.float64 | DoubleType |
| np.str | StringType |
| np.unicode_ | StringType |
| np.bool | BooleanType |
| np.datetime64 | TimestampType |
| np.ndarray | ArrayType(StringType()) |

在 Pandas-on-Spark API 中，Python 数据类型与 PySpark 的内部数据类型的映射:

| Numpy          | Python | PySpark |
|----------------|--------|---------|
| np.character/np.bytes_/np.string_      | bytes  | BinaryType |
| np.int/np.int64| int    | LongType |
| np.float/np.float64 | float   | DoubleType |
| np.str/np.unicode_  | str    | StringType |
| np.bool | bool   | BooleanType |
| np.datetime64 | datetime.datetime | TimestampType|
|  | datetime.date | DateType |
|  | decimal.Decimal | DecimalType(38, 18) |


使用 `pyspark.pandas.typedef.as_spark_type()` 检查上面的数据类型映射

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

使用 Spark 访问器检查 Series 底层的 PySpark 数据类型，或 DataFrame 的 schema

```python
>>> from pyspark.pandas as ps
>>>
>>> ps.Series([0.3, 0.1, 0.8]).spark.date_type
DoubleType

>>> ps.Series(["welcome", "to", "pandas-on-Spark"]).spark.data_type
ArrayType(BooleanType, true)

>>> ps.DataFrame({
...     "d": [0.3, 0.1, 0.8],
...     "s": ["welcome", "to", "pandas-on-Spark"],
...     "b": [False, True, False],
... }).spark.print_schema()
root
 |-- d: double (nullable = false)
 |-- s: string (nullable = false)
 |-- b: boolean (nullable = false)
```


**None:**

Pandas-on-Spark API 目前不支持在一列中出现多种数据类型

```python
>>> ps.Series([1, "A"])
Traceback (most recent call last):
...
TypeError: an integer is required (got type str)
```

## 类型提示

默认情况下，Pandas API on Spark 通过从输出的顶部部分数据记录来推断 schema，
特别当使用的 API 允许 apply 一个函数在 Pandas-on-Spark DataFrame 上时。
然而，根据部分数据推断 schema 的操作是非常昂贵的。比如:

* DataFrame.transform()
* DataFrame.apply()
* DatgFrame.pandas_on_spark.transform_batch()
* DataFrame.pandas_on_spark.apply_batch()
* Series.pandas_on_spark.apply_batch()

为了避免昂贵的操作，Pandas API on Spark 有自己的类型提示方式来指定 schema 避免自动推断 schema。
Pandas API on Spark 理解返回类型中指定的类型提示，
并将其以 Spark schema 的方式转换为内部使用的 Pandas UDF 

### Pandas-on-Spark DataFrame 和 Pandas DataFrame

目前，Pandas API on Spark 和 Pandas 实例都衣蛾用于指定类型提示，
但是，Pandas-on-Spark 计划仅在稳定性得到证明时才逐渐转向使用 Pandas 实例

* 在早期 Pandas-on-Spark 版本中，引入了在函数中指定类型提示，
  以便将其用作 Spark schema

```python
import pandas as pd
import pyspark.pandas as ps

def pandas_div(pdf) -> ps.DataFrame[float, float]:
    """
    pdf is a pandas DataFrame
    """
    return pdf[["B", "C"]] / pdf[["B", "C"]]

df = ps.DataFrame({
    "A": ["a", "a", "b"],
    "B": [1, 2, 3],
    "C": [4, 5, 6],
})
df.groupby("A").apply(pandas_div)
```

* 在 Python 3.7+ 中，pandas.DataFrame 和 pandas.Series 可以作为指定类型提示:

```python
import pandas as pd
import pyspark.pandas as ps

def pandas_div(pdf) -> pd.DataFrame[float, float]:
    """
    pdf is a pandas DataFrame
    """
    return pdf[["B", "C"]] / pdf[["B", "C"]]

df = ps.DataFrame({
    "A": ["a", "a", "b"],
    "B": [1, 2, 3],
    "C": [4, 5, 6],
})
df.groupby("A").apply(pandas_div)
```

```python
import pandas as pd
import pyspark.pandas as ps

def sqrt(x) -> pd.Series[float]:
    return np.sqrt(x)

df = ps.DataFrame(
    [[4, 9]] * 3,
    columns = ["A", "B"],
)
df.apply(sqrt, axis = 0)
```

### Type Hinting with Names

```python
import pandas as pd
import pyspark.pandas as ps

def transform(pdf) -> pd.DataFrame["id": int, "A": int]:
    pdf["A"] = pdf.id + 1
    return pdf

ps.range(5).pandas_on_spark.apply_batch(transform)
```

```python
import panndas as pd
import pyspark.pandas as ps

def transform(pdf) -> pd.DataFrame[zip(sample.columns, sample.dtypes)]:
    return pdf + 1

psdf.pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[sample.dtypes]:
    return pdf + 1

psdf.pandas_on_spark.apply_batch(transform)
```

### Type Hinting with Index

#### Index

* DataFrame

```python
pdf = pd.DataFrame({
    "id": range(5)
})
sample = pdf.copy()
sample["a"] = sample.id + 1
```

```python
def transform(pdf) -> pd.DataFrame[int, [int, int]]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[sample.index.dtype, sample.dtypes]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[("idxA", int), [("id", int), ("a", int)]]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[
        (sample.index.name, sample.index.dtype), 
        zip(sample.columns, sample.dtypes)]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

#### MultiIndex

```python
midx = pd.MultiIndex.from_arrays(
    [(1, 1, 2), (1.5, 4.5, 7.5)],
    names = ("int", "float"),
)
pdf = pd.DataFrame(
    range(3), 
    index = midx, 
    columns = ["id"]
)
sample = pdf.copy()
sample["a"] = sample.id + 1
```

```python
def transform(pdf) -> pd.DataFrame[[int, float], [int, int]]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[sample.index.dtypes, sample.dtypes]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[[("int", int), ("float", float)], [("id", int), ("a", int)]]:
    pdf["a"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```

```python
def transform(pdf) -> pd.DataFrame[zip(sample.index.names, sample.index.dtpes), zip(sample.columns, sample.dtypes)]:
    pdf["A"] = pdf.id + 1
    return pdf

ps.from_pandas(pdf).pandas_on_spark.apply_batch(transform)
```


# From/to others DBMSes

Pandas API on Spark 中与其他 DBMS 交互的 API 与 pandas 中的 API 略有不同，
因为 Pandas API on Spark 利用 PySpark 中的 JDBC API 来读取和写入其他 DBMS

读写外部 DBMS 的 API 如下:

| API | 说明 |
|----|----|
| read_sql_table(table_name, con[, schema, ...]) | 将 SQL 数据库表读入 DataFrame |
| read_sql_query(sql, con[, index_col]) | 将 SQL 查询读入 DataFrame |
| read_sql(sql, con[, index_col, columns]) | 将 SQL 查询或数据库表读入 DataFrame |

Pandas-on-Spark 需要一个规范的 JBDC URL `con`，
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

Pandas API on Spark 在后台使用 Spark; 
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

因为 Pandas API on Spark 是惰性计算的，
所以可以通过在实际计算之前使用 `pyspark.pandas.DataFrame.spark.explain()` 预测高消耗操作

```python
>>> import pyspark.pandas as ps

>>> psdf = ps.DataFrame({
...     "id": range(10)
... })
>>> psdf = psdf[psdf.id > 5]
>>> psdf.spark.explain()
== Physical Plan ==
*(1) Filter (id#1L > 5)
+- *(1) Scan ExistingRDD[__index_level_0__#0L,id#1L]
```

## 使用 checkpoint

在 Spark 对象上对 pandas API 进行大量操作后，由于庞大而复杂的计划，
底层 Spark 计划程序可能会变慢。如果 Spark 计划变得庞大或计划需要很长时间,
`pyspark.pandas.DataFrame.spark.checkpoint()` 或者 
`pyspark.pandas.DataFrame.spark.local_checkpoint()` 会有所帮助

```python
>>> import pyspark.pandas as ps
>>> psdf = ps.DataFrame({
...     "id": range(10)
... })
>>> psdf["id"] = psdf["id"] + (10 * psdf["id"] + psdf["id"])
>>> psdf = psdf.groupby("id").head(2)
>>> psdf.spark.explain()
== Physical Plan ==
*(3) Project [__index_level_0__#0L, id#31L]
+- *(3) Filter (isnotnull(__row_number__#44) AND (__row_number__#44 <= 2))
   +- Window [row_number() windowspecdefinition(__groupkey_0__#36L, __natural_order__#16L ASC NULLS FIRST, specifiedwindowframe(RowFrame, unboundedpreceding$(), currentrow$())) AS __row_number__#44], [__groupkey_0__#36L], [__natural_order__#16L ASC NULLS FIRST]
      +- *(2) Sort [__groupkey_0__#36L ASC NULLS FIRST, __natural_order__#16L ASC NULLS FIRST], false, 0
         +- Exchange hashpartitioning(__groupkey_0__#36L, 200), true, [id=#33]
            +- *(1) Project [__index_level_0__#0L, (id#1L + ((id#1L * 10) + id#1L)) AS __groupkey_0__#36L, (id#1L + ((id#1L * 10) + id#1L)) AS id#31L, __natural_order__#16L]
               +- *(1) Project [__index_level_0__#0L, id#1L, monotonically_increasing_id() AS __natural_order__#16L]
                  +- *(1) Filter (id#1L > 5)
                     +- *(1) Scan ExistingRDD[__index_level_0__#0L,id#1L]


>>> psdf = psdf.spark.local_checkpoint()  # or psdf.spark.checkpoint()
>>> psdf.spark.explain()
== Physical Plan ==
*(1) Project [__index_level_0__#0L, id#31L]
+- *(1) Scan ExistingRDD[__index_level_0__#0L,id#31L,__natural_order__#59L]
```

在 `psdf.spark.explain()` 设置之前，之前的 Spark 计划被删除，
并从一个简单的计划开始。调用时将上一个 DataFrame 的结果存储在配置的文件系统 
`pyspark.pandas.DataFrame.spark.checkpoint()` 中，
或者 `pyspark.pandas.DataFrame.spark.local_checkpoint()`

## 避免 shuffling

一些操作，例如 `sort_values` 在并行或分布式环境中比在单台机器上的内存中更难完成，
因为它需要将数据发送到其他节点，并通过网络在多个节点之间交换数据

```python
>>> import pyspark.pandas as ps
>>> psdf = ps.DataFrame({
...     "id": range(10)
... }).sort_values(by = "id")
>>> psdf.spark.explain()
== Physical Plan ==
*(2) Sort [id#9L ASC NULLS LAST], true, 0
+- Exchange rangepartitioning(id#9L ASC NULLS LAST, 200), true, [id=#18]
   +- *(1) Scan ExistingRDD[__index_level_0__#8L,id#9L]
```

## 避免在单个分区上计算

目前, `DataFrame.rank` 等一些 API 使用 PySpark 的 Window 而不指定分区规范。
这会将所有数据移动到单个机器中的单个分区中，并可能导致严重的性能下降。
对于非常大的数据集，应避免使用此类 API. 相反，可以使用 `GroupBy.rank`，因为它成本较低，
可以为每个组分配和计算数据

```python
>>> import pyspark.pandas as ps
>>> psdf = ps.DataFrame({
...     "id": range(10)
... })
>>> psdf.rank().spark.explain()
== Physical Plan ==
*(4) Project [__index_level_0__#16L, id#24]
+- Window [avg(cast(_w0#26 as bigint)) windowspecdefinition(id#17L, specifiedwindowframe(RowFrame, unboundedpreceding$(), unboundedfollowing$())) AS id#24], [id#17L]
   +- *(3) Project [__index_level_0__#16L, _w0#26, id#17L]
      +- Window [row_number() windowspecdefinition(id#17L ASC NULLS FIRST, specifiedwindowframe(RowFrame, unboundedpreceding$(), currentrow$())) AS _w0#26], [id#17L ASC NULLS FIRST]
         +- *(2) Sort [id#17L ASC NULLS FIRST], false, 0
            +- Exchange SinglePartition, true, [id=#48]
               +- *(1) Scan ExistingRDD[__index_level_0__#16L,id#17L]
```

## 避免使用保留列名

带有前缀 `__` 和后缀 `__` 的列名是在 Pandas API on Spark 中保留的。
为了处理诸如索引之类的内部行为，Pandas API on Spark 使用了一些内部列名。
因此，不鼓励使用此类列名，并且不能保证它们有效

## 不要使用重复列名

Spark SQL 不允许使用重复列名，同样，Pandas API on Spark 继承了这种行为

```python
>>> import pyspark.pandas as ps
>>> psdf = ps.DataFrame({
...     "a": [1, 2],
...     "b": [3, 4],
... })
>>> psdf.columns = ["a", "b"]
...
Reference 'a' is ambiguous, could be: a, a.;
```

另外，强烈建议不要使用区分大小写的列名。Pandas API on Spark 默认不允许它。
但是，可以用配置项 `spark.sql.casSenstive` 在 Spark 配置中打开，但风险很大

```python
>>> import pyspark.pandas as ps
>>> psdf = ps.DataFrame({
...     "a": [1, 2],
...     "A": [3, 4],
... })
...
Reference 'a' is ambiguous, could be: a, a.;
```

```python
>>> from pyspark.sql. import SparkSession
>>> builder = SparkSession.builder
...     .appName("pandas-on-spark")
...     .config("spark.sql.caseSensitive", "true")
...     .getOrCreate()

>>> import pyspark.pandas as ps
>>> psdf = ps.DataFrame({
...     "a": [1, 2],
...     "A": [3, 4],
... })
>>> psdf
   a  A
0  1  3
1  2  4
```

## 在从 Saprk DataFrame 转换为 Pandas-on-Spark DataFrame 时明确指定索引列名

当 Pandas-on-Spark Dataframe 从 Spark DataFrame 转换时，它会丢失索引信息，
这导致在 Spark DataFrame 上使用 pandas API 中的默认索引。
与显式指定索引列相比，默认索引通常效率低下。尽可能指定索引列

## 使用 distributed 或者 distributed-sequence 默认索引

Pandas-on-Spark 用户面临的一个常见问题是默认索引导致性能下降。
当索引未知时，Pandas API on Spark 会附加一个默认索引，
例如 Spark DataFrame 直接转换为 Pandas-on-Spark DataFrame

`sequence` 需要在单个分区上进行计算，不鼓励这么做，如果计划在生产环境中处理大量数据，
建议使用 `distributed`、`distributed-sequence` 使用分布式计算

## 减少在不同 DataFrame/Seris 上的数据操作

Pandas API on Spark 默认不允许对不同 DataFrame（或 Series）进行操作，
以防止昂贵的操作。它在内部执行一个连接操作，这通常会很昂贵，这是不鼓励的。
只要有可能，就应该避免这种操作

## 尽可能直接使用 Pandas API on Spark

尽管 Pandas API on Spark 具有大部分与 pandas 等效的 API，但仍有一些 API 尚未实现或明确不受支持

Spark 上的 pandas API 没有实现 `__iter__()` 阻止用户将所有数据从整个集群收集到客户端(驱动程序)端。
不幸的是，许多外部 API，例如 Python 内置函数，例如 min、max、sum 等，都要求给定参数是可迭代的。
对于 pandas，它开箱即用

```python
>>> import pandas as pd
>>> max(pd.Series([1, 2, 3]))
3
>>> min(pd.Series([1, 2, 3]))
1
>>> sum(pd.Series([1, 2, 3]))
6
```

pandas 数据集存在于单台机器中，自然可以在同一台机器内进行本地迭代。
但是，pandas-on-Spark 数据集存在于多台机器上，并且它们是以分布式方式计算的。
很难在本地迭代，很可能用户在不知情的情况下将整个数据收集到客户端。
因此，最好坚持使用 Pandas API on Spark

```python
>>> import pyspark.pandas as ps
>>> ps.Series([1, 2, 3]).max()
3
>>> ps.Series([1, 2, 3]).min()
1
>>> ps.Series([1, 2, 3]).sum()
6
```

pandas 用户的另一个常见模式可能是依赖列表解析式或生成器表达式。
但是，它还假设数据集在引擎盖下是本地可迭代的。因此，它可以在 pandas 中无缝运行

* pandas API

```python
>>> import pandas as pd
>>> data = []
>>> countries = ["London", "New York", "Helsinki"]
>>> pser = pd.Series([20., 21., 12.], index = countries)
for temperatures in pser:
    assert temperature > 0
    if temperature > 1000:
        temperature = None
    data.append(temperature ** 2)

>>> pd.Series(data, index = countries)
London      400.0
New York    441.0
Helsinki    144.0
dtype: float64
```

* Pandas API on Spark

```python
>>> import pyspark.pandas as ps
>>> import numpy as np
>>> countries = ['London', 'New York', 'Helsinki']
>>> psser = ps.Series([20., 21., 12.], index=countries)
>>> def square(temperature) -> np.float64:
...     assert temperature > 0
...     if temperature > 1000:
...         temperature = None
...     return temperature ** 2
...
>>> psser.apply(square)
London      400.0
New York    441.0
Helsinki    144.0
dtype: float64
```

# 支持的 pandas API

所有在 Pandas API on Spark 实现支持的 Pandas API 都通过分布式执行计算数据，
除了那些需要设计本地执行的数据。例如，`DataFrame.to_numpy()` 需要将数据收集到驱动端

* DataFrame API
* I/O API
* General Function API
* Series API
* Index API
* Window API
* GroupBy API

# 在 Structured Streaming 上使用 Pandas API on Spark 

Pandas API on Spark 目前不支持 Structured Streaming，
作为一种解决方法，你可使用带有 `foreachBatch` 的 Pandas API on Spark

```python
>>> from pyspark.pandas as ps
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.getOrCreate()
>>>
>>> def func(batch_df, batch_id):
...     pandas_on_spark_df = ps.DataFrame(batch_df)
...     pandas_on_spark_df["a"] = 1
...     print(pandas_on_spark_df)
>>> 
>>> spark
...     .readStream.format("rate").load()
...     .writeStream.foreachBatch(func)
...     .start()
                timestamp  value  a
0 2020-02-21 09:49:37.574      4  1
                timestamp  value  a
0 2020-02-21 09:49:38.574      5  1
...
```

# 参考

* [Quickstart: Pandas API on Spark](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_ps.html)
* [Pandas API on Spark](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)

