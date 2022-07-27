---
title: PySpark SQL API
author: 王哲峰
date: '2022-07-27'
slug: spark-pyspark-sql-api
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

- [Package 和 Subpackages](#package-和-subpackages)
- [pyspark.sql内容](#pysparksql内容)
  - [class pyspark.SparkSession()](#class-pysparksparksession)
    - [类、方法、属性](#类方法属性)
    - [示例](#示例)
  - [class pyspark.sql.types](#class-pysparksqltypes)
</p></details><p></p>

内容:

   - pyspark
   - pyspark.sql
   - pyspark.streaming
   - pyspark.ml
   - pyspark.mllib
   - pyspark.resource

核心类:

   - pyspark.SparkContext
   - pyspark.RDD
   - pyspark.streaming.StreamingContext
   - pyspark.streaming.DStream
   - pyspark.sql.SparkSession
   - pyspark.sql.DataFrame

# Package 和 Subpackages

- pyspark
- pyspark.sql
- pyspark.streaming
- pyspark.ml
- pyspark.mllib

# pyspark.sql内容

- class ``pyspark.sql.SparkSession``
    - DataFrame 和 Spark SQL 功能的主要入口
- class ``pyspark.sql.DataFrame``
    - 以列组织的分布式数据集
- class ``pyspark.sql.Column``
    - DataFrame中的一列
- class ``pyspark.sql.Row``
    - DataFrame中的一行
- class ``pyspark.sql.GroupedData``
    - 聚合函数，DataFrame.groupBy()
- class ``pyspark.sql.DataFrameNaFunctions``
    - 处理缺失数据的函数
- class ``pyspark.sql.DataFrameStatFunctions``
    - 统计函数
- class ``pyspark.sql.function``
    - 处理DataFrame内置函数
- class ``pyspark.sql.types``
    - 可用的数据类型
- class ``pyspark.sql.Window``
    - 窗口函数

## class pyspark.SparkSession()

- 用Dataset和DataFrame API进行Spark编程的接口
- 可以创建DataFrame，将DataFrame注册为表，在表上执行SQL，缓存表以及读取parquet文件

### 类、方法、属性

- .builder
- class Builder
    - .master("")
    - .appName("")
    - .config(key = None, value = None, conf = None)
    - .getOrCreate()
    - .enableHiveSupport()
- .version
- .catalog
- .conf
- .udf
- .sparkContext
- createDataFrame(data, schema = None, samplingRatio = None, verifySchema = True)
    - 从一个RDD、list、pandas.DataFrame创建一个DataFrame
- newSession
- range(start, end = None, step = 1, numPartitions = None)
    - 创建一个类型为pyspark.sql.types.LongType的单列DataFrame
- read
    - DataFrameReader,作为一个DataFrame来读取数据
- readSteam
- sql("SELECT \* FROM t")
    - 返回一个DataFrame
- table(tableName)
- streams
- stop()

### 示例

创建spark编程(DataFrame, Spark SQL)的接口:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Word Count") \
    .config(key = "spark.some.config.option", value = "some-value", conf) \
    .getOrCreate()
```

利用已存在的config创建接口:

```python
from pyspark import SparkConf, SparkContext

conf = SparkConf() \
    .setMaster("local") \
    .setAppName("My First Spark App") \
    .setExecutorEnv(key = None, value = None, pairs = None) \
    .setSparkHome(value = "D:/spark/bin") \
    .setIfMissing(key = None, value = None)

sc = SparkContext(conf = conf)

spark = SparkSession.builder \
    .master("local") \
    .appName("Word Count") \
    .config(conf = conf) \
    .getOrCreate()
```

如果返回现有SparkSession，则此构建器中指定的配置选项将应用于现有SparkSession:

```python
s1 = SparkSession.builder.config('k1', 'v1').getOrCreate()
s2 = SparkSession.builder.config('k2', 'v2').getOrCreate()
s1.conf.get('k1') == s2.conf.get('k1')
s1.conf.get('k2') == s2.conf.get('k2')
```

输出结果：

```
True
True
```

创建DataFrame:

```python
# list
L = [("Alice", 1)]
spark.createDataFrame(L).collect()
spark.createDataFrame(L, schema = ['name', 'age']).collect()
```

```python
# dict
D = [{
    'name': 'Alice', 
    'age': 1
}]
spark.createDataFrame(D).collect()
```

```python
# RDD
rdd = sc.parallelize(L)
spark.createDataFrame(RDD).collect()
```

.range()

```python
spark.range(1, 7, 2).collect()
spark.range(3).collect()
```

.sql()

```python
df.createOrReplaceTempView("table1")
df2 = spark.sql("SELECT field1 as f1, field2 as f2 from table1")
df2.collect()
```

.table()

```python
df.createOrReplaceTempView("table1")
df2 = spark.table("table1")
sorted(df.collect() == df2.collect())
```

## class pyspark.sql.types

- pyspark.sql.types.DataType
    - fromInternal(obj)
        - 将一个SQL对象转换为一个Python对象
    - toInternal(obj)
        - 将一个SQL对象转换为一个Python对象
    - json()
    - jsonVale()
    - needConversion()
    - simpleString()
    - classmethod .typeName()
- pyspark.sql.types.NullType
- pyspark.sql.types.StringType
- pyspark.sql.types.BinaryType
- pyspark.sql.types.BooleanType
- pyspark.sql.types.DataType
