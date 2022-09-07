---
title: Spark RDD 编程
author: 王哲峰
date: '2022-09-07'
slug: spark-rdd
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

- [Spark Low-Level API](#spark-low-level-api)
  - [What are the Low-Level APIs ?](#what-are-the-low-level-apis-)
  - [When to Use the Low-Level APIs ?](#when-to-use-the-low-level-apis-)
  - [How to Use the Low-Level APIs ?](#how-to-use-the-low-level-apis-)
- [RDD](#rdd)
  - [Spark 配置](#spark-配置)
  - [PySpark 配置](#pyspark-配置)
  - [创建 RDD](#创建-rdd)
    - [DataFrame, Dataset, RDD 交互操作](#dataframe-dataset-rdd-交互操作)
    - [从 Local Collection 创建 RDD](#从-local-collection-创建-rdd)
      - [Spark](#spark)
      - [PySpark](#pyspark)
    - [从数据源创建 RDD](#从数据源创建-rdd)
      - [Spark](#spark-1)
      - [PySpark](#pyspark-1)
  - [操作 RDD](#操作-rdd)
    - [常用 Transformation 操作](#常用-transformation-操作)
    - [常用 Action 操作](#常用-action-操作)
      - [collect](#collect)
      - [take](#take)
      - [reduce](#reduce)
      - [count](#count)
      - [countApprox](#countapprox)
      - [countApproxDistinct](#countapproxdistinct)
      - [countByValue](#countbyvalue)
      - [countByValueApprox](#countbyvalueapprox)
      - [first](#first)
      - [max/min](#maxmin)
    - [Saving Files](#saving-files)
    - [Caching](#caching)
    - [Checkpointing](#checkpointing)
    - [Pipe RDDs to System Commands](#pipe-rdds-to-system-commands)
- [Key-Value RDD](#key-value-rdd)
- [Distributed Shared Variables(分布式共享变量)](#distributed-shared-variables分布式共享变量)
</p></details><p></p>

# Spark Low-Level API

## What are the Low-Level APIs ?

- Resilient Distributed Dataset (RDD)
- Distributed Shared Variables (共享变量)
     - Accumulators
     - Broadcast Variable

## When to Use the Low-Level APIs ?

- 在高阶 API 中针对具体问题没有可用的函数时
- Maintain some legacy codebase written using RDDs
- 需要进行自定义的共享变量操作时

## How to Use the Low-Level APIs ?

- `SparkContext` 是 Low-Level APIs 的主要入口:
    - `SparkSession.SparkContext`
    - `spark.SparkContext`


# RDD

- RDD 创建
- RDD 操作 API
    - Action
    - Transformation
    - Pair RDD Transformation
- RDD 缓存
- 共享变量
- RDD 持久化
- RDD 分区


## Spark 配置

```python

```

## PySpark 配置

```python
import findspark

import findspark
import pyspark
from pyspark import SparkContext, SparkConf


# 指定 spark_home
spark_home = "/usr/local/spark"
# 指定 Python 路径
python_path = "/Users/zfwang/.pyenv/versions/3.7.10/envs/pyspark/bin/python"
findspark.init(spark_home, python_path)

conf = SparkConf().setAppName("WordCount").setMaster("local[4]")
sc = SparkContext(conf = conf)
print(f"pyspark.__version__ = {pyspark.__version__}")
```

## 创建 RDD

### DataFrame, Dataset, RDD 交互操作

从 DataFrame 或 Dataset 创建 RDD:

```scala
// in Scala: converts a Dataset[Long] to RDD[Long]
spark.range(500).rdd

// convert Row object to correct data type or extract values
spark.range(500).toDF().rdd.map(rowObject => rowObject.getLong(0))
```

```python
# in Python: converts a DataFrame to RDD of type Row
spark.range(500).rdd

spark.range(500).toDF().rdd.map(lambda row: row[0])
```

从 RDD 创建 DataFrame 和 Dataset:

```scala
// in Scala
spark.range(500).rdd.toDF()
```

```python
# in Python
spark.range(500).rdd.toDF()
```

### 从 Local Collection 创建 RDD

#### Spark

- `SparkSession.SparkContext.parallelize()`

```scala
// in Scala
val myCollection = "Spark The Definitive Guide: Big Data Processing Made Simple".split(" ")
val words = spark.sparkContext.parallelize(myCollection, 2)
words.setName("myWords")
println(words.name)
```

```python
# in Python
myCollection = "Spark The Definitive Guide: Big Data Processing Made Simple" \
    .split(" ")
words = spark.sparkContext.parallelize(myCollection, 2)
words.setName("myWords")
print(word.name())
words.collect()
```

#### PySpark

```python
rdd = sc.parallelize(range(1, 11), 2)
rdd.collect()
```

```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


### 从数据源创建 RDD

使用 `textFile` 加载本地或者集群文件系统中的数据

#### Spark

```scala
// in Scala
// each record in the RDD is the a line in the text file
spark.sparkContext.textFile("/some/path/withTextFiles")

// each text file is a single record in RDD
spark.sparkContext.wholeTextFiles("/some/path/withTextFiles")
```

```python
# in Python
# each record in the RDD is the a line in the text file
spark.sparkContext.textFile("/some/path/withTextFiles")

# each text file is a single record in RDD
spark.sparkContext.wholeTextFiles("/some/path/withTextFiles")
```

#### PySpark

从本地文件系统中加载数据:

```python
file = "./data/hello.txt"
rdd = sc.textFile(file, 3)
rdd.collect()
```

```
["hello world",
 "helo spark",
 "spark love jupyter",
 "spark love pandas",
 "spark love sql"]
```

从集群文件系统中加载数据:

```python
file = "hdfs://localhost:9000/user/hadoop/data.txt"
rdd = sc.textFile(file, 3)
rdd.collect()
```

## 操作 RDD

- 操作 raw Java or Scala object instead of Spark types;

### 常用 Transformation 操作

* distinct 

```scala
// in Scala
words
.distinct()
.count()
```

* filter

```scala
// in Scala
def startsWithS(individual: String) = {
    individual.startsWith("S")
}

words
    .filter(word => startsWithS(word))
    .collect()
```

```python

# in Python
def startsWithS(individual):
    return individual.startsWith("S")

words \
    .filter(lambda word: startsWithS(word)) \
    .collect()
```

* map

```scala
val words2 = words.map(word => (word, word(0), word.startsWith("S")))
words2
    .filter(record => record._3)
    .take(5)
```

```python
# in Python
words2 = words.map(lambda word: (word, word[0], word.startsWith("S")))
words2 \
    .filter(lambda record: record[2]) \
    .take(5)
```

* flatMap

```scala
// in Scala
words
    .flatMap(word => word.toSeq)
    .take()
```

```python
# in Python
words \
    .flatMap(lambda word: list(word)) \
    .take()
```

* sort

```scala
// in Scala
words
.sortBy(word => word.length() * -1)
.take(2)
```

```python
# in Python
words \
    .sortBy(lambda word: word.length() * -1) \
    .take(2)
```

* Random Splits

```scala
// in Scala
val fiftyFiftySplit = words.randomSplit(Array[Double](0.5, 0.5))
```

```python
# in Python 
fiftyFiftySplit = words.randomSplit([0.5, 0.5])
```

### 常用 Action 操作

Action 操作将触发基于 RDD 依赖关系的计算

#### collect

collect 操作将数据汇集到 Driver，数据过大时有超内存的风险

```python
rdd = sc.parallelize(range(10), 5)
all_data = rdd.collect()
all_data
```

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

#### take

take 操作将前若干个数据汇集到 Driver，相比 collect 安全





#### reduce

```scala
spark.sparkContext.parallelize(1 to 20)
    .reduce(_ + _) 
```

```python
spark.sparkContext.parallelize(range(1, 21)) \
    .reduce(lambda x, y: x + y)
```


#### count




#### countApprox




#### countApproxDistinct




#### countByValue




#### countByValueApprox




#### first

```scala
// in Scala
words.first()
```

```python
# in Python
words.first()
```

#### max/min





### Saving Files




### Caching





### Checkpointing



### Pipe RDDs to System Commands



# Key-Value RDD



# Distributed Shared Variables(分布式共享变量)
