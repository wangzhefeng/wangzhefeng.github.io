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
    - [map](#map)
    - [filter](#filter)
    - [flatMap](#flatmap)
    - [sample](#sample)
    - [distinct](#distinct)
    - [subtract](#subtract)
    - [union](#union)
    - [intersection](#intersection)
    - [cartesian](#cartesian)
    - [sortBy](#sortby)
    - [zip](#zip)
    - [zipWithIndex](#zipwithindex)
    - [sort](#sort)
    - [Random Splits](#random-splits)
  - [常用 Action 操作](#常用-action-操作)
    - [collect](#collect)
    - [take](#take)
    - [takeSample](#takesample)
    - [first](#first)
    - [count](#count)
    - [reduce](#reduce)
    - [foreach](#foreach)
    - [countByKey](#countbykey)
    - [saveAsTextFile](#saveastextfile)
  - [常用 PairRDD 转换操作](#常用-pairrdd-转换操作)
  - [RDD 缓存操作和设置检查点](#rdd-缓存操作和设置检查点)
    - [缓存操作](#缓存操作)
    - [设置检查点](#设置检查点)
    - [示例](#示例)
  - [RDD 持久化](#rdd-持久化)
  - [RDD 分区](#rdd-分区)
    - [glom](#glom)
    - [coalesce](#coalesce)
    - [repartition](#repartition)
    - [partitionBy](#partitionby)
    - [HashPartitioner](#hashpartitioner)
    - [RangePartitioner](#rangepartitioner)
    - [TaskContext](#taskcontext)
    - [mapPartitions](#mappartitions)
    - [mapPartitionsWithIndex](#mappartitionswithindex)
    - [foreachPartition](#foreachpartition)
    - [aggregate](#aggregate)
    - [aggregateByKey](#aggregatebykey)
  - [RDD 分桶](#rdd-分桶)
- [分布式共享变量](#分布式共享变量)
  - [广播变量](#广播变量)
  - [累加器](#累加器)
- [RDD 编程示例](#rdd-编程示例)
- [Spark 应用依赖](#spark-应用依赖)
- [Spark 初始化](#spark-初始化)
- [RDDs (Resilent Distributed Datasets)](#rdds-resilent-distributed-datasets)
  - [创建 RDD](#创建-rdd-1)
  - [RDD 操作](#rdd-操作)
</p></details><p></p>


Spark 配置:

```python

```

PySpark 配置:

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

# Spark Low-Level API

## What are the Low-Level APIs ?

* Resilient Distributed Dataset (RDD)
    * RDD 创建
    * RDD 操作 API
        - Action
        - Transformation
        - Pair RDD Transformation
    * RDD 缓存操作和设置检查点
    * RDD 持久化
    * RDD 分区
    * 共享变量
* Distributed Shared Variables (分布式共享变量)
    - Accumulators(累加器)
    - Broadcast Variable(广播变量)

## When to Use the Low-Level APIs ?

- 在高阶 API 中针对具体问题没有可用的函数时
- Maintain some legacy codebase written using RDDs
- 需要进行自定义的共享变量操作时

## How to Use the Low-Level APIs ?

`SparkContext` 是 Low-Level APIs 的主要入口:

* `SparkSession.SparkContext`
* `spark.SparkContext`

# 创建 RDD

## DataFrame, Dataset, RDD 交互操作

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

## 从 Local Collection 创建 RDD

### Spark

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

### PySpark

```python
rdd = sc.parallelize(range(1, 11), 2)
rdd.collect()
```

```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


## 从数据源创建 RDD

使用 `textFile` 加载本地或者集群文件系统中的数据

### Spark

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

### PySpark

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

# 操作 RDD

操作 raw Java or Scala object instead of Spark types

## 常用 Transformation 操作

Transformation 转换操作具有懒惰执行的特性，
它只指定新的 RDD 和其父 RDD 的依赖关系，
只有当 Action 操作触发到该依赖的时候，它才被计算

### map

map 操作对每个元素进行一个映射转换

* Spark

```scala
// in Scala
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

* PySpark

```python
rdd = sc.parallelize(range(10), 3)
rdd.map(lambda x: x ** 2)
rdd.collect()
```

```
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### filter

filter 应用过滤条件过滤掉一些数据

* Spark

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

* PySpark

```python
rdd = sc.parallelize(range(10), 3)
rdd.filter(lambda x: x > 5)
rdd.collect()
```

```
[6, 7, 8, 9]
```

### flatMap

flatMap 操作执行将每个元素生成一个 Array 后压平

* Spark

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

* PySpark

```python
rdd = sc.parallelize([
    "hello world",
    "hello China",
])
rdd.map(lambda x: x.split(" "))
rdd.collect()

rdd.flatMap(lambda x: x.split(" "))
rdd.collect()
```

```
[["hello", "world"],
 ["hello", "China"]]

["hello", "world", "hello", "China"]
```

### sample

sample 对原 RDD 在每个分区按照比例进行抽样，第一个参数设置是否可以重复抽样

```python
rdd = sc.parallelize(range(10), 1)
rdd.sample(False, 0.5, 0)
rdd.collect()
```

```
[1, 4, 9]
```

### distinct 

distinct 去重

* Spark

```scala
// in Scala
words
    .distinct()
    .count()
```

* PySpark

```python
rdd = sc.parallelize([1, 1, 2, 2, 3, 3, 4, 5])
rdd.distinct().collect()
```

```
[4, 1, 5, 2, 3]
```

### subtract

subtract 找到属于前一个 RDD 而不属于后一个 RDD 的元素

```python
a = sc.parallelize(range(10))
b = sc.parallelize(range(5, 15))
a.subtract(b).collect()
```

```
[0, 1, 2, 3, 4]
```

### union

union 合并数据

```python
a = sc.parallelize(range(5))
b = sc.parallelize(range(3, 8))
a.union(b).collect()
```

```
[0, 1, 2, 3, 4, 3, 4, 5, 6, 7]
```

### intersection

intersection 求交集

```python
a = sc.parallelize(range(1, 6))
b = sc.parallelize(range(3, 9))
a.intersection(b).collect()
```

```
[3, 4, 5]
```

### cartesian

cartesian 笛卡尔积

```python
boys = sc.parallelize(["LiLei", "Tom"])
girls = sc.parallelize(["HanMeiMei", "Lily"])
boys.cartesian(girls).collect()
```

```
[("LiLei", "HanMeiMei"),
 ("LiLei", "Lily"),
 ("Tom", "HanMeiMei"),
 ("Tom", "Lily")]
```

### sortBy

sortBy 按照某种方式进行排序，指定按照第 3 个元素大小进行排序

```python
rdd = sc.parallelize([(1, 2, 3), (3, 2, 2), (4, 1, 1)])
rdd.sortBy(lambda x: x[2]).collect()
```

```
[(4, 1, 1), (3, 2, 2), (1, 2, 3)]
```

### zip

zip 按照拉链的方式连接两个 RDD，效果类似 Python 的 `zip` 函数，
需要两个 RDD 具有相同的分区，每个分区元素数量相同

```python
rdd_name = sc.parallelize(["LiLei", "Hanmeimei", "Lily"])
rdd_age = sc.parallelize([19, 18, 20])

rdd_zip = rdd_name.zip(rdd_age)
print(rdd_zip.collect())
```

```
[("LiLei", 19), ("Hanmeimei", 18), ("Lily", 20)]
```

### zipWithIndex

将 RDD 和一个从 0 开始的递增序列按照拉链方式连接

```python
rdd_name = sc.parallelize([
    "LiLei", "Hanmeimei", "Lily", "Lucy", "Ann", "Dachui", "RuHua"
])
rdd_index = rdd_name.zipWithIndex()
print(rdd_index.collect())
```

```
[('LiLei', 0), 
 ('Hanmeimei', 1), 
 ('Lily', 2), 
 ('Lucy', 3), 
 ('Ann', 4), 
 ('Dachui', 5), 
 ('RuHua', 6)]
```

### sort

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

### Random Splits

```scala
// in Scala
val fiftyFiftySplit = words.randomSplit(Array[Double](0.5, 0.5))
```

```python
# in Python 
fiftyFiftySplit = words.randomSplit([0.5, 0.5])
```

## 常用 Action 操作

Action 操作将触发基于 RDD 依赖关系的计算

### collect

collect 操作将数据汇集到 Driver，数据过大时有超内存的风险

```python
rdd = sc.parallelize(range(10), 5)
all_data = rdd.collect()
all_data
```

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### take

take 操作将前若干个数据汇集到 Driver，相比 collect 安全

```python
rdd = sc.parallelize(range(10), 5)
part_data = rdd.take(4)
part_data
```

```
[0, 1, 2, 3]
```

### takeSample

takeSample 可以随机取若干个到 Driver，第一个参数设置是否放回抽样

```python
rdd = sc.parallelize(range(10), 5)
sample_data = rdd.takeSample(False, 10, 0)
sample_data
```

```
[7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
```

### first

first 取第一个数据

* Spark

```scala
// in Scala
words.first()
```

```python
# in Python
words.first()
```

* PySpark

```python
rdd = sc.parallelize(range(10), 5)
first_data = rdd.first()
print(first_dasta)
```

```
0
```

### count

count 查看 RDD 元素数量

```python
rdd = sc.parallelize(range(10), 5)
data_count = rdd.count()
print(data_count)
```

```
10
```

### reduce

reduce 利用二元函数对数据进行规约

* Spark

```scala
spark.sparkContext.parallelize(1 to 20)
    .reduce(_ + _) 
```

```python
spark.sparkContext.parallelize(range(1, 21)) \
    .reduce(lambda x, y: x + y)
```

* PySpark

```python
rdd = sc.parallelize(range(10), 5)
rdd.reduce(lambda x, y: x + y)
```

### foreach

foreach 对每一个元素执行某种操作，不生成新 RDD

```python
# 累加法用法需要参考共享变量
accum = sc.accumulator(0)

rdd = sc.parallelize(range(10), 5)
rdd.foreach(lambda x: accum.add(x))
print(accum.value)
```

```
45
```

### countByKey

countByKey 对 Pair RDD 按 key 统计数量

```python
pairRdd = sc.parallelize([(1, 1), (1, 4), (3, 9), (2, 16)])
pairRdd.countByKey()
```

### saveAsTextFile

saveAsTextFile 保存 rdd 成 text 文件到本地

```python
text_file = "./data/rdd.txt"

rdd = sc.parallelize(range(5))
rdd.saveAsTextFile(text_file)

# 重新读入会被解析为文本
rdd_loaded = sc.textFile(text_file)
rdd_loaded.collect()
```

```
['2', '3', '4', '1', '0']
```

## 常用 PairRDD 转换操作

Pair 




## RDD 缓存操作和设置检查点

### 缓存操作

如果一个 RDD 被多个任务用作中间变量，那么对其进行 cache 缓存到内存中对加快计算会非常有帮助

* 声明对一个 RDD 进行 cache 后，该 RDD 不会被立即缓存，而是等到它第一次被计算出来时才进行缓存。
可以使用 `persist` 明确指定存储级别，常用的存储级别是 `MEMORY_ONLY` 和 `EMORY_AND_DISK`
* 如果一个 RDD 后面不再用到，可以用 `unpersist` 释放缓存，`unpersist` 是立即执行的

缓存数据不会切断血缘依赖关系，这是因为缓存数据数据某些分区所在的节点有可能会有故障，例如内存溢出或者节点损坏。
这时可以根据血缘关系重新计算这个分区的数据

### 设置检查点

如果要切断血缘关系，可以用 checkpoint 设置检查点将某个 RDD 保存到磁盘中

声明对一个 RDD 进行 checkpoint 后，该 RDD 不会被立即保存到磁盘，
而是等到它第一次被计算出来时才保存成检查点

通常只对一些计算代价非常高昂的中间结果或者重复计算结果不可保证完全一致的情形下使用，如 `zipWithIndex`

### 示例

1. cache 缓存到内存中，使用存储级别 MEMORY_ONLY。
   MEMORY_ONLY 意味着如果内存存储不下，放弃存储其余部分，需要时重新计算

```python
# RDD 创建
a = sc.parallelize(range(10000), 5)
# RDD 缓存
a.cache()

# RDD 操作
sum_a = a.reduce(lambda x, y: x + y)
cnt_a = a.count()
mean_a = sum_a / cnt_a
print(mean_a)
```

2. persist 缓存到内存或磁盘中，默认使用存储级别 MEMORY_AND_DISK。
   MEMORY_AND_DISK 意味着如果内存存储不下，其余部分存储在磁盘中。
   persist 可以指定其他存储级别，cache 相当于 persist(MEMORY_ONLY)

```python
from pyspark.storagelevel import StorageLevel

# RDD 创建
a = sc.parallelize(range(10000), 5)

# RDD 
a.persist(StorageLevel.MEMORY_AND_DISK)

# RDD 操作
sum_a = a.reduce(lambda x, y: x + y)
cnt_a = a.count()
mean_a = sum_a / cnt_a

# 立即释放缓存
a.unpersist()
print(mean_a)
```

3. 将数据设置成检查点，写入到磁盘中

```python
sc.setCheckpointDir("./data/checkpoint/")
rdd_students = sc.parallelize(["LiLei", "Hanmeimei", "LiLy", "Ann"], 2)
rdd_students_idx = rdd_students.zipWithIndex()
```

4. 设置检查点后，可以避免重复计算，不会因为 `zipWithIndex` 重复计算触发不一致的问题

```python
rdd_students_idx.checkpoint()
rdd_students_idx.take(3)
```

## RDD 持久化


## RDD 分区

分区操作包括改变分区操作，以及针对分区执行的一些转换操作:

* `glom`
    - 将一个分区内的数据转换为一个列表作为一行
* `coalesce`
    - `shuffle` 可选，默认为 `False` 情况下窄依赖，不能增加分区
    - `repartition` 和 `partitionBy` 调用它实现
* `repartition`
    - 按随机数进行 shuffle，相同 key 不一定在同一个分区
* `partitionBy`
    - 按 key 进行 shuffle，相同 key 放入同一个分区
* `HashPartitioner`
    - 默认分区器，根据 key 的 hash 值进行分区，相同的 key 进入同一分区，效率较高，key 不可为 Array
* `RangePartitioner`
    - 只在排序相关函数中使用，除相同的 key 进入同一分区，相邻的 key 也会进入同一分区，key 必须可排序
* `TaskContext`
    - 获取当前芬奇 id 的方法: `TaskContext.get.partitionId`
* `mapPartitions`
    - 每次处理分区内的一批数据，适合需要分批处理数据的情况，
      比如将数据插入某个表，每批数据只需要开启一次数据库连接，
      大大减少了连接开支
* `mapPartitionsWithIndex`
    - 类似 `mapPartitions`，但提供了分区索引，输入参数为 (i, Iterator)
* `foreachPartition`
    - 类似 `foreach`，但每次提供一个 Partition 的一批数据
* `aggregate`
* `aggregateByKey`

### glom

### coalesce

### repartition

### partitionBy

### HashPartitioner

### RangePartitioner

### TaskContext

### mapPartitions

### mapPartitionsWithIndex

### foreachPartition

### aggregate

### aggregateByKey

## RDD 分桶


# 分布式共享变量

当 Spark 集群在多个节点上运行一个函数时，默认情况下会把这个函数涉及到的对象在每个节点生成一个副本。
但是，有时候需要在不同节点或者节点和 Driver 之间共享变量

Spark 提供两种类型的共享变量 Distributed Shared Variables(分布式共享变量)，广播变量和累加器

## 广播变量

广播变量是不可变变量，在所有节点可读，
实现在不同节点不同任务之间共享数据

广播变量在每个节点机器上缓存一个只读的变量，
而不是为每个 Task 生成一个副本，可以减少数据的传输

* 广播变量

```python
broads = sc.broadcast(100)
print(broads.value)
```

```
100
```

* 节点 RDD 操作

```python
rdd = sc.parallelize(range(10))
rdd.map(lambda x: x + broads.value).collect()
```

```
[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
```

## 累加器

累加器主要是不同节点和 Driver 之间共享变量，只能实现计数或者累加功能。
累加器的值只有在 Driver 上是可读的，在其他节点上不可见，只能进行累加

* 示例 1

```python
# 累加器
total = sc.accumulator(0)
# RDD
rdd = sc.parallelize(range(10), 3)
# RDD 操作
rdd.foreach(lambda x: total.add(x))

total.value
```

```
45
```

* 示例 2

```python
# 累加器
total = sc.accumulator(0)
count = sc.accumulator(0)
# RDD
rdd = sc.parallelize([1.1, 2.1, 3.1, 4.1])

def func(x):
    total.add(x)
    count.add(1)

# RDD 操作
rdd.foreach(func)

total.value / count.value
```

```
2.6
```

# RDD 编程示例





# Spark 应用依赖

Spark 的 Maven 依赖：

```
groupId = org.apache.spark
artifactId = spark-core_2.12
version = 2.4.4
```

HDFS 集群的依赖:

```
groupId = org.apache.hadoop
artifactId = hadoop-client
version = <your-hdfs-version>
```

Spark 基本类：

```scala
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
```

# Spark 初始化

- 创建 `SparkContext` 对象, 用来连接到集群(cluster)

```scala
val conf = new SparkConf().setAppName("appName").setMaster("master") // "local"
val sc = new SparkContext(conf)
```

- Shell

```bash
$ ./bin/spark-shell --master local[4]
$ ./bin/spark-shell --master local[4] --jars code.jar
$ ./bin/spark-shell --master local[4] --packages "org.example:example:0.1"
```

# RDDs (Resilent Distributed Datasets)

## 创建 RDD

创建 RDD 的方法：

- 并行化驱动程序中的已有数据集合
- 引用外部存储系统中的数据集

(1) 并行化驱动程序中的已有数据集合

```scala
val conf = new SparkConf().setAppName("appName").setMaster("master") // "local"
val sc = new SparkContext(conf)

val data = Array(1, 2, 3, 4, 5)
val distData = sc.parallelize(data, 10)
```

(2) 引用外部存储系统中的数据集

外部存储系统：

- local file system
- HDFS
- Cassandra
- HBase
- Amazon S3
- ...

数据类型：

- text files
   - csv
   - tsv
   - Plain Text
   - ...
- SequenceFiles
- Hadoop InputFormat

```scala
// text files
val distFile = sc.textFile("data.txt")
val data = sc.wholeTextFiles()

// SequneceFiles
val data = sc.sequenceFile[K, V]

// Hadoop Input
val data = sc.hadoopRDD()
val data = sc.newAPIHadoopRDD()
```

```scala
RDD.saveAsObjectFile()
sc.objectFile()
```

## RDD 操作

