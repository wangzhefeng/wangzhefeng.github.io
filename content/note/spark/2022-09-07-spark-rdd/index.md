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
- [RDD](#rdd)
  - [创建 RDD](#创建-rdd)
    - [DataFrame, Dataset, RDD 交互操作](#dataframe-dataset-rdd-交互操作)
    - [从 Local Collection 创建 RDD](#从-local-collection-创建-rdd)
    - [从数据源创建 RDD](#从数据源创建-rdd)
  - [操作 RDD](#操作-rdd)
    - [Transformation](#transformation)
    - [Action](#action)
    - [Saving Files](#saving-files)
    - [Caching](#caching)
    - [Checkpointing](#checkpointing)
    - [Pipe RDDs to System Commands](#pipe-rdds-to-system-commands)
- [Key-Value RDD](#key-value-rdd)
- [Distributed Shared Variables(分布式共享变量)](#distributed-shared-variables分布式共享变量)
</p></details><p></p>

# Spark Low-Level API

- What are the Low-Level APIs ?
- Resilient Distributed Dataset (RDD)
- Distributed Shared Variables
     - Accumulators
     - Broadcast Variable
- When to Use the Low-Level APIs ?
- 在高阶 API 中针对具体问题没有可用的函数时；
- Maintain some legacy codebase written using RDDs;
- 需要进行自定义的共享变量操作时；
- How to Use the Low-Level APIs ?
- `SparkContext` 是 Low-Level APIs 的主要入口:
     - `SparkSession.SparkContext`
     - `spark.SparkContext`


# RDD

- RDD 创建
- RDD 操作 API
- RDD 持久化
- RDD 分区

## 创建 RDD

### DataFrame, Dataset, RDD 交互操作

**从 DataFrame 或 Dataset 创建 RDD:**

```scala
// in Scala: converts a Dataset[Long] to  RDD[Long]
spark.range(500).rdd

// convert Row object to correct data type or extract values
spark.range(500).toDF().rdd.map(rowObject => rowObject.getLong(0))
```

```python
# in Python: converts a DataFrame to RDD of type Row
spark.range(500).rdd

spark.range(500).toDF().rdd.map(lambda row: row[0])
```

**从 RDD 创建 DataFrame 和 Dataset:**

```scala
// in Scala
spark.range(500).rdd.toDF()
```

```python
# in Python
spark.range(500).rdd.toDF()
```


### 从 Local Collection 创建 RDD

- `SparkSession.SparkContext.parallelize()`

```scala
// in Scala
val myCollection = "Spark The Definitive Guide: Big Data Processing Made Simple"
.split(" ")
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
```

### 从数据源创建 RDD

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


## 操作 RDD

- 操作 raw Java or Scala object instead of Spark types;

### Transformation


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

### Action

* reduce

```scala
spark.sparkContext.parallelize(1 to 20)
    .reduce(_ + _) 
```

```python
spark.sparkContext.parallelize(range(1, 21)) \
    .reduce(lambda x, y: x + y)
```


* count




* countApprox




* countApproxDistinct




* countByValue




* countByValueApprox




* first

```scala
// in Scala
words.first()
```

```python
# in Python
words.first()
```

* max/min

* take




### Saving Files




### Caching





### Checkpointing



### Pipe RDDs to System Commands



# Key-Value RDD



# Distributed Shared Variables(分布式共享变量)
