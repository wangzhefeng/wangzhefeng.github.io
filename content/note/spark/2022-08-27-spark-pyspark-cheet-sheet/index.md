---
title: PySpark Cheet Sheet -- RDD Basic
author: 王哲峰
date: '2022-07-27'
slug: spark-pyspark-cheet-sheet
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

- [Spark](#spark)
- [Initializing Spark](#initializing-spark)
  - [SparkContext](#sparkcontext)
  - [Inspect SparkContext](#inspect-sparkcontext)
  - [Configuration](#configuration)
  - [Using The Shell](#using-the-shell)
- [Loading Data](#loading-data)
  - [Parallelized Collections](#parallelized-collections)
  - [External Data](#external-data)
- [Retrieving RDD Information](#retrieving-rdd-information)
  - [Basic Information](#basic-information)
  - [Summary](#summary)
- [Applying Functions](#applying-functions)
- [Selecting Data](#selecting-data)
  - [Getting](#getting)
  - [Sampling](#sampling)
  - [Filtering](#filtering)
- [Iterating](#iterating)
- [Reshaping Data](#reshaping-data)
  - [Reducing](#reducing)
  - [Grouping by](#grouping-by)
  - [Aggregating](#aggregating)
- [Mathematical Operations](#mathematical-operations)
- [Sort](#sort)
- [Repartitioning](#repartitioning)
- [Saving](#saving)
- [Stopping SparkContext](#stopping-sparkcontext)
- [Execution](#execution)
</p></details><p></p>




# Spark

PySpark is the Spark Python API that exposes the Spark programming model to Python.

# Initializing Spark

## SparkContext

```python
>>> from pyspark import SparkContext
>>> sc = SparkContext(master = "local[2]")
```

## Inspect SparkContext


```python
>>> from pyspark import SparkContext
>>> sc.version                # Retrieve SparkContext version
>>> sc.pythonVer              # Retrieve Python version
>>> sc.master                 # Master URL to connect to
>>> str(sc.sparkHome)         # Path where Spark is installed on worker nodes
>>> str(sc.sparkUser())       # Retrieve name of the Spark User running SparkContext
>>> sc.appName                # Return application name
>>> sc.applicationId          # Retrieve application ID
>>> sc.defaultParallelism     # Return default level of parallelism
>>> sc.defaultMinPartitions   # Default minimum number of partitions for RDDs
```

## Configuration


```python
from pyspark import SparkConf, SparkContext
conf = (
   SparkConf()
      .setMaster("local")
      .setAppName("My app")
      .set("spark.executor.memory", "lg")
)
sc = SparkContext(conf = conf)
```

## Using The Shell

```bash
$ ./bin/sparkshell master local[2]
$ ./bin/pyspark master local[4] pyfiles copy.py
```

# Loading Data

## Parallelized Collections


```python
rdd = sc.parallelize([("a", 7), ("a", 2), ("b", 2)])
rdd2 = sc.parallelize([("a", 2), ("d", 1), ("b", 1)])
rdd3 = sc.parallelize(range(100))
rdd4 = sc.parallelize([("a", ["x", "y", "z"]), ("b", ["p", "r"])])
```

## External Data

```python
textFile = sc.textFile("/my/directory/*.txt")
textFile2 = sc.wholeTextFiles("/my/directory/")
```

# Retrieving RDD Information

## Basic Information

```python
rdd.getNumPartitions()
rdd.count()
rdd.countByKey()
rdd.countByValue()
rdd.collectAsMap()
rdd3.sum()
sc.parallelize([]).isEmpty()
```

## Summary


```python
rdd3.max()
rdd3.min()
rdd3.mean()
rdd3.stdev()
rdd3.variance()
rdd3.histogram(3)
rdd3.stats()
```

# Applying Functions


```python
rdd.map(lambda x: x + (x[1], x[0])).collect()
rdd5 = rdd.flatMap(lambda x: x + (x[1], x[0]))
rdd5.collect()
rdd4.flatMapValues(lambda x: x).collect()
```

# Selecting Data

## Getting

```python
rdd.collect()
rdd.take(2)
rdd.first()
rdd.top(2)
```

## Sampling

```python
rdd3.sample(False, 0.15, 81).collect()
```

## Filtering


```python
rdd.filter(lambda x: "a" in x).collect()
rdd5.distinct().collect()
rdd.keys().collect()
```

# Iterating

```python
def g(x):
    print(x)

rdd.foreach(g)
```

# Reshaping Data

## Reducing

```python
rdd.reduceByKey(lambda x, y: x + y).collect()
rdd.reduce(lambda a, b: a + b)
```

## Grouping by

```python
rdd3.groupBy(lambda x: x % 2).mapValues(list).collect()
rdd.groupByKey().mapValues(list).collet()
```

## Aggregating

```python
seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
rdd3.aggregate((0, 0), seqOp, combOp)
rdd.aggregateByKey((0, 0), seqOp, combOp).collect()
rdd3.fold(0, add)
rdd.foldByKey(0, add).collect()
rdd3.keyBy(lambda x: x + x).collect()
```

# Mathematical Operations

```python
rdd.subtract(rdd2)
rdd2.subtractByKey(rdd)
rdd.cartesian(rdd2).collect()
```

# Sort

```python
rdd2.sortBy(lambda x: x[1]).collect()
rdd2.sortByKey().collect()
```

# Repartitioning

```python
rdd.repartition(4)
rdd.coalesce()
```

# Saving

```python
rdd.saveAsTextFile("rdd.txt")
rdd.saveAsHadoopFile("hdfs://namenodehost/parent/child", "org.apache.hadoop.mapred.TextOutputFormat")
```

# Stopping SparkContext

```python
sc.stop()
```

# Execution

```bash
$ ./bin/sparksubmit examples/src/main/python/pi.py
```

