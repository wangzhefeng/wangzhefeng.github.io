---
title: PySpark 介绍
author: 王哲峰
date: '2022-07-26'
slug: spark-pyspark
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

- [PySpark 简介](#pyspark-简介)
- [PySpark 内容](#pyspark-内容)
  - [Spark SQL 和 DataFrame](#spark-sql-和-dataframe)
  - [pandas API on Spark](#pandas-api-on-spark)
  - [Streaming](#streaming)
  - [MLlib](#mllib)
  - [Spark Core](#spark-core)
  - [PySpark 使用示例](#pyspark-使用示例)
- [参考](#参考)
</p></details><p></p>

# PySpark 简介

Apache Spark is written in Scala programming language. 
PySpark has been released in order to support the collaboration of Apache Spark and Python, 
it actually is a Python API for Spark. In addition, PySpark, 
helps you interface with Resilient Distributed Datasets (RDDs) in Apache Spark and Python programming language. 
This has been achieved by taking advantage of the `Py4J` library.

![img](images/pyspark_logo.png)

`Py4J` is a popular library which is integrated within PySpark and allows python to dynamically interface with JVM objects. PySpark features quite a few libraries for writing efficient programs. 
Furthermore, there are various external libraries that are also compatible. Here are some of them:

* PySparkSQL
* MLlib
* GraphFrames

# PySpark 内容

PySpark is an interface for Apache Spark in Python. 
It not only allows you to write Spark applications using Python APIs, 
but also provides the PySpark shell for interactively analyzing your data in a distributed environment. 
PySpark supports most of Spark’s features such as Spark SQL, DataFrame, 
Streaming, MLlib (Machine Learning) and Spark Core.

![pyspark](images/pyspark.png)

## Spark SQL 和 DataFrame

* structured data processing
* programming abstraction--DataFrame
* distributed SQL query engine

## pandas API on Spark

* pandas
* spark

## Streaming

* on top Spark
* interactive and analytical applications across both streaming and historical data

## MLlib

* scalable machine learning library

## Spark Core

* underlying general execution engine
* RDD(Resilient Distributed Dataset)
* in-memory computing capabilities

## PySpark 使用示例

```python
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import findspark

# 指定 spark_home，指定 Python 路径
spark_home = "/Users/zfwang/.pyenv/versions/3.7.10/envs/pyspark/lib/python3.7/site-packages/pyspark"
python_path = "/Users/zfwang/.pyenv/versions/3.7.10/envs/pyspark/bin/python"
findspark.init(spark_home, python_path)

conf = SparkConf().setAppName("test").setMaster("local[4]")
sc = SparkContext(conf = conf)

print("spark version:", pyspark.__version__)
rdd = sc.parallelize(["hello", "spark"])
print(rdd.reduce(lambda x,y: x + '' + y))
```

# 参考

* [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
* [Databricks PySpark](https://www.databricks.com/glossary/pyspark)

