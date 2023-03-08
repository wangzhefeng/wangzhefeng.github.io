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
- [PySpark 安装](#pyspark-安装)
  - [Spark 官方发布](#spark-官方发布)
  - [pip](#pip)
  - [Conda](#conda)
  - [手工安装](#手工安装)
  - [从源代码安装](#从源代码安装)
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

# PySpark 安装

> Python Version: >=Python 3.7

依赖：

| 包      | 最小支持版本 | 说明                                                            |
|---------|------------|----------------------------------------------------------------|
| pandas  | 1.0.5      | Optional for Spark SQL                                         |
| pyarrow | 1.0.0      | Optional for Spark SQL                                         |
| py4j    | 0.10.9.5   | Required                                                       |
| pandas  | 1.0.5      | Required for pandas API on Spark                               |
| pyarrow | 1.0.0      | Required for pandas API on Spark                               |
| numpy   | 1.15       | Required for pandas API on Spark and MLLib DataFrame-based API |
| Java    | >=8        | `JAVA_HOME`。如果用的是 Java 11，需要设置 `-Dio.netty.tryReflectionSetAccessible=true` |


## Spark 官方发布

* https://spark.apache.org/downloads.html

## pip

安装 PySpark：

```bash
$ pip install pyspark

# Spark SQL
$ pip install pyspark[sql]

# pandas API on Spark
$ pip install pyspark[pandas_on_spark] plotly
```

指定 Hadoop 版本：

```bash
# Hadoop 2
$ PYSPARK_HADOOP_VERSION=2 pip install pyspark

# PySpark 镜像，Hadoop 2
$ PYSPARK_RELEASE_MIRROR=http://mirror.apache-kr.org PYSPARK_HADOOP_VERSION=2 pip install

# Hadoop2，跟踪下载和安装状态
$ PYSPARK_HADOOP_VERSION=2 pip install pyspark -v
```

`PYSPARK_HADOOP_VERSION` 支持的值:

* `without`: Spark pre-built with user-provided Apache Hadoop
* `2`: Spark pre-built for Apache Hadoop 2.7
* `3`: Spark pre-built for Apache Hadoop 3.3 and later (default)

## Conda

创建 Conda 虚拟环境：

```bash
$ conda create -n pyspark_env
$ conda activate pyspark_env
```

安装 PySpark：

```bash
$ conda install -c conda-forge pyspark
```

## 手工安装

1. 下载 Apache Spark `spark-3.x.x-bin-hadoop3.tgz`
2. 解压 `spark-3.x.x-bin-hadoop3.tgz`

```bash
$ tar xzvf spark-3.x.x-bin-hadoop3.tgz
```

3. 设置环境变量

```bash
$ cd spark-3.x.x-bin-hadoop3
$ export SPARK_HOME=`pwd`
$ export PYTHONPATH=$(ZIPS=("$SPARK_HOME"/python/lib/*.zip); IFS=:; echo "${ZIPS[*]}"):$PYTHONPATH
```

## 从源代码安装

* https://spark.apache.org/docs/3.3.2/#downloading

# 参考

* [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
* [Databricks PySpark](https://www.databricks.com/glossary/pyspark)

