---
title: Spark 环境
author: 王哲峰
date: '2022-07-25'
slug: spark-env
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

- [Spark 开发环境](#spark-开发环境)
  - [语言 API](#语言-api)
  - [开发环境](#开发环境)
  - [交互式控制台](#交互式控制台)
  - [云平台](#云平台)
- [安装 Spark](#安装-spark)
- [安装 PySpark](#安装-pyspark)
  - [PyPI](#pypi)
    - [普通安装](#普通安装)
    - [支持 Spark SQL](#支持-spark-sql)
    - [pandas API on Spark](#pandas-api-on-spark)
  - [针对特定的 Hadoop 版本](#针对特定的-hadoop-版本)
    - [查看 Hadoop 版本](#查看-hadoop-版本)
    - [安装 pyspark](#安装-pyspark-1)
  - [Conda](#conda)
  - [手工安装](#手工安装)
  - [从源代码安装](#从源代码安装)
  - [依赖库](#依赖库)
    - [py4j](#py4j)
    - [Note](#note)
  - [云端环境](#云端环境)
  - [ERROR](#error)
    - [pyspark 版本问题](#pyspark-版本问题)
  - [参考](#参考)
- [Spark Shell](#spark-shell)
  - [进入 Spark Shell](#进入-spark-shell)
  - [Spark Shell 默认环境](#spark-shell-默认环境)
  - [运行 Python 脚本](#运行-python-脚本)
  - [Maven 依赖](#maven-依赖)
  - [基本操作](#基本操作)
    - [Python Version](#python-version)
    - [Scala Version](#scala-version)
  - [运行示例](#运行示例)
  - [参考](#参考-1)
</p></details><p></p>

# Spark 开发环境

## 语言 API

* Scala
    - Spark's "default" language.
* Java
* Python
    - ``pyspark``
* SQL
    - Spark support a subset of the ANSI SQL 2003 standard.
* R
    - Spark core
        - ``SparkR``
    - R community-driven package
        - ``sparklyr``

## 开发环境

- local
     - [Java(JVM)](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
     - [Scala](https://www.scala-lang.org/download/)
     - [Python interpreter(version 2.7 or later)](https://repo.continuum.io/archive/)
     - [R](https://www.r-project.org/)
     - [Spark](https://spark.apache.org/downloads.html)
- web-based version in [Databricks Community Edition](https://community.cloud.databricks.com/)

## 交互式控制台

* Python

```bash
./bin/pyspark
```

* Scala

```bash
./bin/spark-shell
```

* SQL

```bash
./bin/spark-sql
```

## 云平台

* [Databricks](https://community.cloud.databricks.com/)
* [Project's Github](https://github.com/databricks/Spark-The-Definitive-Guide)

# 安装 Spark

* 官方安装文档
    - https://spark.apache.org/downloads.html
    ![spark_org_download](images/spark_org_download.png)

# 安装 PySpark

PySpark 包含在 Apache Spark 网站上提供的 Spark 官方版本中。
对于 Python 用户，PySpark 还提供 pip 从 PyPI 安装。
这通常用于本地使用或作为客户端连接到集群，而不是设置集群本身

## PyPI

### 普通安装

```bash
$ pip install pyspark
$ pip install findspark
```

![pyspark_install](images/pyspark_install.png)

### 支持 Spark SQL

```bash
$ pip install "spark[sql]"
```

### pandas API on Spark

```bash
$ pip install "pyspark[pandas_on_spark]" plotly
```

## 针对特定的 Hadoop 版本

* Spark 默认使用 Hadoop 3.3 和 Hive 2.3
* `PYSPARK_HADOOP_VERSION` 可选值:
    - `without`: Spark pre-built with user-provided Apache Hadoop
    - `2`: Spark pre-built for Apache Hadoop 2.7
    - `3`: Spark pre-built for Apache Hadoop 3.3 and later (default)

### 查看 Hadoop 版本

```bash
$ ls /usr/local/Celler/hadoop
drwxr-xr-x  11 zfwang  admin   352B Jan 21  2022 3.3.1
```

### 安装 pyspark

```bash
# 设置 Hadoop 版本
$ PYSPARK_HADOOP_VERSION=3 pip install pyspark

# 设置镜像及 Hadoop 版本加速下载
$ PYSPARK_RELEASE_MIRROR=http://mirror.apache-kr.org PYSPARK_HADOOP_VERSION=2 pip install

# 跟踪安装和下载状态
$ PYSPARK_HADOOP_VERSION=2 pip install pyspark -v
```

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

PySpark 包含在Apache Spark 网站上提供的发行版中。
可以从该站点下载所需的发行版。之后，将 tar 文件解压到要安装 Spark 的目录下

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

## 依赖库

| Package | Minimum supported version | Note |
| ------- | ------------------------- | ---- |
| pandas | 1.0.5 | Optional for Spark SQL |
| NumPy | 1.7| Required for MLlib DataFrame-based API |
| pyarrow | 1.0.0 | Optional for Spark SQL |
| Py4J | 0.10.9.5 | Required |
| pandas | 1.0.5 | Required for pandas API on Spark |
| pyarrow | 1.0.0 | Required for pandas API on Spark |
| Numpy | 1.14 | Required for pandas API on Spark |

### py4j
* `py4j`: https://www.py4j.org/

![py4j](images/py4j.png)

### Note

* Note that PySpark requires Java 8 or later with `JAVA_HOME` properly set. 
  If using JDK 11, set `-Dio.netty.tryReflectionSetAccessible=true`
* Note for AArch64 (ARM64) users: `PyArrow` is required by PySpark SQL, 
  but `PyArrow` support for AArch64 is introduced in `PyArrow` 4.0.0. 
  If PySpark installation fails on AArch64 due to `PyArrow` installation errors, 
  you can install `PyArrow` >= 4.0.0 as below:

```bash
$ pip install "pyarrow>=4.0.0" --prefer-binary
```

## 云端环境

- [kesci](https://www.kesci.com/home/project)
    - 可以直接在 notebook 中运行 pyspark

## ERROR

### pyspark 版本问题

报错信息：

```
py4j.protocol.Py4JError: org.apache.spark.api.python.PythonUtils.getEncryptionEnabled does not exist in the JVM
```

解决方法：

```bash
$ pip install pyspark=="2.4.7"
```

## 参考

* [PySpark 资料](https://spark.apache.org/docs/latest/api/python/index.html)

# Spark Shell

## 进入 Spark Shell

```bash
$ cd usr/lib/spark/bin # linux
$ cd D:/spark/bin      # windows
```

**Python version:**

```bash
$ pyspark --help
$ pyspark --master --py-files --packages -- repositories
$ pyspark                                               
$ PYSPARK_PYTHON=python3.6 pyspark
$ PYSPARK_DRIVER_PYTHON=ipython pyspark
$ PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS=notebook pyspark
```

**Scala version:**

```bash
$ spark-shell --help
$ spark-shell --master --jars --packages --repositories
$ spark-shell
```

## Spark Shell 默认环境

* Spark context Web UI: 
    - http://192.168.0.111:4040
* Spark context: 
    - `sc (master = local[*], app id = local-1556297140303)`
* Spark session: 
    - `spark`

## 运行 Python 脚本

```bash
# run Spark application in python without pip install PySpark(pip install pyspark)
$ /bin/spark-submit my_script.py                          # python
$ PYSPARK_PYTHOH=python3.6 /bin/spark-submit my_scrity.py # python with specify version

# run Spark application in python with pip install PySpark(pip install pyspark)
$ python my_script.py
```

## Maven 依赖

添加一个对于 spark-core 工件的 Maven 依赖 

``` 
// java & scala
groupId = org.apache.spark
artifactid = spark-core_2.10
version = 2.3.0
```

## 基本操作

### Python Version

Python 基本操作：

```python
# 创建一个DataFrame
textFile = spark.read.text("README.md")

# action, transformations
textFile.count()
textFile.first()

# 转换为一个新的DataFrame
lineWithSpark = textFile.filter(textFile.value.contains('Spark'))
lineWithSpark.count()

# or 
textFile.filter(textFile.value.contains('Spark')).count()

# Dataset Transform
from pyspark.sql.functions import *
wordCounts = textFile
    .select(size(split(textFile.value, "\\s+")).name("numWords")) \
    .agg(max(col("numWords"))) \
    .collect()

# MapReduce
wordCounts = textFile \
    .select(explode(split(textFile.value, "\\s+")).alias("word")) \
    .groupBy("word")
    .count()
wordCounts.collect()

# 缓存
lineWithSpark.cache()
lineWithSpark.count()
lineWithSpark.count()
```

Python App：

```python
# setup.py
install_requires = [
    'pyspark=={site.SPARK_VERSION}'
]

# SimpleApp.py
from pyspark.sql import SparkSession

logFile = "D:/spark/README.md"  # Should be some file on your system
spark = SparkSession.builder \
    .appName("SimpleApp") \
    .getOrCreate()
logData = spark.read.text(logFile).cache()

numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()
```

```bash
# Use spark-submit to run your application
$ D:/spark/bin/spark-submit --master local[4] SimpleApp.py

# Use the Python interpreter to run your application(安装了PySpark pip: pip install pyspark)
$ python SimpleApp.py
```

### Scala Version

Scala 基本操作：

```scala
// 创建一个Dataset
val textFile = spark.read.textFile("README.md")

// action, transformations
textFile.count()
textFile.first()


// 转换为一个新的Dataset
val linesWithSpark = textFile.filter(line => line.contains("Spark"))

// or 

textFile.filter(line => line.contains("Spark")).count()


// Dataset transform
textFile.map(line => line.split(" ").size).reduce((a, b) => if (a > b) a else b)

// or

import java.lang.Math
textFile.map(line => line.split(" ").size).reduce((a, b) => Math.max(a, b))

// MapReduce
val wordCounts = textFile.flatMap(line => line.split(" ")).groupByKey(identity).count()
wordCounts.collect()

lineWithSpark.cache()
lineWithSpark.count()
lineWithSpark.count()
```

Scala App：

```scala
/* SimpleApp.scala */
import org.apache.spark.sql.SparkSession

object SimpleApp {
    def main(args: Array[String]) {
        val logFile = "D:/spark/README.md" // Should be some file on your system
        val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
        val logData = spark.read.textFile(logFile).cache()
        val numAs = logData.filter(line => line.contains("a")).count()
        val numBs = logData.filter(line => line.contains("b")).count()
        println(s"Lines with a: $numAs, Lines with b: $numBs")
        spark.stop()
    }
}
```

sbt configuration file: build.sbt

```
name := "Simple Project"

version := "1.0"

scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
```

```bash
# Your directory layout should look like this
$ find .
.
./build.sbt
./src
./src/main
./src/main/scala
./src/main/scala/SimpleApp.scala

# Package a jar containing your application
$ sbt package

# Use spark-submit to run your application
$ YOUR_SPARK_HOME/bin/spark-submit 
    --class "SimpleApp" 
    --master local[4] 
    target/scala-2.11/simple-project_2.11-1.0.jar
```

## 运行示例

```bash
# For Scala and Java, use run-example:
./bin/run-example SparkPi

# For Python examples, use spark-submit directly:
./bin/spark-submit examples/src/main/python/pi.py

# For R examples, use spark-submit directly:
./bin/spark-submit examples/src/main/r/dataframe.R
```

## 参考

* [Submitting Applications](https://spark.apache.org/docs/latest/submitting-applications.html)
