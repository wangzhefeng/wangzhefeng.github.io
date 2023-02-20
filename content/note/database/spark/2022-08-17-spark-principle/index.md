---
title: Spark 基本原理
author: 王哲峰
date: '2022-08-17'
slug: spark-principle
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

- [Spark 优势特点](#spark-优势特点)
  - [高效性](#高效性)
  - [易用性](#易用性)
  - [通用性](#通用性)
  - [兼容性](#兼容性)
- [Spark 基本概念](#spark-基本概念)
  - [基本概念](#基本概念)
  - [Application 总结](#application-总结)
- [Spark 架构设计](#spark-架构设计)
  - [Spark](#spark)
  - [PySpark](#pyspark)
- [Spark 运行流程](#spark-运行流程)
  - [运行流程](#运行流程)
  - [运行架构特点](#运行架构特点)
- [Spark 部署方式](#spark-部署方式)
  - [部署方式](#部署方式)
  - [Hadoop 和 Spark 的统一部署](#hadoop-和-spark-的统一部署)
- [RDD 数据结构](#rdd-数据结构)
  - [RDD 简介](#rdd-简介)
  - [RDD 创建](#rdd-创建)
  - [RDD 操作](#rdd-操作)
  - [RDD 特性](#rdd-特性)
  - [RDD 依赖](#rdd-依赖)
    - [RDD 窄依赖](#rdd-窄依赖)
    - [RDD 宽依赖](#rdd-宽依赖)
    - [DAG 切分为 Stage](#dag-切分为-stage)
- [Apache Spark](#apache-spark)
  - [Spark 的设计哲学和历史](#spark-的设计哲学和历史)
  - [Spark 开发环境](#spark-开发环境)
  - [Spark's Interactive Consoles](#sparks-interactive-consoles)
  - [云平台、数据](#云平台数据)
- [Spark](#spark-1)
  - [Spark's Architecture](#sparks-architecture)
  - [Spark's Language API](#sparks-language-api)
  - [Spark's API](#sparks-api)
  - [Spark 使用](#spark-使用)
    - [SparkSession](#sparksession)
    - [DataFrames](#dataframes)
    - [Partitions](#partitions)
    - [Transformation](#transformation)
      - [Lazy Evaluation](#lazy-evaluation)
    - [Action](#action)
    - [Spark UI](#spark-ui)
    - [一个 🌰](#一个-)
- [Spark 工具](#spark-工具)
  - [Spark 应用程序](#spark-应用程序)
  - [Dataset: 类型安全的结果化 API](#dataset-类型安全的结果化-api)
  - [Spark Structured Streaming](#spark-structured-streaming)
    - [创建一个静态数据集 DataFrame 以及 Schema](#创建一个静态数据集-dataframe-以及-schema)
    - [对数据进行分组和聚合操作](#对数据进行分组和聚合操作)
    - [设置本地模型运行参数配置](#设置本地模型运行参数配置)
    - [将批处理代码转换为流处理代码](#将批处理代码转换为流处理代码)
  - [Spark 机器学习和高级数据分析](#spark-机器学习和高级数据分析)
  - [Spark 低阶 API](#spark-低阶-api)
  - [SparkR](#sparkr)
  - [Spark 生态系统和工具包](#spark-生态系统和工具包)
- [WordCount 示例](#wordcount-示例)
- [参考](#参考)
</p></details><p></p>

# Spark 优势特点

> 作为大数据计算框架 MapReduce 的继任者，Spark 具有以下优势特性

![img](images/spark1.png)

* 高效性：运行速度快
    - 使用 DAG 执行引擎以支持循环数据流与内存计算
* 易用性
    - 支持使用 Scala、Java、Python 和 R 语言进行编程，可以通过 Spark Shell 进行交互式编程 
* 通用性
    - Spark 提供了完整而强大的技术栈，包括 SQL 查询、流式计算、机器学习和图算法组件
* 兼容性：运行模式多样
    - 可运行于独立的集群模式中，可运行于 Hadoop 中，
      也可运行于 Amazon EC2 等云环境中，
      并且可以访问 HDFS、Cassandra、HBase、Hive 等多种数据源 

## 高效性

不同于 MapReduce 将中间计算结果放入磁盘中，Spark 采用内存存储中间计算结果，
减少了迭代运算磁盘 IO，并通过并行计算 DAG 图的优化，减少了不同任务之间的依赖，
降低了延迟等待时间。内存计算下，Spark 比 MapReduce 块 100 倍

## 易用性

不同于 MapReduce 仅支持 Map 和 Reduce 两种编程算子，
Spark 提供了超过 80 中不同的 Transformation 和 Action 算子，
如 map, reduce, filter, groupByKey, sortByKey, foreach 等，
并且采用函数式编程风格，实现相同的功能需要的代码量极大缩小

MapReduce 和 Spark 对比：

| Item              | MapReduce                               | Spark                                         |
|-------------------|-----------------------------------------|-----------------------------------------------|
| 数据存储结构        | 磁盘 HDFS 文件系统的分割                   | 使用内存构建弹性分布数据集(RDD)对数据进行运算和 cache |
| 编程范式           | Map + Reduce                             | DAG: Transformation + Action                  |
| 计算中间结果处理方式 | 计算中间结果写入磁盘，IO及序列化、反序列化代价大 | 中间计算结果在内存中维护，存取速度比磁盘高几个数量级   |
| Task 维护方式      | Task 以进程的方式维护                       | Task 以线程的方式维护                            |


## 通用性

![img](images/spark2.png)

Spark 提供了统一的解决方案。Spark 可以用于批处理、交互式查询(Spark SQL)、
实时流式计算(Spark Streaming)、机器学习(Spark MLlib)和图计算(GraphX)。
这些不同类型的处理都可以在同一个应用中无缝使用，这对于企业应用来说，
就可以使用一个平台来进行不同的工程实现，减少了人力开发和平台部署成本

## 兼容性

![img](images/spark3.png)

Spark 能够跟很多开源工程兼容使用，如 Spark 可以使用 Hadoop 的 YARN 和 Apache Mesos 作为它的资源管理和调度器，
并且 Spark 可以读取多种数据源，如 HDFS、HBase、MySQL 等

# Spark 基本概念

## 基本概念

* RDD
    - 弹性分布式数据集(Resilient Distributed Dataset)的简称，
      是分布式内存的一个抽象概念，提供了一种高度受限的共享内存模型
* DAG
    - Directed Acyclic Graph(有向无环图)的简称，反映 RDD 之间的依赖关系
* Master Node
    - 每个 Master Node 上存在一个 Driver Program
    - Driver Program
        - 控制程序，负责为 Application 构建 DAG 图
* Cluster Manager
    - 集群资源管理中心，负责分配计算资源
* Worker Node
    - 工作节点，负责完成具体计算
    - 每个 Worker Node 上存在一个 Executor 进程
* Executor
    - 运行在 Worker Node 上的一个进程
    - 负责运行 Task，一个 Executor 进程中包含多个 Task 线程
    - 并为应用程序存储数据
* Application
    - 用户编写的 Spark 应用程序
    - 一个 Application 包含多个 Job
* Job
    - 作业
    - 一个 Job 包含多个 RDD 及作用于相应 RDD 上的各种操作
* Stage
    - 阶段，Job 的基本调度单位
    - 一个 Job 会分为多组任务，每组任务被称为 Stage
* Task
    - 任务运行在 Executor 上的工作单元，是 Executor 中的一个线程

## Application 总结

Application 由多个 Job 组成，Job 由多个 Stage 组成，
Stage 由多个 Task 组成。Stage 是 Task 调度的基本单位

```
Application [Driver]
    - Job 1
        - Stage 1
            - Task 1 [Executor]
            - Task 2
            - ...
            - Task p
        - Stage 2
        - ...
        - Stage n
    - Job 2
    - ...
    - Job m
```

# Spark 架构设计

Spark 集群由 Driver、Cluster Manager(Standalone, Yarn 或 Mesos)，以及 Worker Node 组成

## Spark

对于每个 Spark 应用程序，Worker Node 上存在一个 Executor 进程，Executor 进程中包括多个 Task 线程

![img](images/spark架构设计.png)

## PySpark

对于 PySpark，为了不破坏 Spark 已有的运行架构，Spark 在外围包装了一层 Python API

* 在 Driver 端，借助 Py4j 实现 Python 和 Java 交互，进而实现通过 Python 编写 Spark 应用程序
* 在 Executor 端，则不需要借助 Py4j，因为 Executor 端运行的 Task 逻辑是由 Driver 发过来的，那是序列化后的字节码

![img](images/pyspark架构设计.png)

# Spark 运行流程

## 运行流程

1. 首先，Driver 为 Application 构建 DAG，并分解为 Stage
2. 然后，Driver 向 Cluster Manager 申请资源
3. Cluster Manager 向某些 Worker Node 发送征召信号
4. 被征召的 Worker Node 启动 Executor 进程响应征召，并向 Driver 申请任务
5. Driver 分配 Task 给 Worker Node
6. Executor 进程以 Stage 为单位执行 Task，期间 Driver 进行监控
7. Driver 收到 Executor 任务完成的信号后向 Cluster Manager 发送注销信号
8. Cluster Manager 向 Worker Node 发送释放资源信号
9. Worker Node 对应的 Executor 进程停止运行

> * Question 1: Task 如何生成？

![img](images/spark任务流程.png)

## 运行架构特点

* 每个 Application 都有自己专属的 Executor 进程，
  并且该进程在 Application 运行期间一直驻留，
  Executor 进程以多线程的方式运行 Task
* Spark 运行过程与资源管理器无关，只要能够获取 Executor 进程并保持通信即可
* Task 采用了数据本地性和推测执行等优化机制

# Spark 部署方式

## 部署方式

* Local
    - 本地运行模式，非分布式
* Standalone
    - 使用 Spark 自带集群管理器，部署后只能运行 Spark 任务
* Yarn
    - Hadoop 集群管理器，部署后可以同时运行 MapReduce、Spark、Storm、HBase 等各种任务
* Mesos
    - 与 Yarn 最大的不同是 Mesos 的资源分配是二次的，Mesos 负责分配一次，计算框架可以选择接受或者拒绝

## Hadoop 和 Spark 的统一部署

![img](images/hadoop与spark统一部署.png)

# RDD 数据结构

## RDD 简介

RDD 全称 Resilient Distributed Dataset，弹性分布式数据集，
是记录的只读分区集合，是 Spark 的基本数据结构
RDD 代表一个不可变、可分区、元素可并行计算的集合

## RDD 创建

一般有两种方式创建 RDD:

* 读取文件中的数据生成 RDD
* 通过将内存中的对象并行化得到 RDD

## RDD 操作

创建 RDD 后，可以使用各种操作对 RDD 进行编程。
RDD 操作有两种类型:

* Transformation 操作
    - 转换操作是从已经存在的 RDD 创建一个新的 RDD
* Action 操作
    - 行动操作是在 RDD 上进行计算后返回结果到 Driver

转换操作具有 Lazy 特性，即 Spark 不会立即进行实际的计算，
只会记录执行的轨迹，只有触发 Action 操作的时候，才会根据 DAG 进行执行

## RDD 特性

* Spark 用 Scala 实现了 RDD 的 API，可以通过调用 API 实现 RDD 的各种操作
* RDD 提供了一组丰富的操作以支持常见的数据运算，分为 Action(动作)和 Transformation(转换)两种类型
* 表面上 RDD 功能很受限、不够强大，实际上 RDD 已经被时间证明可以高效地表达许多框架的编程模型，
  比如 MapReduce、SQL、Pregel
* RDD 提供的转换接口都非常简单，都是类似 map、filter、groupby、join 等粒度的数据转换操作，
  而不是针对某个数据项的细粒度(不适合网页爬虫)

## RDD 依赖

RDD 操作确定了 RDD 之间的依赖关系。RDD 之间的依赖关系有两种，即窄依赖、宽依赖

### RDD 窄依赖

窄依赖时，父 RDD 的分区和子 RDD 的分区的关系是一对一或者多对一的关系

![img](images/RDD窄依赖.png)

### RDD 宽依赖

宽依赖时，父 RDD 的分区和子 RDD 的分区的关系是一对多或者多对多的关系

![img](images/RDD宽依赖.png)

### DAG 切分为 Stage

RDD 依赖关系确定了 DAG 切分成 Stage 的方式，切割规则为：从后往前，

* 遇到宽依赖就切割 Stage

RDD 之间的依赖关系形成了一个 DAG 有向无环图，DAG 会提交给 DAG Scheduler，
DAG Scheduler 会把 DAG 划分成相互依赖的多个 Stage，划分 Stage 的依据就是 RDD 之间的宽窄依赖。
遇到宽依赖就划分 Stage，每个 Stage 包含一个或多个 Task 任务，
然后将这些 Task 以 TaskSet 的形式提交给 Task Scheduler 运行

![img](images/stage切割原理.png)

# Apache Spark

## Spark 的设计哲学和历史

Apache Spark is **a unified computing engine** and **a set of libraries 
for parallel data processing(big data) on computer cluster**, and Spark 
**support multiple widely used programming language** (Python, Java,
Scala, and R), and Spark **runs anywhere** from a laptop to a cluster of
thousand of servers. This makes it an easy system to start with and
scale-up to big data processing or incredibly large scale.

- **A Unified Computing Engine**
    - [Unified]
        - Spark's key driving goal is to offer a unified platform for
        writing big data applications. Spark is designed to support a
        wide range of data analytics tasks, range from simple data
        loading and SQL queries to machine learning and streaming
        computation, over the same computing engine and with a
        consistent set of APIs.
    - [Computing Engine]
        - Spark handles loading data from storage system and performing
        computation on it, not permanent storage as the end itself, you
        can use Spark with a wide variety of persistent storage
        systems.
        - cloud storage system
            - Azure Stroage
            - Amazon S3
        - distributed file systems
            - Apache Hadoop
        - key-value stroes
            - Apache Cassandra
        - message buses
            - Apache Kafka
- **A set of libraries for parallel data processing on computer cluster**
    - Standard Libraries
        - SQL and sturctured data
        - SparkSQL
        - machine learning
        - MLlib
        - stream processing
        - Spark Streaming
        - Structured Streaming
        - graph analytics
        - GraphX
    - [External Libraries](https://spark-packages.org/) published as third-party packages by open source communities

## Spark 开发环境


- Language API
    - Python
    - Java
    - Scala
    - R
    - SQL
- Dev Env
    - local
         - [Java(JVM)](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
         - [Scala](https://www.scala-lang.org/download/)
         - [Python interpreter(version 2.7 or later)](https://repo.continuum.io/archive/)
         - [R](https://www.r-project.org/)
         - [Spark](https://spark.apache.org/downloads.html)
    - web-based version in [Databricks Community Edition](https://community.cloud.databricks.com/)

## Spark's Interactive Consoles

- Python

```bash
./bin/pyspark
```

- Scala

```bash
./bin/spark-shell
```

- SQL

```bash
./bin/spark-sql
```

## 云平台、数据

- [Project's Github](https://github.com/databricks/Spark-The-Definitive-Guide)
- [Databricks](https://community.cloud.databricks.com/)

# Spark

## Spark's Architecture

- **Cluster**
  - **Cluser**:
     - Single machine do not have enough power and resources to perform
        computations on huge amounts of information, or the user probably
        dose not have the time to wait for the computation to finish;
     - A cluster, or group, of computers, pools the resources of many
        machines together, giving us the ability to use all the cumulative
        resources as if they were a single computer.
     - A group of machines alone is not powerful, you need a framework to
        coordinate work across them. Spark dose just that, managing and
        coordinating the execution of task on data across a cluster of
        computers.
  - **Cluster manager**:
     - Spark's standalone cluster manager
     - YARN
     - Mesos
- **Spark Application**
    - **Cluster Manager**
         - A **Driver** process
            - the heart of a Spark Appliction and maintains all relevant
               information during the lifetime of the application;
            - runs ``main()`` functions;
            - sits on a node in the cluster;
            - responsible for:
                - maintaining information about the Spark Application
                - responding to user's program or input
                - analyzing, distributing and scheduling work across the **executors**
         - A Set of **Executor** process
            - responsible for actually carrying out the work that the **driver** assigns them
            - repsonsible for :
                - executing code assigned to it by the driver
                - reporting the state of the computation on that executor back to the dirver node
    - **Spark Application**
        - Spark employs a **cluster manager** that keeps track of the **resources** available;
        - The **dirver** process is responsible for executing the **dirver program's commands** 
            across the **executors** to complete a given task;
        - The **executors** will be running Spark code;

## Spark's Language API


- Scala
    - Spark's "default" language.
- Java
- Python
    - ``pyspark``
- SQL
    - Spark support a subset of the ANSI SQL 2003 standard.
- R
    - Spark core
        - ``SparkR``
    - R community-driven package
        - ``sparklyr``

## Spark's API

Spark has two fundamental sets of APIS:

- Low-level "unstructured" APIs
    - RDD
    - Streaming
- Higher-level structured APIs
    - Dataset
    - DataFrame
        - ``org.apache.spark.sql.functions``
        - Partitions
        - DataFrame(Dataset) Methods
        - DataFrameStatFunctions
        - DataFrameNaFunctions
        - Column Methods
        - alias
        - contains
    - Spark SQL
    - Structured Streaming

## Spark 使用

- 启动 Spark's local mode
    - 交互模式
        - ``./bin/spark-shell``
        - ``./bin/pyspark``
    - 提交预编译的 Spark Application
        - ``./bin/spark-submit``
- 创建 ``SparkSession``
    - 交互模式, 已创建
        - ``spark``
    - 独立的 APP
        - Scala:

        ```scala
        import org.apache.spark.SparkSession

        val spark = SparkSession
            .builder()
            .master()
            .appName()
            .config()
            .getOrCreate()
        ```

        - Python:

        ```python
        from pyspark import SparkSession

        spark = SparkSession \
            .builder() \
            .master() \
            .appName() \
            .config() \
            .getOrCreate()
        ```
### SparkSession

- SparkSession 简介
   - **Spark Application** controled by a **Driver** process called the **SparkSession**；
   - **SparkSession** instance is the way Spark executes user-defined manipulations across the cluster, 
      and there is a one-to-one correspondence between a **SparkSession** and a **Spark Application**;
- SparkSession 示例
   - Scala 交互模式：

    ```bash
    # in shell
    $ spark-shell
    ```
    
    ```scala
    // in Scala
    scala> val myRange = spark.range(1000).toDF("number")
    ```

   - Scala APP 模式：

    ```scala
    // in Scala
    import org.apache.spark.SparkSession
    val spark = SparkSession 
        .builder()
        .master()
        .appName()
        .config()
        .getOrCreate()
    ```

   - Python 交互模式：

    ```bash
    # in shell
    $ pyspark
    ```

    ```python
    # in Pyton
    >>> myRange = spark.range(1000).toDF("number")
    ```

   - Python APP 模式：

    ```python
    # in Python
    from pyspark import SparkSession
    spark = SparkSession \
        .builder() \
        .master() \
        .appName() \
        .config() \
        .getOrCreate()
    ```

### DataFrames

- A DataFrame is the most common Structured API;
- A DataFrame represents a table of data with rows and columns;
- The list of DataFrame defines the columns, the types within those columns is called the schema;
- Spark DataFrame can span thousands of computers:
- the data is too large to fit on one machine
- the data would simply take too long to perform that computation on one machine

### Partitions


### Transformation

#### Lazy Evaluation


### Action

转换操作能够建立逻辑转换计划, 为了触发计算, 需要运行一个动作操作(action)。一个动作指示 Spark 在一系列转换操作后计算一个结果。

### Spark UI

- **Spark job** represents **a set of transformations** triggered by **an individual action**, and can monitor the Spark job from the Spark UI;
- User can monitor the progress of a Spark job through the **Spark web UI**:
- Spark UI is available on port ``4040`` of the **dirver node**;
- Local Mode: ``http://localhost:4040``
- Spark UI displays information on the state of:
   - Spark jobs
   - Spark environment
   - cluster state
   - tunning
   - debugging

### 一个 🌰

(1)查看数据集

```bash
$ head /data/flight-data/csv/2015-summary.csv
```

(2)读取数据集

```scala
// in Scala
val flightData2015 = spark
    .read
    .option("inferSchema", "true")
    .option("header", "true")
    .csv("/data/flight-data/csv/2015-summary.csv")
```

```python
# in Python
flightData2015 = spark \
    .read \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .csv("/data/flight-data/csv/2015-summary.csv")
```

(3)在数据上执行转换操作并查看 Spark 执行计划

```scala
// in Scala
// 转换操作 .sort()
flightData2015.sort("count").explain()
flightData2015.sort("count")
```

(4)在数据上指定动作操作执行技术

```scala
// in Scala
// 配置 Spark shuffle
spark.conf.set("spark.sql.shuffle.partitions", "5")
// 动作操作 .take(n)
flightData2015.sort("count").take(2)
```

(5)DataFrame 和 SQL

```scala
// in Scala
flightData2015.createOrReplaceTempView("flight_data_2015")
```

```scala      
// in Scala
val sqlWay = spark.sql("""
    SELECT DEST_COUNTRY_NAME, count(1)
    FROM flight_data_2015
    GROUP BY DEST_COUNTRY_NAME
    """)

val dataFrameWay = flightData2015
    .groupBy("DEST_COUNTRY_NAME")
    .count()

sqlWay.explain()
dataFrameWay.explain()
```

```python
# in Python
sqlWay = spark.sql("""
    SELECT DEST_COUNTRY_NAME, count(1)
    FROM flight_data_2015
    GROUP BY DEST_COUNTRY_NAME
    """)

dataFrameWay = flightData2015 \
    .groupBy("DEST_COUNTRY_NAME") \
    .count()

sqlWay.explain()
dataFrameWay.explain()
```

```scala
// in Scala
spark.sql("""
    SELECT max(count) 
    FROM flight_data_2015
    """)
    .take(1)

import org.apache.spark.sql.functions.max
flightData2015
    .select(max("count"))
    .take(1)
```

```python
# in Python
spark.sql("""
    SELECT max(count)
    FROM flight_data_2015
    """) \
    .take(1)

from pyspark.sql.functions import max
flightData2015.select(max("count")).take(1)
```

# Spark 工具

## Spark 应用程序

Spark 可以通过内置的命令行工具 ``spark-submit`` 轻松地将测试级别的交互程序转化为生产级别的应用程序.

通过修改 ``spark-submit`` 的 ``master`` 参数, 可以将将应用程序代码发送到一个集群并在那里执行, 
应用程序将一直运行, 直到正确退出或遇到错误。应用程序需要在集群管理器的支持下进行, 常见的集群管理器有 
Standalone, Mesos 和 YARN 等.

- 示例 1

```bash
./bin/spark-submit \
--class org.apache.spark.examples.SparkPi \      # 运行的类 
--master local \                                 # 在本地机器上运行程序
./examples/jars/spark-examples_2.11-2.2.0.jar 10 # 运行的 JAR 包
```

- 示例 2

```bash
./bin/spark-submit \
-- master local \
./examples/src/main/python/pi.py 10
```

## Dataset: 类型安全的结果化 API

- 示例

```scala
case class Flight(DEST_COUNTRY_NAME: String, 
                 ORIGIN_COUNTRY_NAME: String,
                 count: BigInt)
val flightDF = spark
  .read
  .parquet("/data/flight-data/parquet/2010-summary.parquet/")

val flights = flightDF.as[Flight]

flights
  .fliter(flight_row => flight_row.ORIGIN_COUNTRY_NAME != "Canada")
  .map(flight_row => flight_row)
  .take(5)

flights
  .take(5)
  .filter(flight_row => flight_row.ORIGIN_COUNTRY_NAME != "Canada")
  .map(fr => Flight(fr.DEST_COUNTRY_NAME, fr.ORIGIN_COUNTRY_NAME, fr.count + 5))
```

## Spark Structured Streaming

Spark Structured Streaming(Spark 结构化流处理) 是用于数据流处理的高阶 API, 
在 Spark 2.2 版本之后可用。可以像使用 Spark 结构化 API 在批处理模式下一样, 
执行结构化流处理, 并以流式方式运行它们, 使用结构化流处理可以减少延迟并允许增量处理.
最重要的是, 它可以快速地从流式系统中提取有价值的信息, 而且几乎不需要更改代码。
可以按照传统批处理作业的模式进行设计, 然后将其转换为流式作业, 即增量处理数据, 
这样就使得流处理变得异常简单.

- 数据集：https://github.com/databricks/Spark-The-Definitive-Guide/tree/master/data/retail-data

### 创建一个静态数据集 DataFrame 以及 Schema

```scala
// in Scala
val staticDataFrame = spark
    .read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("/data/retail-data/by-day/*.csv")

staticDataFrame.createOrReplaceTempView("retail_data")
cal staticSchema = staticDataFrame.schema
```

```python
# in Python
staticDataFrame = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/data/retail-data/by-day/*.csv")

staticDataFrame.createOrReplaceTempView("retail_data")
staticSchema = staticDataFrame.schema
```

### 对数据进行分组和聚合操作

```scala
// in Scala
import org.apache.spark.sql.functions.{window, column, desc, col}
staticDataFrame
    .selectExpr(
    "CustomerId", 
    "(UnitPrice * Quantity) as total_cost", 
    "InvoiceDate"
    )
    .groupBy(
    col("CustomerId"), 
    window(col("InvoiceDate"), "1 day")
    )
    .sum("total_cost")
    .show(5)
```

```python
# in Python
from pyspark.sql.functions import window, column, desc, col
staticDataFrame \
    .selectExpr(
    "CustomerId", 
    "(UnitPrice * Quantity) as total_cost", 
    "InvoiceDate"
    ) \
    .groupBy(
    col("CustomerId"), 
    window(col("InvoiceDate"), "1 day")
    ) \
    .sum("total_cost") \
    .show(5)
```


### 设置本地模型运行参数配置

```scala
// in Scala
spark.conf.set("spark.sql.shuffle.partitions", "5")
```

```python
# in Python
spark.conf.set("spark.sql.shuffle.partitions", "5")
```

### 将批处理代码转换为流处理代码

(1)读取流式数据：

```scala
// in Scala
val streamingDataFrame = spark
    .readStream
    .schema(staticSchema)
    .option("maxFilesPerTrigger", 1)       // 指定一次应该读入的文件数量, 在实际场景中被省略
    .format("csv")
    .option("header", "true")
    .load("/data/retail-data/by-day/*.csv")
```

```python

# in Python
streamingDataFrame = spark \
    .readStream \
    .schema(staticSchema) \
    .option("maxFilesPerTrigger", 1) \
    .format("csv") \
    .option("header", "true") \
    .load("/data/retail-data/by-day/*.csv")
```

(2)查看 DataFrame 是否代表流数据：

```scala
// in Scala
streamingDataFrame.isStreaming // 返回 true
```

```python
# in Python
streamingDataFrame.isStreaming # 返回 true
```

(3)对流式数据执行分组聚合操作(转换操作)

```scala
# in Scala
val purchaseByCustomerPerHour = streamingDataFrame
    .selectExpr(
    "CustomerId", 
    "(UnitPrice * Quantity) as total_cost", 
    "InvoiceDate"
    )
    .groupBy(
    $"CustomerId", 
    window($"InvoiceDate", "1 day")
    )
    .sum("total_cost")
```

```python
# in Python
purchaseByCustomerPerHour = streamingDataFrame \
    .selectExpr(
    "CustomerId", 
    "(UnitPrice * Quantity) as total_cost", 
    "InvoiceDate"
    ) \
    .groupBy(
    col("CustomerId"), 
    window(col("InvoiceDate"), "1 day")
    ) \
    .sum("total_cost") \
    .show(5)
```

(4)调用对流数据的动作操作, 将数据缓存到内存中的一个表中, 在每次被触发后更新这个内存缓存

```scala
// in Scala
purchaseByCustomerPerHour.writeStream
    .format("memory")               // memory 代表将表存入内存
    .queryName("customer_purchases") // 存入内存的表的名称
    .outputMode("complete")         // complete 表示保存表中所有记录
    .start()
```

```python
# in Python
purchaseByCustomerPerHour.writeStream \
    .format("memory") \
    .queryName("customer_purchases") \
    .outputMode("complete") \
    .start()
```

(5)运行查询调试结果

```scala
// in Scala
spark.sql("""
    SELECT * 
    FROM customer_purchases
    ORDER BY `sum(total_cost)` DESC
    """)
    .show(5)
```

```python
# in Python
spark.sql("""
    SELECT * 
    FROM customer_purchases
    ORDER BY `sum(total_cost)` DESC
    """) \
    .show(5)
```

(6)将结果输出到控制台

```scala
// in Scala
purchaseByCustomerPerHour.writeStream
    .format("console")
    .queryName("customer_purchases_2")
    .outputMode("complete")
    .start()
```

```python
# in Python
purchaseByCustomerPerHour.writeStream \
    .format("console") \
    .queryName("customer_purchases_2") \
    .outputMode("complete") \
    .start()
```

## Spark 机器学习和高级数据分析



## Spark 低阶 API


Spark 中的所有对象都是建立在 RDD 之上的. Spark 的高阶 API 及所支持的高级操作都会被编译到较低级的 RDD 上执行, 
以方便和实现其较高效的分布式执行. 使用 RDD 可以并行化已经存储在驱动器机器内存中的原始数据.

大多数情况下用户只需要使用 Spark 的高阶 API 或高级操作就可以实现所需的业务逻辑, 有时候可能需要使用 RDD, 
特别是在读取或操作原始数据(未处理或非结构化的数据)时.

- 示例 1

```scala
// in Scala
spark.sparkContext.parallelize(Seq(1, 2, 3)).toDF() // 将 RDD 转化为 DataFrame
```

- 示例 2

```python
# in Python
from pyspark.sql import Row

spark.sparkContext.parallelize([Row(1), Row(2), Row(3)]).toDF()
```

## SparkR

SparkR 是一个在 Spark 上运行的 R 语言工具, 它具有与 Spark 其他支持语言相同的设计准则. 
SparkR 与 Spark 的 Python API 非常相似, 在大多数情况下, SparkR 支持 Python 支持的所有功能.

- 示例 1

```r
# in R
library(SparkR)
sparkDf <- read.df("/data/flight-data/csv/2015-summary.csv", source = "csv", header = "true", inferSchema = "true")
take(sparkDF, 5)
collect(orderBy(sparkDF, "count"), 20)
```

- 示例 2

```r
# in R
library(magrittr)

sparkDF %>% 
    orderBy(desc(sparkDF$count)) %>%
    groupBy("ORIGIN_COUNTRY_NAME") %>%
    count() %>%
    limit(10) %>%
    collect()
```

## Spark 生态系统和工具包

可以在 [Spark Packages 索引](https://spark-packages.org) 找到所有的开源社区维护的工具包, 
用户也可以将自己开发的工具包发布到此代码库中, 也可以在 GitHub 上找到各种其他项目和工具包.


# WordCount 示例

```python
import findspark
import pyspark
from pyspark import SparkContext, SparkConf


# 指定 spark_home 为 Spark 解压路径
spark_home = "/usr/local/spark"
# 指定 Python 路径
python_path = "/Users/zfwang/.pyenv/versions/3.7.10/envs/pyspark/bin/python"
findspark.init(spark_home, python_path)

conf = SparkConf().setAppName("WordCount").setMaster("local[4]")
sc = SparkContext(conf = conf)

rdd_line = sc.textFile("./data/hello.text")
rdd_word = rdd_line.flatMap(lambda x: x.split(" "))
rdd_one = rdd_word.map(lambda t: (t, 1))
rdd_count = rdd_one.reduceByKey(lambda x, y: x + y)
rdd_count.collect()
```

```
[('world', 1),
 ('love', 3),
 ('jupyter', 1),
 ('pandas', 1),
 ('hello', 2),
 ('spark', 4),
 ('sql', 1)]
```


# 参考

* [Spark的基本原理](https://mp.weixin.qq.com/s/dontNjAGFyskhHz7tbdWeg)

