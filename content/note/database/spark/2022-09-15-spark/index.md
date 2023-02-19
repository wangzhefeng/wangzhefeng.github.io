---
title: Spark
author: 王哲峰
date: '2022-09-15'
slug: spark
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

- [spark 核心概念](#spark-核心概念)
- [驱动器程序(driver program)](#驱动器程序driver-program)
- [SparkContext对象](#sparkcontext对象)
- [RDD](#rdd)
- [创建RDD](#创建rdd)
- [调用RDD操作进行求值](#调用rdd操作进行求值)
- [RDD缓存和持久化](#rdd缓存和持久化)
- [执行器节点(executor)](#执行器节点executor)
- [传递函数的API](#传递函数的api)
- [键值对RDD(Pair RDD)](#键值对rddpair-rdd)
  - [键值对 RDD 转化操作](#键值对-rdd-转化操作)
  - [键值对 RDD 的行动操作](#键值对-rdd-的行动操作)
- [数据读取与保存](#数据读取与保存)
  - [三种常见的数据源](#三种常见的数据源)
    - [文件格式](#文件格式)
    - [文件系统](#文件系统)
  - [Spark SQL中的结构化数据源](#spark-sql中的结构化数据源)
  - [JSON](#json)
  - [创建 HiveContext 对象](#创建-hivecontext-对象)
  - [HiveContext.jsonFile("")](#hivecontextjsonfile)
  - [Hive 查询语言(HSQL)](#hive-查询语言hsql)
  - [由行组成的 RDD](#由行组成的-rdd)
  - [Apache Hive](#apache-hive)
  - [创建 HiveContext 对象(hive-site.xml)](#创建-hivecontext-对象hive-sitexml)
  - [Hive 查询语言(HSQL)](#hive-查询语言hsql-1)
  - [由行组成的 RDD](#由行组成的-rdd-1)
  - [数据库与键值存储](#数据库与键值存储)
  - [Cassandra](#cassandra)
  - [HBase](#hbase)
  - [Elasticsearch](#elasticsearch)
  - [JDBC源](#jdbc源)
</p></details><p></p>

# spark 核心概念

- 在较高的层次上, 每个Spark应用程序都包含一个 `驱动程序`, 该程序运行用户的main功能并在集群上执行各种并行操作。
    - Spark提供的主要抽象是 `弹性分布式数据集（RDD）`, 它是跨群集节点分区的元素的集合, 可以并行操作
        - RDD是通过从Hadoop文件系统（或任何其他Hadoop支持的文件系统）中的文件或驱动程序中的现有Scala集合开始, 并对其进行转换而创建的
        - Spark 在内存中保留 RDD, 允许它在并行操作中有效地重用
        - RDD 会自动从节点故障中恢复
    - Spark中的第二个抽象是可以在并行操作中使用的 `共享变量`。默认情况下, 
      当Spark并行运行一个函数作为不同节点上的一组任务时, 它会将函数中使用的每个变量的副本发送给每个任务。
      有时, 变量需要跨任务共享, 或者在任务和驱动程序之间共享
    - Spark支持两种类型的共享变量：
        - 广播变量, 可用于缓存所有节点的内存中的值;
        - 累加器, 它们是仅“添加”到的变量, 例如计数器和总和；
- Spark的主要抽象是一个名为Dataset的分布式项目集合。可以从 Hadoop InputFormats（例如HDFS文件）或通过转换其他数据集来创建数据集


# 驱动器程序(driver program)

- 每个spark应用都由一个驱动器程序来发起集群上的并行操作
- 包含应用的main函数
- 定义了集群上的分布式数据集
- 对分布式数据集应用相关操作
- 即：spark shell

# SparkContext对象

- 访问 spark, 代表对计算集群的一个连接
- shell 启动时已经自动创建了一个 SparkContext 对象,即变量 sc
- 利用 SparkContext 对象创建 RDD


# RDD

- RDD是一个不可变的分布式对象集合
- 每个RDD被分为多个分区,这些分区运行在集群中的不同节点上
- RDD可以包含Python,Java,Scala中的任意类型的对象,甚至可以包含用户自定义的对象

# 创建RDD

1. 读取一个外部数据集
2. 在驱动器程序里分发驱动器程序中的对象集合(list,set)

# 调用RDD操作进行求值

- 转化操作(transformation)RDD, 转化操作会由一个RDD生成一个新的RDD
   - sc.map()
   - sc.filter()
   - sc.flatMap()
   - sc.union()
   - sc.intersection()
   - sc.distinct()
   - sc.substract()
   - sc.cartesian()
   - sc.sample(withReplacement = true, fraction = 0.5, [seed])
- 行动操作(action)RDD, 行动操作会对 RDD 计算出一个结果,
  并把结果返回到驱动器程序中,或把结果存储到外部存储系统(HDFS)中, Spark 只会惰性计算 RDD,
  他们只有在第一次在一个行动操作中用到时才会真正计算
   - sc.count()
   - sc.countByValue()
   - sc.first()
   - sc.reduce()
   - sc.combine()
   - sc.fold(init)(func)
   - sc.aggregate(zeroValue)(seqOp, comOp)
   - sc.collect()
   - sc.take(n)
   - sc.takeOrdered(n)(ordering)
   - sc.top()
   - sc.takeSample(withReplacement, num, seed)
   - sc.foreach()

# RDD缓存和持久化

- 默认情况下,Spark 的 RDD 会在每次对他们进行行动操作时重新计算.
  如果想在多个行动中重用一个 RDD, 可以使用 RDD.persist() 让 Spark 把这个 RDD 缓存下来. 
  可以让 Spark 把数据持久化到许多不同的地方.
- 在第一次持久化的 RDD 计算之后, Spark 会把 RDD 的内容保存到内存中(以分区方式存储到集群中的个机器上), 
  这样在之后的行动操作中, 就可以重新用这些数据了.

# 执行器节点(executor)

- 要对RDD执行操作,驱动器程序一般需要管理多个执行器节点
- 每个执行器节点上都有多个需要执行的任务

# 传递函数的API

- spark自动将对应的操作(函数)分发到各个执行器节点,
- 即只需在单个的驱动器程序中编程,并且让代码自动运行在多个节点(集群)上

# 键值对RDD(Pair RDD)

- 键值对 RDD 通常用来进行聚合计算
- 键值对 RDD 提供了一些新的操作接口
- 创建键值对 RDD
- 存储键值对的数据格式会在读取时直接返回由其键值对数据组成的 pair RDD；
- 把一个普通的 RDD 转换为 pair RDD, 调用 map() 函数, 传递的函数需要返回键值对。
- 用 Scala 和 Python 从一个内存中的数据集创建 pari RDD 时, 
  只需要对这个二元组组成的集合调用 SparkContext.parallelize() 方法


## 键值对 RDD 转化操作

- sc.map()
- sc.filter()
- sc.flatMap()
- sc.union()
- sc.intersection()
- sc.distinct()
- sc.substract()
- sc.cartesian()
- sc.sample(withReplacement = true, fraction = 0.5, [seed])
- sc.reduceByKey(func)
- sc.foldByKey(init)(func)
- sc.groupByKey()
- sc.combineByKey(createCombiner, mergeCombiners, partitioner)
- sc.sortByKey(ascending = True, )
- sc.mapValues(func)
- sc.flatMapValues(func)
- sc.keys()
- sc.values()
- sc.subtractByKey(other_sc)
- sc.join(other_sc)
- sc.leftOuterJoin(other_sc)
- sc.rightOuterJoin(other_sc)
- sc.cogroup(other_sc)

## 键值对 RDD 的行动操作

- sc.count()
- sc.countByValue()
- sc.first()
- sc.reduce()
- sc.fold(init)(func)
- sc.aggregate(zeroValue)(seqOp, comOp)
- sc.collect()
- sc.take(n)
- sc.takeOrdered(n)(ordering)
- sc.top()
- sc.takeSample(withReplacement, num, seed)
- sc.foreach()
- sc.collectAsMap()
- sc.lookup()

# 数据读取与保存

## 三种常见的数据源

### 文件格式

- 文本文件(非结构化)
   - sc.textFile("")
   - sc.wholeTextFiles("")
   - sc.saveAsTextFile("")
- JSON(半结构化)
- SequenceFile(结构化)
- csv(结构化)
- protocol buffer(结构化)
- 对象文件(结构化)

### 文件系统

- 本地文件系统
   - sc.textFile("file://...")
- 分布式文件系统DFS(NFS,HDFS,Amazon S3等)
   - NFS
   - Amazon S3
   - S3访问凭据AWS\ *ACCESS*\ KEY\ *ID,AWS*\ SECRET\ *ACCESS*\ KEY
   - s3n://bucket/path-within-bucket
   - HDFS
   - hdfs://master:port/path(Hadoop版本)

## Spark SQL中的结构化数据源


## JSON



## 创建 HiveContext 对象


## HiveContext.jsonFile("")



## Hive 查询语言(HSQL)



## 由行组成的 RDD



## Apache Hive



## 创建 HiveContext 对象(hive-site.xml)



## Hive 查询语言(HSQL)



## 由行组成的 RDD



## 数据库与键值存储



## Cassandra



## HBase



## Elasticsearch



## JDBC源

