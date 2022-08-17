---
title: Spark 基本原理
author: 王哲峰
date: '2022-08-17'
slug: spark-basic
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
- [Spark 架构设计](#spark-架构设计)
- [Spark 运行流程](#spark-运行流程)
- [Spark 部署方式](#spark-部署方式)
- [RDD 数据结构](#rdd-数据结构)
</p></details><p></p>

# Spark 优势特点

作为大数据计算框架 MapReduce 的继任者，Spark 具有以下优势特性

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
| 编程范式           | Map + Reduce                             | Transformation + Action                       |
| 计算中间结果处理方式 | 计算中间结果写入磁盘，IO及序列化、反序列化代价大 | 中间计算结果在内存中维护，存取速度比磁盘高几个数量级   |
| Task 维护方式      | Task 以进程的方式维护                       | Task 以线程的方式维护                            |


## 通用性

Spark 提供了统一的解决方案。Spark 可以用于批处理、交互式查询(Spark SQL)、
实时流式计算(Spark Streaming)、机器学习(Spark MLlib)和图计算(GraphX)。
这些不同类型的处理都可以在同一个应用中无缝使用，这对于企业应用来说，
就可以使用一个平台来进行不同的工程实现，减少了人力开发和平台部署成本

## 兼容性

Spark 能够跟很多开源工程兼容使用，如 Spark 可以使用 Hadoop 的 YARN 和 Apache Mesos 作为它的资源管理和调度器，
并且 Spark 可以读取多种数据源，如 HDFS、HBase、MySQL 等

# Spark 基本概念

* RDD
* DAG
* Driver Program
* Cluster Manager
* Worker Node
* Executor
* Application
* Job
* Stage
* Task

总结：

Application 由多个 Job 组成，Job 由多个 Stage 组成，Stage 由多个 Task 组成。Stage 是 Task 调度的基本单位

```
Application
    - Job 1
        - Stage 1
            - Task 1
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


# Spark 运行流程



# Spark 部署方式



# RDD 数据结构


