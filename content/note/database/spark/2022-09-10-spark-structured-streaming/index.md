---
title: Spark Structured Streaming
author: 王哲峰
date: '2022-09-10'
slug: spark-structured-streaming
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

- [流计算工具](#流计算工具)
- [Structured Streaming 基本概念](#structured-streaming-基本概念)
  - [Spark Structured Streaming API](#spark-structured-streaming-api)
  - [流计算和批处理](#流计算和批处理)
  - [Spark Streaming 和 Spark Structured Streaming](#spark-streaming-和-spark-structured-streaming)
  - [source 和 sink](#source-和-sink)
    - [source](#source)
    - [sink](#sink)
  - [operation 和 query](#operation-和-query)
- [Structured Streaming 示例](#structured-streaming-示例)
- [Operator 转换](#operator-转换)
- [Structured Streaming 结果输出](#structured-streaming-结果输出)
</p></details><p></p>


# 流计算工具

市面上主流的开源流计算工具主要有 Storm、Flink、Spark

* Storm 
    - 延迟最低，一般为几毫秒到几十毫秒
    - 数据吞吐量较低，每秒能够处理的事件在几十万左右
    - 建设成本高
* Flink
    - 是目前国内互联网厂商主要使用的流计算工具
    - 延迟一般在几十到几百毫秒
    - 数据吞吐量非常高，每秒才能处理的事件可以达到几百上千万
* Spark
    - 通过 Spark Streaming 或 Spark Structured Streaming 支持流计算，
      但 Spark 的流计算是将流数据按照时间分割成一个一个的小批次(mini-batch)进行处理的，
    - 延迟一般在 1 秒左右
    - 吞吐量和 Flink 相当
    - 值的注意的是 Spark Structured Streaming 现在也支持了 Continous Streaming 模式，
      即在数据到达时就进行计算，不过目前还处于测试阶段，不是特别成熟

虽然从目前来看，在流计算方面，Flink 比 Spark 更具性能优势，是当之无愧的王者。
但是由于 Spark 拥有比 Flink 更加活跃的社区，其流计算功能也在不断完善和发展，
未来在流计算领域或许足以挑战 Flink 的地位


# Structured Streaming 基本概念

## Spark Structured Streaming API

```python
import os
import time
import random

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("structured streaming") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "8") \
    .config("master", "local[4]") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext
```

## 流计算和批处理

* 批处理是处理离线数据，单个处理数据量大，处理速度比较慢
* 流计算是处理在线实时产生的数据，单次处理的数据量小，但处理速度更快

## Spark Streaming 和 Spark Structured Streaming

Spark 在 2.0 之前，主要使用 Spark Streaming 来支持流计算，
其数据结构模型为 DStream，其实就是一个个小批次数据构成的 RDD 队列

目前，Spark 主要推荐的流计算模块是 Structured Streaming，
其数据结构模型是 Unbounded DataFrame，即没有边界的数据表

相当于 Spark Streaming 建立在 RDD 数据结构上面，Structured Streaming 是建立在 SparkSQL 基础上，
DataFrame 的绝大部分 API 也能够用在流计算上，实现了流计算和批处理的一体化，并且由于 SparkSQL 的优化，
具有更好的性能，容错性也更好

## source 和 sink

### source

source 即流数据从何而来

在 Spark Structured Streaming 中，主要可以从以下方式接入流数据:

* Kafka Source
    - 当消息生产者发送的消息到达某个 topic 的消息队列时，将触发计算
    - 是 Structured Streaming 最常用的流数据来源
* File Source
    - 当路径下游文件被更新时，将触发计算
    - 这种方式通常要求文件到达路径是原子性的(瞬间到达、不是慢慢写入)，以确保读取到数据的完整性
    - 在大部分文件系统中，可以通过 `move` 操作实现这个特性
* Socket Source
    - 需要指定 host 地址和 port 端口号
    - 这种方式一般只用来测试代码
    - Linux 环境下可以用 `nc` 命令来开启网络通信端口发送消息测试
 

### sink

sink 即流数据被处理后输出到哪里

在 Spark Structured Streaming 中，主要可以用以下方式输出流数据计算结果

* Kafka Sink
    - 将处理后的流数据输出到 Kafka 某个或某些 topic 中
* File Sink
    - 将处理后的流数据写入到文件系统中
* ForeachBatch Sink
    - 对于某一个 micro-batch 的流数据处理后的结果，用户可以编写函数实现自定义处理逻辑
    - 例如写入到多个文件中，或者写入到文件并打印
* Foreach Sink
    - 一般在 Continuous 触发模式下使用，用户编写函数实现每一行的处理
* Console Sink
    - 打印到 Driver 端控制台，如果日志量大，谨慎使用
    - 一般供调试使用
* Memory Sink
    - 输出到内存中，供调试使用

流数据输出到 sink 中有三种方式，叫做 output mode:

* append mode
    - 默认方式，将新溜过来的数据的计算结果添加到 sink 中
* complete mode
    - 一般使用于有 aggregation 查询的情况
    - 流计算启动开始到目前为止接收到的全部数据的计算结果添加到 sink 中
* update mode
    - 只有本次结果中和之前结果不一样的记录才会添加到 sink 中


## operation 和 query

在 SparkSQL 批处理中，算子被分为 Transformation 算子和 Action 算子。
Spark Structured Streaming 有所不同，所有针对流数据的算子都是懒惰执行的，叫做 operation





# Structured Streaming 示例





# Operator 转换



# Structured Streaming 结果输出

