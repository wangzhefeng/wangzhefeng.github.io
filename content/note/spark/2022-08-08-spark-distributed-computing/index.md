---
title: Spark 分布并行计算
author: 王哲峰
date: '2022-08-08'
slug: spark-distributed-computing
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

- [分布式并行计算](#分布式并行计算)
- [Spark 简化了分布式计算的开发](#spark-简化了分布式计算的开发)
- [Spark 是基于数据集的计算框架](#spark-是基于数据集的计算框架)
- [数据集概念](#数据集概念)
  - [SQL 中的数据集](#sql-中的数据集)
  - [JavaScript 中的数据集](#javascript-中的数据集)
- [Spark RDD 是一切魔力产生的根源](#spark-rdd-是一切魔力产生的根源)
- [Spark 操作符](#spark-操作符)
- [参考](#参考)
</p></details><p></p>

# 分布式并行计算

一个资源密集型的任务，需要一组资源并行地完成。
当计算任务过重时，就把计算任务拆分，
然后放到多个计算节点上同时执行，这就是分布式并行计算

* 分布式并行计算，强调硬件的堆叠，解决问题
* 分布式并行计算，强调用机器的蛮力，让笨重算法也能跑高分

# Spark 简化了分布式计算的开发

Spark 屏蔽了分布式并行计算的细节，可以快速开发分布式并行应用。
只要把数据和计算程序交给 Spark，它会智能地进行数据切分、算法复制、分布执行、结果合并。

# Spark 是基于数据集的计算框架

Spark 的计算范式：数据集上的计算

* Spark 用起来的确简单，但有一点特别要注意，得按照 Spark 的范式写算法
* Spark 是在数据集的层次上进行分布并行计算的，它只认成堆的数据
* 提交给 Spark 的计算任务，必须满足两个条件:
    - 1.数据是可以分块的，每块构成一个集合
    - 2.算法只能在集合级别执行操作

Spark 是一种粗粒度、基于数据集的并行计算框架

# 数据集概念

## SQL 中的数据集

SQL 里的操作都是集合级别的

## JavaScript 中的数据集

JavaScript 中也有集合操作

# Spark RDD 是一切魔力产生的根源

Spark的RDD自动进行数据的切分和结果的整合

# Spark 操作符

Spark 提供了 80 多种操作符对集合进行操作，并且还可以自行扩展

* 变换：变换操作总是获得一个新的 RDD

    - `map(func)`: 将原始数据集的每一个记录使用传入的函数 `func` ，映射为一个新的记录，并返回新的 RDD
    - `filter(func)`: 返回一个新的 RDD，仅包含那些符合条件的记录，即 `func` 返回 `true`
    - `flatMap(func)`: 和 `map` 类似，只是原始记录的一条可能被映射为新的 RDD 中的多条
    - `union(otherDataset)`: 合并两个 RDD，返回一个新的 RDD
    - `intersection(otherDataset)`返回一个新的 RDD，仅包含两个 RDD 共有的记录

* 动作：动作操作总是获得一个本地数据，意味着控制权回到程序

    - `reduce(func)`: 使用 `func` 对 RDD 的记录进行聚合
    - `collect()`: 返回 RDD 中的所有记录
    - `count()`: 返回 RDD 中的记录总数


# 参考

- http://www.hubwiz.com/