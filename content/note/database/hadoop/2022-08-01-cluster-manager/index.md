---
title: 集群管理器
author: 王哲峰
date: '2022-08-01'
slug: database-cluster-manager
categories:
  - database
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

- [Spark Standalone](#spark-standalone)
- [Apache Mesos](#apache-mesos)
- [Hadoop YARN](#hadoop-yarn)
- [参考](#参考)
</p></details><p></p>


* Spark独立集群管理器（Standalone），一种简单的 Spark 集群管理器，很容易建立集群，
  基于 Spark 自己的 Master-Worker 集群
* Apache Mesos，一种能够运行 Haoop MapReduce 和服务应用的集群管理器
* Hadoop YARN，Spark 可以和 Hadoop 集成，利用 Yarn 进行资源调度

# Spark Standalone


# Apache Mesos



# Hadoop YARN

1.查看帮助

```bash
$ yarn application --help
```

2.查看提交到集群上的所有任务

```bash
$ yarn application -list
```

3.杀死某个任务

```bash
$ yarn application -kill <applicationId>
```

# 参考

* [spark几种集群管理器总结](https://blog.csdn.net/yawei_liu1688/article/details/112305234)

