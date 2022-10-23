---
title: Flink 环境安装配置
author: 王哲峰
date: '2022-10-22'
slug: flink-env
categories:
  - flink
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

- [TODO](#todo)
  - [Flink 程序如何开发](#flink-程序如何开发)
  - [Flink 程序开发的语言](#flink-程序开发的语言)
  - [Flink 程序开发的 API](#flink-程序开发的-api)
  - [Flink 程序开发的应用](#flink-程序开发的应用)
- [本地模式](#本地模式)
  - [步骤 1: 下载](#步骤-1-下载)
    - [Java 11](#java-11)
    - [Flink](#flink)
  - [步骤 2: 启动集群](#步骤-2-启动集群)
  - [步骤 3: 提交作业(Job)](#步骤-3-提交作业job)
  - [步骤 4: 停止集群](#步骤-4-停止集群)
- [参考](#参考)
</p></details><p></p>

# TODO

## Flink 程序如何开发

## Flink 程序开发的语言

## Flink 程序开发的 API

## Flink 程序开发的应用


# 本地模式

## 步骤 1: 下载

### Java 11

```bash
$ java -version
```

### Flink

下载 [release 版本的压缩包](https://flink.apache.org/zh/downloads.html) 并解压 

```bash
$ tar -xzf flink-1.15.2-bin-scala_2.12.tgz
$ cd flink-1.15.2-bin-scala_2.12
```

## 步骤 2: 启动集群

Flink 附带了一个 bash 脚本，可以用于启动本地集群

```bash
$ ./bin/start-cluster.sh
Starting cluster.
Starting standalonesession daemon on host.
Starting taskexecutor daemon on host.
```

## 步骤 3: 提交作业(Job)

Flink 的 Releases 附带了许多的示例作业。你可以任意选择一个，快速部署到已运行的集群上。

```bash
$ ./bin/flink run examples/streaming/WordCount.jar
$ tail log/flink-*taskexecutor-*.out
  (nymph,1)
  (in,3)
  (thy,1)
  (orisons,1)
  (be,4)
  (all,2)
  (my,1)
  (sins,1)
  (remember,1)
  (d,4)
```

另外，你可以通过 Flink 的 Web UI(http://localhost:8081/) 来监视集群的状态和正在运行的作业

## 步骤 4: 停止集群

可以快速停止集群和所有正在运行的组件

```bash
$ ./bin/stop-cluster.sh
```

# 参考

* https://nightlies.apache.org/flink/flink-docs-release-1.15/zh//docs/try-flink/local_installation/