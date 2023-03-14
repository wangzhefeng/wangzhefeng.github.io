---
title: Hadoop
author: 王哲峰
date: '2022-05-06'
slug: database-hadoop
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

- [Hadoop 介绍](#hadoop-介绍)
- [Hadoop 安装](#hadoop-安装)
  - [Mac](#mac)
  - [Linux](#linux)
  - [Windows](#windows)
- [Hadoop 构造模块](#hadoop-构造模块)
  - [NameNode](#namenode)
  - [DataNode](#datanode)
  - [Secondary NameNode](#secondary-namenode)
  - [JobTracker](#jobtracker)
  - [TaskTracker](#tasktracker)
- [Hadoop 基本文件命令](#hadoop-基本文件命令)
  - [指定文件和目录确切位置的URI](#指定文件和目录确切位置的uri)
  - [基本形式](#基本形式)
  - [基本文件命令](#基本文件命令)
- [参考](#参考)
</p></details><p></p>


# Hadoop 介绍

Google 率先提出了 MapReduce 用来应对数据处理需求的系统, 
Doug Cutting 领导开发了一个开源版本的 MapReduce, 称为 Hadoop

Hadoop 是一个开源框架,可编写和运行分布式应用处理大规模数据;

* 方便: Hadoop 运行在由一般商用机器构成的大型集群上, 或者如 Amazon 弹性计算云(EC2)等云服务器之上;
* 健壮: Hadoop 致力于在一般商用硬件上运行, 其架构假设硬件会频繁地出现失效. 它可以从容地处理大多数此类故障
* 可扩展: Hadoop 通过增加集群节点, 可以线性地扩展以处理更大的数据集
* 简单: Hadoop 允许用户快速编写出高效的并行代码

Hadoop 集群是在同一地点用网络互联的一组通用机器. 数据存储和处理都发生在这个机器"云"中。
不同的用户可以从独立的客户端提交计算作业到 Hadoop, 这些客户端可以是远离 Hadoop 集群的个人台式机

通常在一个 Hadoop 集群中的机器都是相对同构的 x86 Linux 服务器.
而且它们几乎位于同一个数据中心,并且通常在同一组机架里;

Hadoop 强调把代码向数据迁移,而不是相反的过程; Hadoop 的集群环境中既包含数据又包含计算环境, 
客户端仅需发送待执行的 MapReduce 程序, 而这些程序一般都很小(通常为几千字节)

# Hadoop 安装

## Mac


## Linux


## Windows


# Hadoop 构造模块

在一个全配置的集群上, "运行 Hadoop" 意味着在网络分布的不同服务器上运行一组守护进程(daemons). 
这些守护进程有着特殊的角色, 一些仅存在单个服务器上, 一些则运行在多个服务器上: 

* NameNode(名字节点)
* DataNode(数据节点)
* Secondary NameNode(次名字节点)
* JobTracker(作业跟踪节点)
* TaskTracker(任务跟踪节点)

## NameNode

Hadoop 在分布式计算机与分布式存储中都采用了 **主/从(master/slave)** 结构。
分布式存储系统被称为 Hadoop 文件系统, 或简称 HDFS 

* **NameNode** 位于 HDFS 的主端, 它指导 **从端的 DataNode** 执行底层的 I/O 任务; 
  NameNode 是 HDFS 的书记员, 他跟踪文件如何被分割成文件块, 而这些块又被哪些节点存储, 
  以及分布式文件系统的整体运行状态是否正常; 
* 运行 NameNode 会消耗大量的内存和 I/O 资源, 因此, 为了减轻机器的负载, 
  驻留在 NameNode 的服务器通常不会存储用户数据或者执行 MapReduce 程序的计算任务, 
  这意味着 NameNode 服务器不会同时是 DataNode 或者 TaskTracker; 

## DataNode

## Secondary NameNode

## JobTracker

JobTracker 守护进程是应用程序和 Hadoop 之间的纽带。
每个 Hadoop 集群只有一个 JobTracker 守护进程, 
通常位于服务器集群的主节点上；一旦提交代码到集群上, JobTracker 就会确定执行计划, 
包括决定处理哪些文件、为不同的任务分配节点以及监控所有任务的运行; 
如果任务失败, JobTracker 将自动重启任务, 但分配的节点可能会不同, 
同时受到预定义的重试次数限制

## TaskTracker

# Hadoop 基本文件命令

1. 添加目录和文件
2. 获取文件
3. 删除文件

## 指定文件和目录确切位置的URI

* Hadoop的文件命令既可以与HDFS文件系统交互, 也可以和本地文件系统交互; 
* URI精确地定位一个特定文件或目录的位置, 完整的URI格式为 `scheme//authority/path`
* `scheme` 类似于一个协议, 它可以是 `hdfs` 或 `file` , 分别指定HDFS文件系统或本地文件系统; 
* `authority` 是HDFS中NameNode的主机名; 
* `path` 是文件或目录的路径; 
* 例如, 对于在本地机器的9000端口上, 以标准伪分布式模型运行的HDFS, 
  访问用户目录/user/chunk中文件example.txt的URI:  `hdfs://localhost:9000/user/chuck/example.txt`
* 大多数设置不需要制定URI中的 `scheme://authority` 部分; 
* 当在本地文件系统和HDFS之间复制文件时, Hadoop中的命令会分别吧本地文件熊作为源和目的, 而不需要制定scheme为file; 
* 对于其他命令, 如果未设置URI中的 `scheme://authority` , 就会采用Hadoop的默认设置; 
    - 假如 `conf/core-site.xml` 文件已经更改为伪分布式配置, 
      则文件中的 `fs.default.name` 属性应为 `hdfs://localhost:9000`. 
      在此配置下, URI `hdfs://localhost:9000/user/chuck/example.txt` 可以缩短为 `/user/chuck/example.txt`
   - HDFS默认当前工作目录为 `/user/$USER` , 其中 `$USER` 是登录用户名, 如果作为 `chuck` 登录, 
     则 URI `hdfs://localhost:9000/user/chuck/example.txt` 就缩短为 `example` ; 

## 基本形式

```bash
# method 1
$ hadoop fs -cmd <args>

# method 2
$ hadoop dfs -cmd <args>

# method 3
$ alias hdfs="hadoop dfs"
$ hdfs -cmd <args>
```

其中

* `cmd` : 特定的文件命令
* `<args>` : 一个数目可变的参数 

## 基本文件命令

1. 查看帮助

```bash
$ hadoop dfs -help <cmd>
```

2. 查看目录

查看某个目录: 

```bash
$ hadoop dfs -ls <path>
```

查看当前目录: 

```bash
hadoop dfs -ls /
```

查看当前目录下的文件和子目录, 类似于 `ls -r` : 

```bash
$ hadoop dfs -lsr /
```

3. 创建、删除文件夹

- 删除一般是删除到 `.Trash` 中, 一般有一定的时效清空的, 如果误删可以找回; 

```bash
# 创建
$ hadoop dfs -mkdir </hadoop dir path/dir name>
$ hadoop dfs -mkdir -p </hadoop dir path/dir name>

# 删除
$ hadoop dfs -rmr <hadoop dir path>
```

4. 创建、删除空文件

- 删除一般是删除到 `.Trash` 中, 一般有一定的时效清空的, 如果误删可以找回; 

```bash
# 创建
$ hadoop dfs -touchz </hadoop file path/file name>

# 删除
$ hadoop dfs -rm <hadoop file path>
```

5. 检索文件

* 复制; 
* 移动或重命名; 
* 从HDFS中下载文件到本地文件系统; 
* 从本地文件系统复制文件到HDFS中; 

```bash
# 复制
$ hadoop dfs -cp <hadoop source file path > <hadoop target file path>
$ hadoop dfs -cp -r <hadoop source path> <hadoop target path>

# 移动、重命名
$ hadoop dfs -mv <hadoop source path> <hadoop target path>

# HDFS => local(从HDFS上把文件或文件夹下载到本地)
$ hadoop dfs -get <hadoop source path> <local target path>
$ hadoop dfs -copyToLocal <hadoop source path> <local target path>

# 将HDFS上一个目录下的所有文件合并成一个文件下载到本地
$ hadoop dfs -getmerge <hadoop dir path> <local file path>

# local => HDFS(上传本地文件或文件夹到HDFS)
$ hadoop dfs -put <local source path> <hadoop target path>
$ hadoop dfs -copyFromLocal <local source path> <hadoop target path>
$ hadoop dfs -moveToLocal <local source path> <hadoop target path>
```

6. 查看文件内容

```bash
$ hadoop dfs -cat <hadoop file path>
$ hadoop dfs -text <hadoop file path>

$ hadoop dfs -tail <hadoop file path>
$ hadoop dfs -tail -f <hadoop file path>
```

7. 查看文件、文件夹大小

```bash
# 字节为单位展示
$ hadoop dfs -du <hadoop file path>

# GB为单位展示
$ hadoop dfs -du -s -h <hadoop file path>

# 查看文件夹下每个文件大小
$ hadoop dfs -du -s -h <hadoop dir path/*>
```

8. 判断文件、目录、大小

```bash
# 检查文件是否存在, 存在返回0
$ hadoop dfs -test -e filename

# 检查文件是否是0字节, 是返回0
$ hadoop dfs -test -z filename

# 检查文件是否是目录, 是返回1, 否则返回0
$ hadoop dfs -test -d filename
```

# 参考

* [什么是HDFS](https://mp.weixin.qq.com/s?__biz=MzI4Njg5MDA5NA==&mid=2247486743&idx=1&sn=658d90686b4b7e80d3042f4208bf07eb&chksm=ebd74c16dca0c5009f6e12750306ea55803b6e9d02d21017a429ac4e65bcbd12dbf8c925f1ec&token=1109491988&lang=zh_CN#rd)

