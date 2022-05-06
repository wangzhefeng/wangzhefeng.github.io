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

- [环境配置](#环境配置)
- [Hadoop](#hadoop)
  - [Hadoop架构](#hadoop架构)
    - [Hadoop是什么?](#hadoop是什么)
    - [分布式存储、分布式计算](#分布式存储分布式计算)
    - [Hadoop构造模块](#hadoop构造模块)
      - [NameNode](#namenode)
      - [JobTracker](#jobtracker)
  - [Hadoop集群安装](#hadoop集群安装)
  - [Hadoop运行](#hadoop运行)
- [HDFS文件操作](#hdfs文件操作)
  - [Hadoop基本文件命令](#hadoop基本文件命令)
    - [指定文件和目录确切位置的URI](#指定文件和目录确切位置的uri)
    - [基本形式](#基本形式)
    - [基本文件命令](#基本文件命令)
  - [yarn命令行](#yarn命令行)
- [MapReduce程序](#mapreduce程序)
  - [MapReduce 程序通过操作键值对来处理数据](#mapreduce-程序通过操作键值对来处理数据)
  - [Hadoop 数据类型](#hadoop-数据类型)
  - [Mapper](#mapper)
  - [Reducer](#reducer)
  - [Partitioner: 重定向Mapper的输出](#partitioner-重定向mapper的输出)
  - [Combiner: 本地reduce](#combiner-本地reduce)
- [读/写](#读写)
</p></details><p></p>



# 环境配置

- Java:Java 1.6及以上版本;
- OS: Linux;
- Hadoop

# Hadoop

## Hadoop架构

### Hadoop是什么?

- Google 率先提出了 MapReduce 用来应对数据处理需求的系统, 
   Doug Cutting 领导开发了一个开源版本的 MapReduce, 称为 Hadoop.
- Hadoop是一个开源框架,可编写和运行分布式应用处理大规模数据;

   - 方便: Hadoop运行在由一般商用机器构成的大型集群上,或者如Amazon弹性计算云(EC2)等云服务器之上;
   - 健壮: Hadoop致力于在一般商用硬件上运行,其架构假设硬件会频繁地出现失效.它可以从容地处理大多数此类故障;
   - 可扩展: Hadoop通过增加集群节点,可以线性地扩展以处理更大的数据集;
   - 简单: Hadoop允许用户快速编写出高效的并行代码;

- Hadoop集群

   - Hadoop集群是在同一地点用网络互联的一组通用机器.数据存储和处理都发生在这个机器"云"中.不同的用户可以从独立的客户端提交计算作业到Hadoop,这些客户端可以是远离Hadoop集群的个人台式机;

- 通常在一个 Hadoop 集群中的机器都是相对同构的 x86 Linux 服务器.
   而且它们几乎位于同一个数据中心,并且通常在同一组机架里;
- Hadoop 强调把代码向数据迁移,而不是相反的过程;
- Hadoop 的集群环境中既包含数据又包含计算环境, 客户端仅需发送待执行的 MapReduce 程序, 
   而这些程序一般都很小(通常为几千字节).

### 分布式存储、分布式计算

### Hadoop构造模块

在一个全配置的集群上, "运行Hadoop"意味着在网络分布的不同服务器上运行一组守护进程(daemons). 
这些守护进程有着特殊的角色, 一些仅存在单个服务器上, 一些则运行在多个服务器上: 

- NameNode(名字节点)
- DataNode(数据节点)
- Secondary NameNode(次名字节点)
- JobTracker(作业跟踪节点)
- TaskTracker(任务跟踪节点)

#### NameNode

- Hadoop在分布式计算机与分布式存储中都采用了 **主/从(master/slave)** 结构; 
- 分布式存储系统被称为Hadoop文件系统, 或简称HDFS; 
- **NameNode** 位于HDFS的主端, 它指导 **从端的DataNode** 执行底层的I/O任务; 
  NameNode是HDFS的书记员, 他跟踪文件如何被分割成文件块, 而这些块又被哪些节点存储, 
  以及分布式文件系统的整体运行状态是否正常; 
- 运行NameNode会消耗大量的内存和I/O资源, 因此, 为了减轻机器的负载, 
  驻留在NameNode的服务器通常不会存储用户数据或者执行MapReduce程序的计算任务, 
  这意味着NameNode服务器不会同时是DataNode或者TaskTracker; 

#### JobTracker

 - JobTracker守护进程是应用程序和Hadoop之间的纽带; 每个Hadoop集群只有一个JobTracker守护进程, 通常位于服务器集群的主节点上; 
 - 一旦提交代码到集群上, JobTracker就会确定执行计划, 包括决定处理哪些文件、为不同的任务分配节点以及监控所有任务的运行; 
 - 如果任务失败, JobTracker将自动重启任务, 但分配的节点可能会不同, 同时受到预定义的重试次数限制; 

## Hadoop集群安装

## Hadoop运行

# HDFS文件操作

- HDFS是一种文件系统, 专为MapReduce这类框架下的大规模分布式数据处理设计, 
   可以把一个大数据集在HDFS中存储位单个文件, 而大多数其它的文件系统无力实现这一点. 
- HDFS并不是一个天生的Unix文件系统, 不支持标准的Unix文件命令和操作; 
  Hadoop提供了一套与Linux文件命令类似的命令行工具, 即Hadoop操作文件的shell命令, 
  它们是与HDFS系统的主要借口; 
- 一个典型的Hadoop工作流会:  
   - 在别的地方生成数据文件(如日志文件),再将其复制到HDFS中 
   - 接着由MapReduce程序处理这个数据, 但它们通常不会直接读任何一个HDFS文件, 
      相反, 它们依靠MapReduce框架来读取HDFS文件, 并将其解析为独立的记录(键值对),
      这些记录才是MapReduce程序所处理的数据单元, 除非需要定制数据的导入与导出, 
      否则几乎不必编程来读写HDFS文件; 

## Hadoop基本文件命令

1. 添加目录和文件
2. 获取文件
3. 删除文件

### 指定文件和目录确切位置的URI

- Hadoop的文件命令既可以与HDFS文件系统交互, 也可以和本地文件系统交互; 
- URI精确地定位一个特定文件或目录的位置, 完整的URI格式为 `scheme//authority/path`
- `scheme` 类似于一个协议, 它可以是 `hdfs` 或 `file` , 分别指定HDFS文件系统或本地文件系统; 
- `authority` 是HDFS中NameNode的主机名; 
- `path` 是文件或目录的路径; 
- 例如, 对于在本地机器的9000端口上, 以标准伪分布式模型运行的HDFS, 访问用户目录/user/chunk中文件example.txt的URI:  `hdfs://localhost:9000/user/chuck/example.txt`
- 大多数设置不需要制定URI中的 `scheme://authority` 部分; 
- 当在本地文件系统和HDFS之间复制文件时, Hadoop中的命令会分别吧本地文件熊作为源和目的, 而不需要制定scheme为file; 
- 对于其他命令, 如果未设置URI中的 `scheme://authority` , 就会采用Hadoop的默认设置; 
   - 假如 `conf/core-site.xml` 文件已经更改为伪分布式配置, 则文件中的 `fs.default.name` 属性应为 `hdfs://localhost:9000` . 在此配置下, URI `hdfs://localhost:9000/user/chuck/example.txt` 可以缩短为 `/user/chuck/example.txt`
   - HDFS默认当前工作目录为 `/user/$USER` , 其中 `$USER` 是登录用户名, 如果作为 `chuck` 登录, 则URI `hdfs://localhost:9000/user/chuck/example.txt` 就缩短为 `example` ; 

### 基本形式

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

   - `cmd` : 特定的文件命令; 
   - `<args>` : 一个数目可变的参数; 

### 基本文件命令

1.查看帮助

```bash
$ hadoop dfs -help <cmd>
```

2.查看目录

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

3.创建、删除文件夹

- 删除一般是删除到 `.Trash` 中, 一般有一定的时效清空的, 如果误删可以找回; 

```bash
# 创建
$ hadoop dfs -mkdir </hadoop dir path/dir name>
$ hadoop dfs -mkdir -p </hadoop dir path/dir name>

# 删除
$ hadoop dfs -rmr <hadoop dir path>
```

4.创建、删除空文件

- 删除一般是删除到 `.Trash` 中, 一般有一定的时效清空的, 如果误删可以找回; 

```bash
# 创建
$ hadoop dfs -touchz </hadoop file path/file name>

# 删除
$ hadoop dfs -rm <hadoop file path>
```

5.检索文件

- 复制; 
- 移动或重命名; 
- 从HDFS中下载文件到本地文件系统; 
- 从本地文件系统复制文件到HDFS中; 

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

6.查看文件内容

```bash
$ hadoop dfs -cat <hadoop file path>
$ hadoop dfs -text <hadoop file path>

$ hadoop dfs -tail <hadoop file path>
$ hadoop dfs -tail -f <hadoop file path>
```

7.查看文件、文件夹大小

```bash
# 字节为单位展示
$ hadoop dfs -du <hadoop file path>

# GB为单位展示
$ hadoop dfs -du -s -h <hadoop file path>

# 查看文件夹下每个文件大小
$ hadoop dfs -du -s -h <hadoop dir path/*>
```

8.判断文件、目录、大小

```bash
# 检查文件是否存在, 存在返回0
$ hadoop dfs -test -e filename

# 检查文件是否是0字节, 是返回0
$ hadoop dfs -test -z filename

# 检查文件是否是目录, 是返回1, 否则返回0
$ hadoop dfs -test -d filename
```

## yarn命令行

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

# MapReduce程序

## MapReduce 程序通过操作键值对来处理数据

一般形式:

```
map: (K1, V1) -> (K2, V2)
reduce: (K2, list(V2)) -> (K3, V3)
```

高阶视图:

## Hadoop 数据类型


## Mapper

一个类要作为mapper,需要继承MapReduceBase基类,并且实现Mapper接口;
MapReduceBase基类包含了类的构造与解构方法;

- void configure (JobConf job)
   - 该函数提取XML配置文件或应用程序主类中的参数,在数据处理之前调用该函数;
- void close()
   - 作为map任务结束前的最后一个操作,该函数完成所有的结尾工作,如关闭数据库连接,关闭文件等;

Mapper接口负责数据处理阶段. 它采用的形式为 `Mapper<K1, V1, K2, V2>` 的 java 泛型, 
这里的键类和值类分别实现 `WritableComparable` 和 `Writable` 接口. 
Mapper 只有一个方法: `map`,用于处理一个单独的键/值对:

```java
void map(
   K1 key, 
   V1 value,
   OutputCollector<K2, V2> output,
   Reporter reporter
) throws IOException
```

- 该函数处理一个给定的键/值对(K1, V1),生成一个键/值对(K2,V2)的列表(该列表可能为空);
- OutputCollector接收上面的映射过程的输出;
- Reporter可提供对mapper相关附加信息的记录,形成任务进度;

Hadoop提供了一些有用的mapper实现:

   - `IdentityMapper<K, V>`
      - 实现Mapper<K, V, K, V>, 将输入直接映射到输出
   - `InverseMapper<K, V>`
      - 实现Mapper<K, V, K, V>, 反转键/值对
   - `RegexMapper<K>`
      - 实现Mapper<K, Text, Text, LongWritable>,为每个常规表达式的匹配项生成一个(match, 1)对
   - `TokenCountMapper<K>`
      - 实现Mapper<K, Text, Text, LongWritable>,当输入的值为分词时,生成一个(token, 1)对

## Reducer

一个类要作为reducer,需要继承MapReduceBase基类,
允许配置和清理.并且实现Reducer接口;
MapReduceBase基类包含了类的构造与解构方法;

Reducer只有一个方法: `reduce`:

```java
void reduce(
   K2 key,
   Iterator<V2> values,
   OutputCollector<K3, V3> output,
   Reporter reporter,
) throws IOException
```

- 当 reducer 任务接收来自各个 mapper 的输出时,它按照键/值对中的键对输入数据进行排序,
  并将相同键的值归并. 然后调用 reduce() 函数, 并通过迭代处理那些与指定键相关联的值, 
  生成一个(可能为空)的列表 (K3,V3).
- OutputCollecotr 接收 reduce 阶段的输出, 并写入输出文件;
- Reporter 可提供对 reducer 相关附加信息的记录, 形成任务进度;

Hadoop提供了一些基本的reducer实现

- `IdentityReducer<K, V>`
  - 实现Reducer<K, V, K, V>,将输入直接映射到输出
- `LongSumReducer<K>`
  - 实现<K, LongWritable, K, LongWritable>,计算与给定键相对应的所有值的和

## Partitioner: 重定向Mapper的输出

虽然将 Hadoop 程序称为 MapReduce 应用, 但是在 map 和 reduce 两个阶段之间还有一个及其重要的步骤:将 mapper 的结果输出给不同的reducer.这就是partitioner的工作.
一个MapReduce应用需要使用多个reducer,但是,当使用多个reducer时,就需要采取一些办法来确定mapper应该把键值对输出给谁.

- 默认的做法是对键进行散列来确定 reducer.Hadoop通过 `HashPartitioner` 类强制执行这个策略.但有时HashPartitioner会让你出错.
- 一个定制的partitioner只需要实现 `configure()` 和 `getPartition()` 两个函数.

  - configure()将Hadoop对作业的配置应用在partition上;
  - getPartition()返回一个介于0和reduce任务数之间的整数,指向键/值对将要发送到的reducer;

在 map 和 reduce 阶段之间,一个 MapReduce 应用必然从 mapper 任务得到输出结果,
并把这些结果发布给 reducer 任务. 该过程通常被称为洗牌(shuffling), 
因为在单节点上的 mapper 输出可能被送往分布在集群多个节点上的 reducer.

## Combiner: 本地reduce

在许多MapReduce应用场景中,不妨在分发mapper结果之前做一下"本地Reduce".

预定义Mapper和Reducer类的WordCount:

```java
public class WordCount2 {
   public static void main(String[] args) {
      JobClient client = new JobClient();
      JobConf conf = new JobConf(WordCount2.class);

      FileInputFormat.addInputPath(conf, new Path(args[0]));
      FileOutputFormat.setOutputPath(conf, new Path(args[1]));

      conf.setOutputKeyClass(Text.class);
      conf.setOutputValueClass(LongWritable.class);
      conf.setMapperClass(TokenCountMapper.class);
      conf.setCombinerClass(LongSumReducer.class);
      conf.setReducerClass(LongSumReducer.class);

      client.setConf(conf);
      try {
         JobClient.runJob(conf);
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
```

# 读/写
